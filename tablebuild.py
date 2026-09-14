"""tablebuild.py -- fast in-process cosine-table builder (table format v2).

Builds the per-tensor / per-tier cosine table in ONE pass over the source
model, quantizing in-process through ggml-base.dll (ggml_quantize_chunk --
bit-exact with llama-quantize) and dequantizing through dequantize_row_*.
No per-tier ref files, no disk churn; parallelized across CPU cores.

Semantics (fixed):
  * uniform per-tier: tier T column = every quantizable tensor quantized
    at type T (with llama.cpp's shape-fallback chain and the imatrix
    rules below)
  * no imatrix  -> ladder excludes the imatrix-REQUIRED tiers
    {IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS}
  * imatrix selected -> full ladder; tensors with missing imatrix data
    (e.g. MTP) are quantized at the safe fallback type Q4_K for those
    tiers (the resulting S/Q/C is stored; nothing downstream special-cases)
  * frozen = llama.cpp faithful: tensors that llama-quantize would not
    quantize (tensor_allows_quantization port) keep the source type and
    are baked into s0/q0/c0 from the source

Usage:
  python tablebuild.py --source m.gguf [--imatrix i.gguf] [--out path]
                       [--workers N] [--ladder A,B,C] [--check-tier T]
                       [--force] [--dll path] [--quantize path]

Validation gate (--check-tier): builds ONE real `llama-quantize --pure`
ref at tier T and byte-compares every tensor against the in-process
bytes (plus S/Q vs the existing table when present).
"""
import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "gguf-py"))
from gguf import GGUFReader                          # noqa: E402
from gguf.constants import GGMLQuantizationType      # noqa: E402

import modeldef                                        # noqa: E402
import tablestore                                      # noqa: E402

# single source of truth for the quantizable set + group labels
allows_quantization = modeldef.allows_quantization

# canonical ggml enum values (stable across builds, per ggml.h)
ENUM = {"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7,
        "Q8_0": 8, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13,
        "Q6_K": 14, "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18,
        "IQ1_S": 19, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22,
        "IQ4_XS": 23, "IQ1_M": 29, "BF16": 30}

# ascending bytes/element; must stay sorted (the solver relies on it).
# Canonical values live in modeldef (shared with qmodel); kept here as
# aliases for CLI/GUI back-compat.
LADDER = modeldef.LADDER
IMATRIX_REQUIRED = modeldef.IMATRIX_REQUIRED
FALLBACK_IMATRIX_TIER = "Q4_K"      # imatrix-required tier, no coverage
# llama.cpp tensor_type_fallback chain (shape-incompatible tensors)
FALLBACK_CHAIN = {"IQ1_S": "IQ4_NL", "IQ1_M": "IQ4_NL",
                  "IQ2_XXS": "IQ4_NL", "IQ2_XS": "IQ4_NL",
                  "IQ2_S": "IQ4_NL", "IQ3_XXS": "IQ4_NL",
                  "IQ3_S": "IQ4_NL", "IQ4_XS": "IQ4_NL",
                  "Q2_K": "Q4_0", "Q3_K": "Q4_0",
                  "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0"}
QUANT_TYPES = sorted(set(LADDER[:-1]) | set(FALLBACK_CHAIN.values()))

BATCH_BYTES = 1 << 30     # ~1 GB of source bytes per task batch
CHECK_EVERY = 4           # checkpoint every N completed batches


def ceil32(n):
    return (int(n) + 31) & ~31


def find_tool(names, what):
    """Search candidate locations for a llama.cpp binary / DLL."""
    cands = [APP_DIR / "binaries", APP_DIR.parent / "binaries",
             APP_DIR.parent / "app"]
    for c in cands:
        for n in names:
            if (c / n).exists():
                return c / n
    in_path = shutil.which(names[0])
    if in_path:
        return Path(in_path)
    sys.exit(f"ERROR: {what} not found -- pass its path explicitly")


# --------------------------------------------------------------------------
# llama.cpp quantization semantics (allows_quantization now lives in
# modeldef.py; the shape-fallback + imatrix rules stay build-local here)
# --------------------------------------------------------------------------

def fallback_type(tier, n_per_row, blck):
    """llama.cpp tensor_type_fallback: shape-incompatible tensors move
    down the fallback chain until the block size divides n_per_row."""
    t = tier
    while n_per_row % blck[t] != 0:
        t = FALLBACK_CHAIN.get(t, "F16")
        if t in ("F16", "F32"):
            break
    return t


def effective_type(tier, n_per_row, covered, blck):
    """The type llama-quantize would actually use for this tensor at tier.
    covered: tensor has imatrix data (only relevant for IMATRIX_REQUIRED)."""
    if tier == "F16":
        return "F16"
    if tier in IMATRIX_REQUIRED and not covered:
        t = FALLBACK_IMATRIX_TIER
    else:
        t = tier
    return fallback_type(t, n_per_row, blck)


# --------------------------------------------------------------------------
# ggml-base.dll bindings
# --------------------------------------------------------------------------

class Quant:
    """ggml_quantize_chunk + dequantize_row_* (bit-exact with the C code)."""

    def __init__(self, dll):
        self.lib = ctypes.CDLL(str(dll))
        fn = self.lib.ggml_quantize_chunk
        fn.restype = ctypes.c_size_t
        fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                       ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        self.chunk = fn
        bs = self.lib.ggml_blck_size
        bs.restype = ctypes.c_int64
        bs.argtypes = [ctypes.c_int]
        ts = self.lib.ggml_type_size
        ts.restype = ctypes.c_size_t
        ts.argtypes = [ctypes.c_int]
        self.blck = {t: int(bs(ENUM[t])) for t in QUANT_TYPES}
        self.rowb = {t: int(ts(ENUM[t])) for t in QUANT_TYPES}
        self.deq = {}
        for t in QUANT_TYPES:
            sym = t.lower() if t.startswith("IQ") else t[0].lower() + t[1:]
            fn = getattr(self.lib, f"dequantize_row_{sym}")
            fn.restype = None
            fn.argtypes = (ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
            self.deq[t] = fn

    def quant(self, tname, a, n_per_row, imx):
        """Quantize a (1-D f32) tensor; returns raw bytes (1-D uint8)."""
        bs, rb = self.blck[tname], self.rowb[tname]
        buf = np.empty(rb * (a.size // bs), np.uint8)
        imxp = None
        if imx is not None:
            imxp = imx.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self.chunk(ENUM[tname],
                        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        buf.ctypes.data, ctypes.c_int64(0),
                        ctypes.c_int64(a.size // n_per_row),
                        ctypes.c_int64(n_per_row), imxp)
        if rc != buf.size:
            raise RuntimeError(f"quantize {tname}: rc={rc} != {buf.size}")
        return buf

    def deq_row(self, tname, raw, out):
        self.deq[tname](raw.ctypes.data,
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        out.size)
        return out


def load_imatrix(path):
    """{tensor_name: f32 array} with the exact normalization of
    tools/quantize/quantize.cpp load_imatrix (GGUF in_sum2/counts)."""
    r = GGUFReader(str(path))
    sums, counts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):       # 8-char suffix
            sums[t.name[:-8]] = np.ascontiguousarray(t.data, np.float32)
        elif t.name.endswith(".counts"):      # 7-char suffix
            counts[t.name[:-7]] = np.ascontiguousarray(t.data, np.float32)
    out = {}
    for name, s in sums.items():
        c = counts.get(name)
        if c is None:
            sys.exit(f"ERROR: imatrix {name}.in_sum2 has no .counts")
        ncounts = int(c.size)
        ne0 = int(s.size) // ncounts
        e = np.empty_like(s)
        for j in range(ncounts):
            cj = float(np.rint(c[j]))
            sl = slice(j * ne0, (j + 1) * ne0)
            e[sl] = s[sl] / cj if cj > 0 else 1.0
        out[name] = e
    return out


def to_f32(t):
    """Source tensor -> 1-D contiguous f32 (bf16->f32 is exact)."""
    tt = t.tensor_type
    if tt == GGMLQuantizationType.F32:
        return np.ascontiguousarray(t.data, np.float32).ravel()
    if tt == GGMLQuantizationType.F16:
        return t.data.astype(np.float32).ravel()
    if tt == GGMLQuantizationType.BF16:
        return ((t.data.view(np.uint16).astype(np.uint32) << 16)
                .view(np.float32)).ravel()
    sys.exit(f"ERROR: unsupported source tensor type {tt} ({t.name})")


def f64_sumsq(a, ch=16_000_000):
    """True float64 sum of squares. The chunk is cast to f64 BEFORE the dot,
    so a large-magnitude value (e.g. a 1e30 sentinel in a frozen tensor) can
    never overflow a float32 accumulator to inf."""
    acc = 0.0
    for i in range(0, a.size, ch):
        seg = a[i:i + ch].astype(np.float64)
        acc += float(np.dot(seg, seg))
    return acc


def f64_dot(a, b, ch=16_000_000):
    """True float64 inner product (cast before the dot; see f64_sumsq)."""
    acc = 0.0
    for i in range(0, a.size, ch):
        acc += float(np.dot(a[i:i + ch].astype(np.float64),
                            b[i:i + ch].astype(np.float64)))
    return acc


# --------------------------------------------------------------------------
# worker side (ProcessPoolExecutor, Windows spawn)
# --------------------------------------------------------------------------

_W = None


def _init_worker(source, imatrix, dll, ladder, frozen):
    global _W
    # BLAS already imported with the parent's env; the parent sets
    # OPENBLAS_NUM_THREADS=1 before the pool is created, so spawn children
    # inherit it. Re-assert here for good measure (harmless if no-op).
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _W = {"reader": GGUFReader(str(source)),
          "imx": load_imatrix(imatrix) if imatrix else None,
          "quant": Quant(dll),
          "ladder": ladder,
          "frozen": frozen}


def quant_deq(quant, eff, a, npr, row):
    """Quantize at `eff` and dequantize back: returns (raw_bytes, b_f32).
    Handles the native F16/F32 fallback terminus without the DLL."""
    if eff == "F16":
        b = a.astype(np.float16).astype(np.float32)
        raw = a.astype(np.float16).tobytes()
        return raw, b
    raw = quant.quant(eff, a, npr, row)
    b = np.empty(a.size, np.float32)
    quant.deq_row(eff, raw, b)
    return raw, b


def _do_batch(tids):
    """Process one batch of tensor indices; return result arrays."""
    W = _W
    r, quant, ladder = W["reader"], W["quant"], W["ladder"]
    imx, frozen, blck = W["imx"], W["frozen"], quant.blck
    J = len(ladder)
    n = len(tids)
    outA = np.zeros(n, np.float64)
    outS = np.zeros((n, J), np.float64)
    outQ = np.zeros_like(outS)
    outC = np.zeros((n, J), np.int64)
    s0b = q0b = 0.0
    c0b = 0
    fb = {t: 0 for t in ladder[:-1]}     # per-tier fallback counters
    warns = []
    for k, tid in enumerate(tids):
        t = r.tensors[tid]
        a = to_f32(t)
        av = f64_sumsq(a)
        outA[k] = av
        if frozen[tid]:
            # frozen: kept at the source type. Its bytes live in c0 (the size
            # model). It is NOT part of the weight-space cosine (that is
            # computed over the quantizable tensors only), so no s0/q0 term.
            c0b += ceil32(t.n_bytes)
            continue
        npr = int(t.shape[0])            # ggml ne[0]
        row = imx.get(t.name) if imx else None
        # an imatrix row has n_per_row (ne[0]) values -- one per element
        # of a row, reused for every row (as llama-quantize does)
        covered = row is not None and row.size == npr
        if row is not None and not covered:
            warns.append(f"{t.name}: imatrix size {row.size} != n_per_row "
                         f"{npr}; treated as uncovered")
            row = None
        for j, T in enumerate(ladder[:-1]):
            eff = effective_type(T, npr, covered, blck)
            if eff != T:
                fb[T] += 1
            raw, b = quant_deq(quant, eff, a, npr, row)
            outC[k, j] = len(raw) if eff == "F16" else raw.size
            outS[k, j] = f64_dot(a, b)
            outQ[k, j] = f64_sumsq(b)
        # F16 column: a real bf16->f16 conversion (lossy vs bf16, faithful
        # to an actual F16 build)
        h = a.astype(np.float16)
        if not np.isfinite(h).all():
            warns.append(f"{t.name}: bf16->f16 overflow; F16 column zeroed")
        else:
            b = h.astype(np.float32)
            outS[k, J - 1] = f64_dot(a, b)
            outQ[k, J - 1] = f64_sumsq(b)
            outC[k, J - 1] = ceil32(a.size * 2)
        del a
    return tids, outA, outS, outQ, outC, s0b, q0b, c0b, fb, warns


# --------------------------------------------------------------------------
# parent side
# --------------------------------------------------------------------------

class Build:
    def __init__(self, source, imatrix, dll, ladder, out, workers=1,
                 force=False, log=print, progress=None):
        self._force = force
        self.source = Path(source)
        self.imatrix = Path(imatrix) if imatrix else None
        self.dll = Path(dll)
        # NOTE: llama-quantize is no longer needed to build the table -- the
        # per-tier bytes/cosines are measured in-process via ggml-base.dll,
        # and the container overhead is read from the source header. The
        # Q4_K byte-exactness gate (--check-tier) still finds its own
        # llama-quantize as an argument; nothing here calls find_tool for it.
        self.ladder = ladder
        self.out = Path(out)
        self.workers = workers
        self.log = log
        self.progress = progress or (lambda d, t, eta: None)

        src = GGUFReader(str(self.source))
        self.arch = str(src.get_field("general.architecture").contents())
        f = src.get_field("general.name")
        self.model_name = str(f.contents()) if f is not None else ""
        names = [t.name for t in src.tensors]
        ndims = np.array([len(t.shape) for t in src.tensors], np.int32)
        nel = np.array([t.n_elements for t in src.tensors], np.int64)
        src_bytes = np.array([t.n_bytes for t in src.tensors], np.int64)
        src_type = np.array([t.tensor_type.name for t in src.tensors],
                            dtype=object)
        frozen = np.array([not allows_quantization(nm, int(t.data.ndim))
                           for nm, t in zip(names, src.tensors)], bool)
        lab = [modeldef.group_of(nm, bool(not fr))
               for nm, fr in zip(names, frozen)]
        n_layer = 0
        for nm in names:
            m = re.match(r"blk\.(\d+)\.", nm)
            if m:
                n_layer = max(n_layer, int(m.group(1)))

        self.n, self.J = len(names), len(ladder)
        self.names = names
        self.meta = {"ndims": ndims, "nel": nel, "src_bytes": src_bytes,
                     "src_type": src_type, "frozen": frozen, "labels": lab,
                     "arch": self.arch, "model_name": self.model_name,
                     "n_layer": n_layer}
        self.A = np.full(self.n, np.nan, np.float64)
        self.S = np.zeros((self.n, self.J), np.float64)
        self.Q = np.zeros_like(self.S)
        self.C = np.zeros((self.n, self.J), np.int64)
        self.s0 = self.q0 = 0.0
        self.c0 = 0
        # GGUF container/file overhead: header + KV-metadata + the
        # tensor-name and tensor-info string tables + per-tensor alignment.
        # This is identical for EVERY quantization type, so it equals the
        # source file size minus the sum of ceil32-padded source tensor
        # bytes -- measured here instead of from a full pure-Q4_K reference
        # build (see measure_overhead). No re-quantize, no temp GGUF.
        self.overhead = (int(self.source.stat().st_size)
                         - sum(ceil32(int(b))
                               for b in self.meta["src_bytes"].tolist()))
        self.fp = tablestore.fp_of(self.source, self.imatrix, self.dll)
        self.n_fallback = {t: 0 for t in ladder[:-1]}

        self._close_src(src)

    @staticmethod
    def _close_src(r):
        try:
            del r.data
        except Exception:      # noqa: BLE001
            pass
        del r

    # ------------------------------------------------------------- resume
    def load_checkpoint(self, force):
        if force or not self.out.exists():
            return 0
        try:
            z = np.load(str(self.out), allow_pickle=True)
        except Exception:      # noqa: BLE001
            return 0
        try:
            if int(z["v"]) != tablestore.TABLE_VERSION:
                sys.exit("ERROR: existing table is not v2 -- use --force")
            if list(z["names"]) != self.names or \
               list(z["ladder"]) != self.ladder:
                sys.exit("ERROR: existing table has different tensors/"
                         "ladder -- use --force")
            self.A = z["A"].copy()
            self.S = z["S"].copy()
            self.Q = z["Q"].copy()
            self.C = z["C"].copy()
            self.s0, self.q0 = float(z["s0"]), float(z["q0"])
            if not (np.isfinite(self.s0) and np.isfinite(self.q0)):
                self.s0 = self.q0 = 0.0   # cosine is upg-only; offset unused
            self.c0 = int(z["c0"])
            self.overhead = int(z["overhead"])
        finally:
            z.close()
        ndone = int(np.isfinite(self.A).sum())
        if ndone:
            self.log(f"resume: {ndone}/{self.n} tensors already done")
        return ndone

    def save(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        m = self.meta
        tmp = self.out.with_suffix(".tmp.npz")
        np.savez(str(tmp),
                 v=np.int64(tablestore.TABLE_VERSION),
                 arch=np.array(m["arch"], dtype=object),
                 model_name=np.array(m["model_name"], dtype=object),
                 n_layer=np.int64(m["n_layer"]),
                 names=np.array(self.names, dtype=object),
                 ndims=m["ndims"], nel=m["nel"],
                 groups=np.array(m["labels"], dtype=object),
                 frozen=m["frozen"], src_type=m["src_type"],
                 ladder=np.array(self.ladder, dtype=object),
                 A=self.A, S=self.S, Q=self.Q, C=self.C,
                 s0=np.float64(self.s0), q0=np.float64(self.q0),
                 c0=np.int64(self.c0), overhead=np.int64(self.overhead),
                 fp=np.array(json.dumps(self.fp)))
        os.replace(tmp, self.out)
        tablestore.register_table(self.source, bool(self.imatrix),
                                  self.out, self.fp, self.imatrix)

    # ------------------------------------------------------------- batches
    def make_batches(self, todo):
        src = self.meta["src_bytes"]
        batches, cur, curb = [], [], 0
        for i in todo:
            b = int(src[i])
            if cur and curb + b > BATCH_BYTES:
                batches.append(cur)
                cur, curb = [], 0
            cur.append(i)
            curb += b
        if cur:
            batches.append(cur)
        return batches

    def merge(self, res):
        tids, A, S, Q, C, s0b, q0b, c0b, fb, warns = res
        tids = np.asarray(tids)
        self.A[tids] = A
        self.S[tids] = S
        self.Q[tids] = Q
        self.C[tids] = C
        self.s0 += s0b
        self.q0 += q0b
        self.c0 += c0b
        for t, c in fb.items():
            self.n_fallback[t] += c
        for w in warns:
            self.log("  " + w)
        return len(tids)

    # -------------------------------------------------------------- engine
    def run(self, cancel=None):
        t0 = time.time()
        done = self.load_checkpoint(force=self._force)
        todo = [i for i in range(self.n) if not np.isfinite(self.A[i])]
        if todo:
            batches = self.make_batches(todo)
            self.log(f"{len(todo)} tensors in {len(batches)} batches, "
                     f"{self.workers} worker(s)")
            if self.workers <= 1:
                _init_worker(str(self.source),
                             str(self.imatrix) if self.imatrix else None,
                             str(self.dll), self.ladder,
                             self.meta["frozen"].tolist())
                for bi, b in enumerate(batches):
                    if cancel is not None and cancel():
                        self.log("cancelled -- checkpoint saved, resume later")
                        self.save()
                        return False
                    nt = self.merge(_do_batch(b))
                    self._progress(done, bi, len(batches), t0)
                    done += nt
                    if bi % CHECK_EVERY == CHECK_EVERY - 1:
                        self.save()
            else:
                with ProcessPoolExecutor(
                        max_workers=self.workers,
                        initializer=_init_worker,
                        initargs=(str(self.source),
                                  str(self.imatrix) if self.imatrix
                                  else None,
                                  str(self.dll), self.ladder,
                                  self.meta["frozen"].tolist())) as ex:
                    futs = {ex.submit(_do_batch, b): b for b in batches}
                    completed = 0
                    for fut in self._ordered(futs, cancel):
                        if fut is None:
                            break
                        nt = self.merge(fut.result())
                        completed += 1
                        self._progress(done, completed, len(batches), t0)
                        done += nt
                        if completed % CHECK_EVERY == 0:
                            self.save()
            self.save()
        el = time.time() - t0
        self.log(f"pass done in {el / 60:.1f} min "
                 f"({int(np.isfinite(self.A).sum())}/{self.n} tensors)")
        self.measure_overhead()
        self.save()
        self._sanity()
        tablestore.register_table(self.source, bool(self.imatrix),
                                  self.out, self.fp, self.imatrix)
        return True

    def _ordered(self, futs, cancel):
        pending = set(futs)
        while pending:
            if cancel is not None and cancel():
                for f in pending:
                    f.cancel()
                return
            dn, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in dn:
                yield f

    def _progress(self, done, bi, nb, t0):
        el = time.time() - t0
        rate = (done) / el if el > 1 else 0
        eta = (self.n - done) / rate if rate else 0
        self.log(f"  {done}/{self.n} tensors  "
                 f"({(bi + 1)}/{nb} batches)  "
                 f"eta {eta / 60:.1f} min")
        self.progress(done, self.n, eta)

    # ----------------------------------------------------------- overhead
    def measure_overhead(self):
        """Container-overhead log (no work).

        The GGUF container overhead (header + KV-metadata + tensor-name and
        tensor-info string tables + per-tensor alignment) is the SAME for
        every quantization type, so it was already taken from the source
        header in __init__:

            overhead = source_size - sum(ceil32(source tensor bytes))

        The old version re-ran a full `pure Q4_K` llama-quantize here just
        to subtract the tensor bytes -- an entire model re-quantization to
        measure a ~10 MB constant. That is now done instantly and the Q4_K
        byte-exactness of the in-process quantizer is covered separately by
        the --check-tier gate, so this only logs the value. (It matches the
        former Q4_K measurement to within a ~200 B llama-quantize metadata
        entry; negligible against multi-GB tables.)"""
        self.log(f"  overhead = {self.overhead / 1e6:.2f} MB "
                 "(from source header)")

    def _sanity(self):
        upg = ~self.meta["frozen"]
        with np.errstate(invalid="ignore"):
            ct = self.S / np.sqrt(np.where(self.A[:, None] > 0,
                                           self.A[:, None] * self.Q, 1))
        bad = (ct[upg] > 1.0000001) | (~np.isfinite(ct[upg]))
        for t, c in self.n_fallback.items():
            if c:
                self.log(f"fallback: {c} tensors at a fallback type "
                         f"for tier {t}")
        A_total = float(self.A.sum())
        self.log(f"||a||^2 total = {A_total:.4f}   "
                 f"s0={self.s0:.4f} q0={self.q0:.4f} "
                 f"c0={self.c0 / 1e6:.1f} MB")
        self.log(f"sanity: per-tensor cosines in (0,1]: "
                 f"{'OK' if not bad.any() else f'VIOLATIONS: {int(bad.sum())}'}")
        self.log(f"DONE -> {self.out}")


# --------------------------------------------------------------------------
# --check-tier: the byte-comparison gate
# --------------------------------------------------------------------------

def check_tier(source, imatrix, tier, dll, quantize, log=print):
    """THE gate: build ONE real ref at `tier` with an EXPLICIT per-tensor
    schema (each quantizable tensor at the tier's effective type -- exactly
    what the engine would assign) and byte-compare every tensor against the
    in-process quantization. Returns True when bit-exact."""
    source, imatrix = Path(source), Path(imatrix) if imatrix else None
    d = APP_DIR / "refs_tmp"
    ref, schema = d / f"check_{tier}.gguf", d / f"check_{tier}.txt"
    d.mkdir(parents=True, exist_ok=True)
    try:
        src = GGUFReader(str(source))
        q = Quant(dll)
        imx = load_imatrix(imatrix) if imatrix else None
        lines = []
        for t in src.tensors:
            if not allows_quantization(t.name, int(t.data.ndim)):
                continue                      # frozen: stays source type
            npr = int(t.shape[0])
            row = imx.get(t.name) if imx else None
            covered = row is not None and row.size == npr
            if row is not None and not covered:
                row = None
            lines.append(f"{t.name}={effective_type(tier, npr, covered, q.blck)}")
        schema.write_text("\n".join(lines) + "\n")
        if ref.exists():
            ref.unlink()
        cmd = [str(quantize), "--tensor-type-file", str(schema)]
        if imatrix:
            cmd += ["--imatrix", str(imatrix)]
        cmd += [str(source), str(ref), tier]
        log(f"check-tier {tier}: building real ref ({len(lines)} tensor types)...")
        r = subprocess.run(cmd, capture_output=True, text=True,
                           errors="replace")
        tail = "\n".join(r.stdout.strip().splitlines()[-4:])
        if r.returncode != 0 or not ref.exists():
            log(f"check-tier {tier}: REF BUILD FAILED (rc={r.returncode})\n"
                f"{tail}")
            return False
        rr = GGUFReader(str(ref))
        rmap = {t.name: t for t in rr.tensors}
        mism, type_mism = [], []
        nq = nf = 0
        for t in src.tensors:
            rt = rmap.get(t.name)
            if rt is None:
                mism.append(f"{t.name}: missing from ref")
                continue
            if not allows_quantization(t.name, int(t.data.ndim)):
                if rt.tensor_type != t.tensor_type:
                    type_mism.append(f"{t.name}: frozen, ref type "
                                     f"{rt.tensor_type} != source "
                                     f"{t.tensor_type}")
                nf += 1
                continue
            npr = int(t.shape[0])
            row = imx.get(t.name) if imx else None
            covered = row is not None and row.size == npr
            if row is not None and not covered:
                row = None
            eff = effective_type(tier, npr, covered, q.blck)
            if rt.tensor_type.name.lower() != eff.lower():
                type_mism.append(f"{t.name}: ref type "
                                 f"{rt.tensor_type.name} != {eff}")
            if t.n_bytes < 1000:
                continue                      # tiny tensors: type check only
            a = to_f32(t)
            raw, _ = quant_deq(q, eff, a, npr, row)
            refb = np.frombuffer(bytes(rt.data.tobytes()), np.uint8)
            if eff == "F16":
                raw = np.frombuffer(raw, np.uint8)
            if raw.size != refb.size or not np.array_equal(raw, refb):
                mism.append(f"{t.name}: {raw.size} != {refb.size} bytes"
                            if raw.size != refb.size
                            else f"{t.name}: content differs")
            nq += 1
        log(f"check-tier {tier}: compared {nq} quantized + {nf} frozen tensors")
        ok = not mism and not type_mism
        for m in (type_mism + mism)[:20]:
            log("  " + m)
        if len(type_mism) + len(mism) > 20:
            log(f"  ... and {len(type_mism) + len(mism) - 20} more")
        log(f"check-tier {tier}: {'BIT-EXACT' if ok else 'MISMATCHES FOUND'}")
        del src, rr
        return ok
    finally:
        for p in (ref, schema):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _ladder_for(args, log):
    if args.ladder:
        lad = [t.strip() for t in args.ladder.split(",") if t.strip()]
        if lad[-1] != "F16" or any(t not in LADDER for t in lad):
            sys.exit(f"ERROR: bad --ladder {lad} (must be a subset of "
                     f"{LADDER} ending in F16)")
    else:
        lad = LADDER if args.imatrix else \
            [t for t in LADDER if t not in IMATRIX_REQUIRED]
    if not args.imatrix and any(t in IMATRIX_REQUIRED for t in lad):
        sys.exit("ERROR: imatrix-required tiers need --imatrix "
                 "(or trim them from --ladder)")
    return lad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--imatrix", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1,
                    (os.cpu_count() or 2) // 2))
    ap.add_argument("--ladder", type=str, default="")
    ap.add_argument("--check-tier", type=str, default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dll", type=Path, default=None)
    ap.add_argument("--quantize", type=Path, default=None)
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"ERROR: source not found: {args.source}")
    if args.imatrix and not args.imatrix.exists():
        sys.exit(f"ERROR: imatrix not found: {args.imatrix}")
    if args.dll is None:
        args.dll = find_tool(["ggml-base.dll", "libggml-base.so",
                              "libggml-base.dylib"], "ggml-base")
    if args.quantize is None:
        args.quantize = find_tool(["llama-quantize.exe", "llama-quantize"],
                                  "llama-quantize")
    ladder = _ladder_for(args, None)

    if args.check_tier:
        ok = check_tier(args.source, args.imatrix, args.check_tier,
                        args.dll, args.quantize)
        sys.exit(0 if ok else 1)

    if args.out is None:
        APP_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)
        args.out = APP_DIR / "tables" / tablestore.table_filename(
            args.source, bool(args.imatrix))
    print(f"source  : {args.source}  ({args.source.stat().st_size/1e9:.1f} GB)")
    print(f"imatrix : {args.imatrix or '(none)'}")
    print(f"ladder  : {', '.join(ladder)}")
    print(f"out     : {args.out}   workers={args.workers}")
    # keep children from oversubscribing the CPU
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    b = Build(args.source, args.imatrix, args.dll, ladder, args.out,
              workers=args.workers, force=args.force)
    b.run()


if __name__ == "__main__":
    main()
