"""tablebuild2.py -- fused-kernel cosine table builder (table format v2).

The fast engine: per tensor it calls the fastq.dll fused kernel (quantize
+ dequantize + f64 S/Q/A accumulation + F16 column in ONE pass per
64M-elem segment) instead of the numpy f64 path in tablebuild.py. Parallel
across THREADS -- the ctypes call releases the GIL while the C kernel runs,
so no ProcessPool / spawn overhead, and per-thread memory is bounded by
the segment size.

Everything else is REUSED from tablebuild.Build: v2 save/resume, batches,
cheap container-overhead read from the source header (no re-quantize --
see tablebuild.Build.measure_overhead), sanity. Ladders: modeldef.LADDER_K
(fast, no imatrix) or the full LADDER (imatrix build).

The full ladder includes the 6 imatrix-REQUIRED IQ tiers. When an imatrix
is provided those tiers are measured as the REAL imatrix-weighted quants
(per-tensor imatrix row passed to the kernel); a tensor without an imatrix
row (e.g. MTP) falls back to Q4_K for them. Without an imatrix they all
measure as Q4_K -- so a full table is ALWAYS built with an imatrix (the
GUI enforces it), otherwise the 6 IQ columns are Q4_K clones and the size
bounds are wrong.

Measured cost per element (Zen 2, ref/no-imatrix path ~2x faster than the
weighted path): K ladder ~0.5 us/elem -> 27B in ~25-40 min at 12 threads;
full ladder ~4 us/elem -> ~3-5 h.

Usage:
  python tablebuild2.py --source m.gguf [--out path] [--ladder-kind k|full]
                        [--imatrix i.gguf] [--workers N] [--ladder A,B,C]
                        [--seg N] [--force] [--dll path]

  --imatrix is only used with --ladder-kind full (K builds ignore it).
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "gguf-py"))
from gguf import GGUFReader                          # noqa: E402
from gguf.constants import GGMLQuantizationType      # noqa: E402

import modeldef           # noqa: E402
import tablebuild         # noqa: E402  (Build base + helpers, reference engine)
import tablestore         # noqa: E402
import fastq              # noqa: E402

ceil32 = tablebuild.ceil32
to_f32 = tablebuild.to_f32
f64_sumsq = tablebuild.f64_sumsq
CHECK_EVERY = tablebuild.CHECK_EVERY

DEFAULT_SEG = 64_000_000     # elements per kernel call (memory cap)

_tls = threading.local()


def _get_ctx(source, dll, seg_max, imatrix):
    """Per-thread context (GGUFReader memmap + fastq handle + imatrix rows).
    Keyed on (source, dll, seg_max, imatrix) so a thread reused for a
    different build re-loads instead of serving stale data."""
    key = (source, dll, seg_max, imatrix)
    ctx = getattr(_tls, "ctx", None)
    if ctx is None or ctx.get("key") != key:
        ctx = {"key": key,
               "reader": GGUFReader(str(source)),
               "fq": fastq.FastQ(dll),
               "seg_max": seg_max,
               "imx": tablebuild.load_imatrix(imatrix) if imatrix else None}
        _tls.ctx = ctx
    return ctx


def _f32_rows(t, r0, r1):
    """Rows [r0, r1) of a 2-D source tensor -> 1-D contiguous f32."""
    d = t.data[r0:r1]
    tt = t.tensor_type
    if tt == GGMLQuantizationType.F32:
        return np.ascontiguousarray(d, np.float32).ravel()
    if tt == GGMLQuantizationType.F16:
        return d.astype(np.float32).ravel()
    if tt == GGMLQuantizationType.BF16:
        return ((d.view(np.uint16).astype(np.uint32) << 16)
                .view(np.float32)).ravel()
    sys.exit(f"ERROR: unsupported source tensor type {tt} ({t.name})")


def _do_batch(tids, source, dll, ladder, frozen, seg_max, imatrix):
    """Process one batch of tensor indices with the fused kernel; returns
    the same tuple as tablebuild._do_batch (so Build.merge works)."""
    ctx = _get_ctx(source, dll, seg_max, imatrix)
    r, fq, imx = ctx["reader"], ctx["fq"], ctx["imx"]
    blck, rowb = fq.blck, fq.rowb
    J = len(ladder)
    n = len(tids)
    outA = np.zeros(n, np.float64)
    outS = np.zeros((n, J), np.float64)
    outQ = np.zeros_like(outS)
    outC = np.zeros((n, J), np.int64)
    c0b = 0
    fb = {t: 0 for t in ladder[:-1]}
    warns = []
    for k, tid in enumerate(tids):
        t = r.tensors[tid]
        if frozen[tid]:
            a = to_f32(t)
            outA[k] = f64_sumsq(a)
            c0b += ceil32(t.n_bytes)
            continue
        npr = int(t.shape[0])             # ggml ne[0]
        nrows = int(t.n_elements) // npr
        ntot = nrows * npr
        # imatrix coverage for this tensor (llama-quantize rules): an
        # imatrix row holds npr values, one per element of a row. A tensor
        # without a matching row (e.g. MTP) has the 6 IMATRIX_REQUIRED IQ
        # tiers fall back to Q4_K -- the kernel aborts (fq_gate case 2)
        # if it is sent an imatrix-required type without an imx. WITH a
        # row, every tier is measured as the REAL quant it is (the 6 IQ
        # tiers imatrix-weighted) -- passing row=None here used to silently
        # turn all 6 of those columns into Q4_K clones.
        row = imx.get(t.name) if imx else None
        covered = row is not None and row.size == npr
        if row is not None and not covered:
            warns.append(f"{t.name}: imatrix size {row.size} != n_per_row "
                         f"{npr}; treated as uncovered")
            row = None
        effs = []
        for T in ladder[:-1]:
            eff = tablebuild.effective_type(T, npr, covered, blck)
            if eff != T:
                fb[T] += 1
            effs.append(eff)
        tnames = [e for e in effs if e != "F16"]
        Aacc = 0.0
        Sacc = [0.0] * (J - 1)
        Qacc = [0.0] * (J - 1)
        s16 = q16 = 0.0
        f16_ok = True
        rows_per_seg = max(1, seg_max // npr)
        for r0 in range(0, nrows, rows_per_seg):
            r1 = min(nrows, r0 + rows_per_seg)
            a = _f32_rows(t, r0, r1)
            A, S, Q, (s16seg, q16seg, ok16) = \
                fq.segment(a, npr, tnames, row)
            Aacc += A
            for j in range(J - 1):
                if effs[j] != "F16":
                    Sacc[j] += float(S[j])
                    Qacc[j] += float(Q[j])
            s16 += s16seg
            q16 += q16seg
            f16_ok = f16_ok and ok16
        for j, e in enumerate(effs):
            if e == "F16":
                # F16 terminus: same column values as the F16 rung
                Sacc[j] = s16
                Qacc[j] = q16
                outC[k, j] = 2 * ntot       # exact f16 bytes (no ceil32)
            else:
                outC[k, j] = (ntot // blck[e]) * rowb[e]
        if not f16_ok:
            warns.append(f"{t.name}: bf16->f16 overflow; F16 column zeroed")
            s16 = q16 = 0.0
            for j, e in enumerate(effs):
                if e == "F16":
                    Sacc[j] = Qacc[j] = 0.0
        outA[k] = Aacc
        outS[k, :J - 1] = Sacc
        outQ[k, :J - 1] = Qacc
        outS[k, J - 1] = s16
        outQ[k, J - 1] = q16
        outC[k, J - 1] = ceil32(ntot * 2)
    return list(tids), outA, outS, outQ, outC, 0.0, 0.0, c0b, fb, warns


class Build(tablebuild.Build):
    """Same table, same save/resume/overhead/sanity; threaded fused
    kernel instead of the numpy f64 path. Overhead is a cheap source-
    header read (no Q4_K re-quantize); adds wall-clock timers."""

    def __init__(self, source, imatrix, dll, ladder, out, workers=1,
                 force=False, log=print, progress=None, seg=DEFAULT_SEG):
        super().__init__(source, imatrix, dll, ladder, out, workers=workers,
                         force=force, log=log, progress=progress)
        self.seg = seg
        self.t_total = None
        self.t_pass = None

    @property
    def kind(self):
        """Table-kind dimension: 'k' (fast K ladder) or 'full'."""
        return "k" if self.ladder == modeldef.LADDER_K else "full"

    # ------------------------------------------------------- index fixup
    def save(self):
        super().save()
        # the parent's save() registers under the imatrix-only key
        # ("plain"/"im"); a K table must live under the "k" key (and must
        # not be discoverable as a plain full table), so re-register it.
        idx = tablestore._load_index()
        src = str(self.source.resolve())
        entry = idx.get(src) or idx.get(src.lower())
        if entry is not None:
            if self.kind == "k":
                if entry.get("plain") == self.out.name: 
                    del entry["plain"]
                entry["k"] = self.out.name
            else:
                entry["im" if self.imatrix else "plain"] = self.out.name
            tablestore._save_index(idx)

    # -------------------------------------------------------------- engine
    def run(self, cancel=None):
        t0 = time.time()
        done = self.load_checkpoint(force=self._force)
        todo = [i for i in range(self.n) if not np.isfinite(self.A[i])]
        if todo:
            batches = self.make_batches(todo)
            self.log(f"{len(todo)} tensors in {len(batches)} batches, "
                     f"{self.workers} thread(s), seg={self.seg // 1e6}M")
            args = (str(self.source), str(self.dll), self.ladder,
                    self.meta["frozen"].tolist(), self.seg,
                    str(self.imatrix) if self.imatrix else None)
            bt0_all = time.time()
            b_times = []
            if self.workers <= 1:
                for bi, b in enumerate(batches):
                    if cancel is not None and cancel():
                        self.log("cancelled -- checkpoint saved, "
                                 "resume later")
                        self.save()
                        return False
                    bt0 = time.time()
                    nt = self.merge(_do_batch(b, *args))
                    b_times.append(time.time() - bt0)
                    self._progress_timed(done, bi + 1, len(batches),
                                          time.time() - bt0_all, b_times)
                    done += nt
                    if bi % CHECK_EVERY == CHECK_EVERY - 1:
                        self.save()
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
                    futs = {ex.submit(_do_batch, b, *args): time.time()
                            for b in batches}
                    completed = 0
                    for fut in self._ordered(futs, cancel):
                        if fut is None:
                            break
                        bt = time.time() - futs.pop(fut)
                        b_times.append(bt)
                        nt = self.merge(fut.result())
                        completed += 1
                        self._progress_timed(done, completed, len(batches),
                                             time.time() - bt0_all, b_times)
                        done += nt
                        if completed % CHECK_EVERY == 0:
                            self.save()
            self.save()
        self.t_pass = time.time() - t0
        self.log(f"pass done in {self.t_pass / 60:.1f} min "
                 f"({int(np.isfinite(self.A).sum())}/{self.n} tensors)")
        self.measure_overhead()
        self.save()
        self._sanity()
        tablestore.register_table(
            self.source, bool(self.imatrix), self.out, self.fp,
            self.imatrix, kind=self.kind)
        self.t_total = time.time() - t0
        self.log(f"TOTAL BUILD TIME: {self.t_total:.0f} s "
                 f"({self.t_total / 60:.1f} min)")
        return True

    def _progress_timed(self, done, bi, nb, el, b_times):
        rate = done / el if el > 1 else 0
        eta = (self.n - done) / rate if rate else 0
        tail = b_times[-8:]
        avg = sum(tail) / len(tail)
        self.log(f"  {done}/{self.n} tensors  ({bi}/{nb} batches)  "
                 f"batch {b_times[-1]:.1f}s (avg8 {avg:.1f}s)  "
                 f"elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min")
        self.progress(done, self.n, eta)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--imatrix", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ladder-kind", choices=["k", "full"], default="k")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--ladder", type=str, default="")
    ap.add_argument("--seg", type=int, default=DEFAULT_SEG)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dll", type=Path, default=None)
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"ERROR: source not found: {args.source}")
    if args.imatrix and not args.imatrix.exists():
        sys.exit(f"ERROR: imatrix not found: {args.imatrix}")
    if args.dll is None:
        try:
            args.dll = tablebuild.find_tool(
                ["ggml-base.dll", "libggml-base.so", "libggml-base.dylib"],
                "ggml-base")
        except RuntimeError as e:
            sys.exit(f"ERROR: {e}")
    if args.ladder:
        lad = [t.strip() for t in args.ladder.split(",") if t.strip()]
        if lad[-1] != "F16" or any(t not in modeldef.LADDER for t in lad):
            sys.exit(f"ERROR: bad --ladder {lad} (subset of "
                     f"{modeldef.LADDER} ending in F16)")
    else:
        lad = modeldef.ladder_for(args.imatrix is not None,
                                  args.ladder_kind)
    use_im = args.imatrix is not None
    if args.ladder_kind == "k" and use_im:
        print("note: --imatrix is ignored for the K ladder "
              "(imatrix applies to the final quantize step only)")
    if args.out is None:
        tablestore.TABLES_DIR.mkdir(parents=True, exist_ok=True)
        args.out = tablestore.TABLES_DIR / tablestore.table_filename(
            args.source, use_im, args.ladder_kind)
    print(f"source  : {args.source}  "
          f"({args.source.stat().st_size / 1e9:.1f} GB)")
    print(f"imatrix : {args.imatrix or '(none)'}")
    print(f"ladder  : {', '.join(lad)}   kind={args.ladder_kind}")
    print(f"out     : {args.out}   workers={args.workers} "
          f"seg={args.seg // 1e6}M")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    b = Build(args.source, args.imatrix, args.dll, lad, args.out,
              workers=args.workers, force=args.force, seg=args.seg)
    b.run()


if __name__ == "__main__":
    main()
