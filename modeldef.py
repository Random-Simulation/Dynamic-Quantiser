"""modeldef.py -- model-agnostic model definition read from a GGUF header.

This is the single source of structural truth for the whole tool. Given a
llama.cpp-compatible GGUF (bf16/f16/f32 source) it reports, with NO
knowledge of any particular model:

  * which tensors are quantizable   -- a faithful port of llama.cpp's
    `tensor_allows_quantization` (src/llama-quant.cpp)
  * the display/search GROUP each quantizable tensor belongs to -- one
    ordered pattern table over GGML tensor names
  * the tensor inventory (name, ndims, nel, source type, source bytes)
  * n_layer (the max `blk.N` index present)

The ONLY model-name knowledge in the entire tool lives here, in two places:
  (a) the allows_quantization rules, and
  (b) the group pattern table.
Everything measured (S/Q/C per tier, ladder, overhead) comes from the built
cosine table (tablebuild.py); everything structural (groups, frozen,
n_layer, nel) comes from here, read live from the GGUF.
"""
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR / "gguf-py") not in sys.path:
    sys.path.insert(0, str(APP_DIR / "gguf-py"))
from gguf.constants import (GGUF_MAGIC, GGMLQuantizationType,   # noqa: E402
                            GGML_QUANT_SIZES)


# ---------------------------------------------------------------------------
# 1. allows_quantization -- faithful port of tensor_allows_quantization.
#    llama-quantize defaults to quantize_output_tensor=true (there is an
#    opt-out --leave-output-tensor), so a separate output.weight IS
#    quantizable here.
# ---------------------------------------------------------------------------
_TIME_MIX_FROZEN = (
    "time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight",
    "time_mix_w2.weight", "time_mix_v0.weight", "time_mix_v1.weight",
    "time_mix_v2.weight", "time_mix_a0.weight", "time_mix_a1.weight",
    "time_mix_a2.weight", "time_mix_g1.weight", "time_mix_g2.weight",
    "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight")
_SUBSTR_FROZEN = ("_norm.weight", "ffn_gate_inp.weight", "altup", "laurel",
                  "per_layer_model_proj", "ssm_conv1d",
                  "shortconv.conv.weight", "attn_rel_b.weight",
                  ".position_embd", "sam.pos_embd", "sam.neck.", "sam.net_",
                  ".rel_pos", ".patch_embd", ".patch_merger")


def allows_quantization(name, ndims):
    """True iff llama-quantize would quantize this tensor (keep source type
    when False)."""
    if ndims < 2:
        return False
    if not name.endswith("weight"):
        return False
    if name in _TIME_MIX_FROZEN:
        return False
    for p in _SUBSTR_FROZEN:
        if p in name:
            return False
    return True


# ---------------------------------------------------------------------------
# 2. group classifier -- one ordered pattern table over GGML names.
#    First match wins; applied only to quantizable tensors. Anything
#    quantizable that matches nothing falls into the "other" group, so no
#    tensor is ever left ungrouped. Groups are reported as the subset
#    actually present in the model.
# ---------------------------------------------------------------------------
GROUP_PATTERNS = [
    # split attention (most archs)
    ("attn_q",       r"\.attn_q\.weight$"),
    ("attn_k",       r"\.attn_k\.weight$"),
    ("attn_v",       r"\.attn_v\.weight$"),
    ("attn_output",  r"\.attn_output\.weight$"),
    # fused / linear-attention attention
    ("attn_qkv",     r"\.attn_qkv[a-z_]*\.weight$"),
    ("attn_gate",    r"\.attn_gate\.weight$"),
    ("ssm_out",      r"\.ssm_out\.weight$"),
    # dense / shared-expert MLP (ffn_gate_up must precede ffn_gate)
    ("ffn_gate_up",  r"\.ffn_gate_up\.weight$"),
    ("ffn_gate",     r"\.ffn_gate\.weight$"),
    ("ffn_up",       r"\.ffn_up\.weight$"),
    ("ffn_down",     r"\.ffn_down\.weight$"),
    # MoE expert blocks (separate or fully-fused)
    ("moe_ffn",      r"_exps\.weight$"),
    # MTP / speculative head
    ("mtp",          r"\.(?:nextn|mtp)\..*\.weight$"),
    # embeddings / head
    ("Embed",        r"^(?:[a-z_]+_)?token_embd\.weight$"),
    ("Output",       r"^output\.weight$"),
]
_GROUP_RE = [(g, re.compile(p)) for g, p in GROUP_PATTERNS]
GROUP_ORDER = [g for g, _ in GROUP_PATTERNS] + ["other"]

# canonical tier ladder (smallest -> largest) and the tiers that REQUIRE an
# imatrix. No imatrix -> those tiers are absent from the ladder. Shared by
# the builder (tablebuild/tablebuild2) and the size/cosine model (qmodel).
LADDER = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S",
          "Q2_K", "IQ3_XXS", "Q3_K", "IQ3_S", "IQ4_XS",
          "Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16"]
IMATRIX_REQUIRED = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S",
                    "IQ3_XXS"}

# K-only fast ladder: no imatrix tiers (the IQ tiers are the slow
# grid-search quantizers and need imatrix anyway). Builds ~10x faster
# than the full ladder. Imatrix is irrelevant to the TABLE (K-quants only
# consume imatrix in the weighted quantize path, which a no-imatrix table
# does not take); it can still be used for the FINAL llama-quantize step.
LADDER_K = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16"]
LADDER_KINDS = {"k": LADDER_K, "full": None}    # None = ladder_for(imatrix)


def ladder_for(use_imatrix, kind="full"):
    """The tier ladder a table uses for this (imatrix, ladder-kind) choice.
    kind: 'k' -> LADDER_K (no imatrix tiers); 'full' -> LADDER trimmed by
    the imatrix choice (the historical behaviour)."""
    if kind == "k":
        return list(LADDER_K)
    return LADDER if use_imatrix else \
        [t for t in LADDER if t not in IMATRIX_REQUIRED]


def group_of(name, quantizable=True):
    """Group label for a tensor: "" when frozen / non-quantizable, else the
    first matching pattern (or "other")."""
    if not quantizable:
        return ""
    for g, rx in _GROUP_RE:
        if rx.search(name):
            return g
    return "other"


# ---------------------------------------------------------------------------
# 3. load_model -- read ONLY the GGUF header (KV metadata + tensor-info
#    table, a few MB at most) and classify. The tensor DATA section is
#    never mapped or touched, so even a multi-GB source parses in
#    milliseconds (the old full-file memmap reader took ~25 s on 9 GB).
# ---------------------------------------------------------------------------
_BLK_RE = re.compile(r"^blk\.(\d+)\.")

# GGUFValueType (gguf.h): UINT8=0 INT8=1 UINT16=2 INT16=3 UINT32=4
# INT32=5 FLOAT32=6 BOOL=7 STRING=8 ARRAY=9 UINT64=10 INT64=11 FLOAT64=12
_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                      6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class _HeaderReader:
    """Small sequential reader over the GGUF header (buffered reads only)."""

    def __init__(self, fh, endian):
        self.fh = fh
        self.e = endian

    def u32(self):
        return struct.unpack(self.e + "I", self.fh.read(4))[0]

    def u64(self):
        return struct.unpack(self.e + "Q", self.fh.read(8))[0]

    def raw(self, n):
        data = self.fh.read(n)
        if len(data) != n:
            raise ValueError(f"GGUF truncated: expected {n} more bytes")
        return data

    def string(self):
        return self.raw(self.u64()).decode("utf-8")

    def skip_value(self, vtype, depth=0):
        if depth > 12:
            raise ValueError("GGUF array nesting too deep")
        if vtype == _GGUF_ARRAY:
            sub, n = self.u32(), self.u64()
            for _ in range(n):
                self.skip_value(sub, depth + 1)
        elif vtype == _GGUF_STRING:
            self.raw(self.u64())
        else:
            self.raw(_GGUF_SCALAR_SIZES[vtype])


def _read_gguf_header(path):
    """(arch, model_name, [(name, dims, GGMLQuantizationType), ...]) from
    the GGUF header only -- no tensor data is read."""
    with open(path, "rb") as fh:
        head = fh.read(4)
        if len(head) < 4:
            raise ValueError("not a GGUF file (empty)")
        if int.from_bytes(head, "little") == GGUF_MAGIC:
            e = "<"
        elif int.from_bytes(head, "big") == GGUF_MAGIC:
            e = ">"
        else:
            raise ValueError("GGUF magic invalid")
        r = _HeaderReader(fh, e)
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = r.u64()
        kv_count = r.u64()
        arch = ""
        model_name = ""
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            if key == "general.architecture" and vtype == _GGUF_STRING:
                arch = r.string()
            elif key == "general.name" and vtype == _GGUF_STRING:
                model_name = r.string()
            else:
                r.skip_value(vtype)
        tensors = []
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = GGMLQuantizationType(r.u32())
            r.u64()                                # data offset (unused)
            tensors.append((name, dims, ttype))
    return arch, model_name, tensors


def load_model(path):
    """Return a dict describing the model:
        arch, model_name, n_layer (max blk.N),
        names, ndims, nel, src_type, src_bytes   (per tensor)
        frozen (bool per tensor), groups (str per tensor, "" if frozen),
        pos {name: i}, GROUPS (ordered present non-empty group labels).
    """
    arch, model_name, tensors = _read_gguf_header(str(path))
    n = len(tensors)
    names = [t[0] for t in tensors]
    ndims = np.array([len(d) for _, d, _ in tensors], np.int32)
    # math.prod: pure-Python big ints (np.prod of a python list can
    # overflow to int32 for tensors with > 2**31 elements)
    nel = np.array([math.prod(d) for _, d, _ in tensors], np.int64)
    src_bytes = np.array(
        [math.prod(d) * GGML_QUANT_SIZES[tt][1] // GGML_QUANT_SIZES[tt][0]
         for _, d, tt in tensors], np.int64)
    src_type = np.array([tt.name for _, _, tt in tensors], dtype=object)
    frozen = np.array([not allows_quantization(nm, int(d))
                       for nm, d in zip(names, ndims)], bool)
    labels = [group_of(nm, bool(not fr))
              for nm, fr in zip(names, frozen)]
    n_layer = -1
    for nm in names:
        m = _BLK_RE.match(nm)
        if m:
            n_layer = max(n_layer, int(m.group(1)))

    # ordered list of present non-empty groups (only quantizable tensors)
    present = set(labels[i] for i in range(n)
                  if (not frozen[i]) and labels[i])
    groups_present = [g for g in GROUP_ORDER if g in present]
    return {
        "arch": arch,
        "model_name": model_name,
        "n_layer": n_layer,
        "names": names,
        "ndims": ndims,
        "nel": nel,
        "src_type": src_type,
        "src_bytes": src_bytes,
        "frozen": frozen,
        "groups": labels,
        "pos": {nm: i for i, nm in enumerate(names)},
        "GROUPS": groups_present,
    }
