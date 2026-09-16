"""qmodel.py -- model-agnostic adapter over modeldef + the cosine table.

Everything structural (groups, frozen, n_layer, per-tensor element counts,
source type) comes from modeldef.load_model(source) -- read live from the
GGUF, so no model is hardcoded. Everything measured (per-tensor S/Q/C per
tier, ladder, overhead, the baked-in frozen s0/q0/c0) comes from the
discovered cosine table (tablebuild.py, format v2).

  cos(x) = (S(x) + s0) / sqrt( A_total * (Q(x) + q0) )

Frozen tensors (llama.cpp's not-quantizable list) keep the SOURCE type and
are baked into s0/q0/c0 at build time, so only the quantizable group/tier
choices drive the estimate. There is no KLD anywhere: the tool reports
cosine and deviation (1 - cos) only.
"""
from pathlib import Path

import re

import modeldef
import tablestore

try:
    import numpy as np
except ImportError:      # pragma: no cover
    np = None


def ladder_for(use_imatrix, kind=None):
    """The tier ladder for an (imatrix, build-kind) choice (no table yet).
    kind defaults to the current build kind (set_build_kind)."""
    return modeldef.ladder_for(use_imatrix, kind or _BUILD_KIND)

APP_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = APP_DIR / "schema.txt"
TABLES_DIR = tablestore.TABLES_DIR

# bytes per element per ggml type (ggml block sizes). BF16 == 2 bytes.
BYTES_PER_ELEM = {
    "IQ1_S":  50 / 256,
    "IQ1_M":  56 / 256,
    "IQ2_XXS": 66 / 256,
    "IQ2_XS":  74 / 256,
    "IQ2_S":   82 / 256,
    "Q2_K":   84 / 256,
    "IQ3_XXS": 98 / 256,
    "IQ3_S":  110 / 256,
    "Q3_K":   110 / 256,
    "IQ4_XS": 136 / 256,
    "Q4_K":   144 / 256,
    "Q5_K":   176 / 256,
    "Q6_K":   210 / 256,
    "Q8_0":   34 / 32,
    "BF16":   2.0,
    "F16":    2.0,
    "F32":    4.0,
}

_FALLBACK_LADDER = modeldef.LADDER

# current model + table source
_SRC = {"source": None, "imatrix": None}
_MODEL = None               # modeldef.load_model(...) structural context
_model_key = None           # (path, mtime) of the cached _MODEL
_UNSET = object()           # "table not looked up yet" sentinel
_TBL = {"plain": _UNSET, "imatrix": _UNSET}  # merged measured contexts

# module-level convenience views (updated by set_table_source); the GUI and
# solvers may read these, but the per-kind context in _TBL is authoritative.
LADDER = list(_FALLBACK_LADDER)
GROUPS = []
n_layer = 0
NAME_NEL = {}

# table build-kind dimension (separate from the imatrix choice):
#   "k"    -> fast K ladder (Q2_K..Q8_0, F16), no imatrix tiers
#   "full" -> the full imatrix-aware ladder (15 tiers with imatrix)
_BUILD_KIND = "k"


def set_build_kind(kind):
    """Select which table (K-fast vs full) the GUI/solvers use."""
    global _BUILD_KIND
    kind = "k" if kind == "k" else "full"
    if kind != _BUILD_KIND:
        _BUILD_KIND = kind
        _TBL["plain"] = _TBL["imatrix"] = _UNSET


def _load_model_cached(source):
    """modeldef.load_model with a (path, mtime) cache so the GUI does not
    re-read the GGUF on every recompute."""
    global _MODEL, _model_key
    if not source:
        _MODEL = None
        _model_key = None
        return None
    try:
        st = Path(source).stat()
    except OSError:
        _MODEL = None
        _model_key = None
        return None
    key = (str(Path(source).resolve()), int(st.st_mtime))
    if key != _model_key:
        _model_key = key
        _MODEL = modeldef.load_model(source)
    return _MODEL


def set_table_source(source, imatrix):
    """Re-target discovery at (source gguf, imatrix gguf). Loads the model
    structure and clears the measured-table cache; refreshes LADDER/GROUPS."""
    global LADDER, GROUPS, n_layer, NAME_NEL
    _SRC["source"] = str(Path(source).resolve()) if source else None
    _SRC["imatrix"] = str(Path(imatrix).resolve()) if imatrix else None
    _TBL["plain"] = _TBL["imatrix"] = _UNSET
    m = _load_model_cached(_SRC["source"])
    if m is not None:
        GROUPS = list(m["GROUPS"])
        n_layer = int(m["n_layer"])
        NAME_NEL = {n: int(c) for n, c in zip(m["names"], m["nel"])}
    else:
        GROUPS, n_layer, NAME_NEL = [], 0, {}
    t = _load_cos("plain")
    LADDER = t["ladder"] if t else list(modeldef.ladder_for(False,
                                                            _BUILD_KIND))
    return m


def find_cos_table(kind):
    """Valid v2-table path for (current source, build-kind, imatrix choice)
    or None. K tables never carry an imatrix, so the imatrix dimension is
    ignored for them (one K table serves both checkbox states)."""
    src, imx = _SRC["source"], _SRC["imatrix"]
    if src:
        if _BUILD_KIND == "k":
            p = tablestore.find_table(src, False, None, "k")
        else:
            # Full builds are always built with imatrix (build_table
            # enforces it), so always look up the imatrix table -- the
            # GUI checkbox only gates the final llama-quantize --imatrix
            # flag, not table discovery.
            p = tablestore.find_table(src, True, imx, "full")
        if p is not None:
            return p
    return None


def _derive_structural(names, upg, z):
    """Structural fields from the table alone (fallback when the GGUF is not
    available). Groups are recomputed from names via the generic classifier."""
    n_layer = -1
    for nm in names:
        m = modeldef._BLK_RE.match(nm)
        if m:
            n_layer = max(n_layer, int(m.group(1)))
    groups = [modeldef.group_of(nm, bool(u)) for nm, u in zip(names, upg)]
    present = set(g for g, u in zip(groups, upg) if u and g)
    GROUPS = [g for g in modeldef.GROUP_ORDER if g in present]
    nel = z["nel"] if "nel" in z.files else None
    src_type = z["src_type"] if "src_type" in z.files else None
    arch = str(z["arch"]) if "arch" in z.files else ""
    model_name = str(z["model_name"]) if "model_name" in z.files else ""
    return {"n_layer": n_layer, "groups": groups, "GROUPS": GROUPS,
            "nel": nel, "src_type": src_type, "arch": arch,
            "model_name": model_name, "frozen": ~upg}


def _load_cos(kind):
    """Merged context for a table kind ('plain' | 'imatrix'), or None.
    Measured fields from the table; structural fields prefer the live GGUF
    (modeldef) so a stale table's group labels never leak through."""
    if _TBL[kind] is not _UNSET:
        return _TBL[kind]
    p = find_cos_table(kind)
    if p is None or np is None:
        _TBL[kind] = None
        return None
    z = np.load(str(p), allow_pickle=True)
    names = z["names"].tolist()
    ladder = z["ladder"].tolist()
    S, Q, C, A = z["S"], z["Q"], z["C"], z["A"]
    upg = (C[:, 0] != 0)
    t = {
        "path": str(p), "ladder": ladder, "names": names,
        "pos": {n: i for i, n in enumerate(names)},
        "upg": upg, "S": S, "Q": Q, "C": C,
        # Weight-space cosine is measured over the tensors that are actually
        # quantized (upg). Frozen tensors are stored exactly and are not part
        # of "how well the weights held up", so A_total sums upg only and there
        # is no frozen s0/q0 offset. This is robust to a sentinel value in any
        # never-quantized tensor (e.g. a 1e30 rope_freqs entry). Frozen tensors
        # still count for file size via c0 (unchanged). Stored s0/q0 are
        # ignored on purpose so old tables keep working.
        "A_total": float(A[upg].sum()),
        "s0": 0.0, "q0": 0.0,
        "c0": int(z["c0"]), "overhead": int(z["overhead"]),
    }
    m = _load_model_cached(_SRC["source"])
    if m is not None and set(m["names"]) == set(names):
        t["groups"] = list(m["groups"])
        t["frozen"] = np.asarray(m["frozen"])
        t["GROUPS"] = list(m["GROUPS"])
        t["nel"] = np.asarray(m["nel"])
        t["src_type"] = np.asarray(m["src_type"])
        t["n_layer"] = int(m["n_layer"])
        t["arch"] = m["arch"]
        t["model_name"] = m["model_name"]
    else:
        t.update(_derive_structural(names, upg, z))
    z.close()
    _TBL[kind] = t
    return t


# ---------------------------------------------------------------------------
# assignment -> rows -> schema / size
# ---------------------------------------------------------------------------

def bumped_tier(tier, steps):
    if steps <= 0:
        return tier
    i = LADDER.index(tier)
    return LADDER[min(i + steps, len(LADDER) - 1)]


def parse_layers(text, lo=0, hi=None):
    """Parse '0, 63' / '5-9, 12' -> sorted set of layer ints. hi defaults to
    the current model's n_layer. Raises ValueError on out-of-range input."""
    if hi is None:
        hi = n_layer
    out = set()
    text = (text or "").strip()
    if not text:
        return out
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = (x.strip() for x in tok.split("-", 1))
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    bad = sorted(x for x in out if x < lo or x > hi)
    if bad:
        raise ValueError(f"layers must be in {lo}..{hi}, got {bad}")
    return out


def build_assignment(tiers, override_layers, steps):
    """Full tensor assignment: ordered list of (name, tier, n_elements).

    tiers: {group: tier} for the present quantizable groups.
    override_layers: set of layer indices whose quantizable tensors are
    bumped `steps` tiers up. Frozen tensors keep their source type.
    """
    m = _load_model_cached(_SRC["source"])
    if m is None:
        raise ValueError("no source model loaded")
    t = _load_cos("plain") or _load_cos("imatrix")
    ladder = t["ladder"] if t else LADDER
    for g, tier in tiers.items():
        if tier not in ladder:
            raise ValueError(f"tier {tier!r} not in ladder for group {g}")
    rows = []
    names, groups, frozen, nel, src_type = (m["names"], m["groups"],
                                            m["frozen"], m["nel"],
                                            m["src_type"])
    for i, name in enumerate(names):
        if frozen[i]:
            rows.append((name, str(src_type[i]), int(nel[i])))
            continue
        g = groups[i]
        tier = tiers.get(g, "Q4_K")
        m2 = modeldef._BLK_RE.match(name)
        if m2 and int(m2.group(1)) in override_layers:
            tier = bumped_tier(tier, steps)
        rows.append((name, tier, int(nel[i])))
    return rows


def schema_text(rows):
    # Each tensor-type-file line's name is compiled by llama-quantize as a
    # REGEX and matched with std::regex_search (substring, first-match-wins
    # in file order). Tensor names contain '.' (a regex wildcard) and have
    # prefix relationships, so a raw name can match a DIFFERENT tensor: e.g.
    # a frozen 1-D tensor 'blk.N.ssm_a' (F32) is a regex prefix of the
    # quantizable 'blk.N.ssm_alpha.weight' and would hijack its type, storing
    # it as F32 instead of its intended tier (size drift). Anchoring the
    # escaped name to the full string (^...$) makes each line match ONLY its
    # own tensor exactly. (llama-quantize lowercases the pattern; tensor names
    # are already lowercase, so exact-match is preserved.)
    return "\n".join(f"^{re.escape(n)}$={t}" for n, t, _ in rows) + "\n"


def expected_bytes(rows):
    return sum(int(nel) * BYTES_PER_ELEM[t] for _, t, nel in rows)


def gb(nbytes):
    return nbytes / 1e9


def cos_estimate(rows, use_imatrix=False):
    """Whole-model weight-space cosine for an assignment. Returns
    (cos, deviation, note); cos is None if the table is missing or a chosen
    tier is not on its ladder. Every quantizable tensor has full S/Q/C rows
    (frozen is baked), so there is no unmodelled-tier caveat."""
    t = _load_cos("imatrix" if use_imatrix else "plain")
    if t is None:
        return None, None, ("cosine table not found for the current "
                            "source -- use 'Build table'")
    tier_col = {n: i for i, n in enumerate(t["ladder"])}
    bad = set()
    Su = Qu = 0.0
    for n, tier, _ in rows:
        i = t["pos"].get(n)
        if i is None or not t["upg"][i]:
            continue                      # frozen: baked into s0/q0
        if tier not in tier_col:
            bad.add(tier)
            continue
        Su += t["S"][i, tier_col[tier]]
        Qu += t["Q"][i, tier_col[tier]]
    if bad:
        return None, None, "tier(s) not in cosine table: " + ", ".join(sorted(bad))
    cos = (Su + t["s0"]) / np.sqrt(t["A_total"] * (Qu + t["q0"]))
    return float(cos), 1.0 - float(cos), ""


def size_estimate(rows, use_imatrix=False):
    """Whole-model FILE size (GB) for an assignment -- matches the actual
    llama-quantize output, i.e. what the built table predicts.

        size = sum(C[i, tier]) + c0 + overhead
          * C[i,tier] is the MEASURED per-tensor byte count (already carries
            any shape-fallback and the per-tensor GGUF alignment -- the C
            column is always 32-byte aligned, so no separate padding is
            needed),
          * c0 bakes the frozen tensors' bytes in exactly,
          * overhead is the type-independent GGUF container size (header +
            KV metadata + tensor-name/info tables + alignment).

    Unlike expected_bytes() (the BYTES_PER_ELEM approx, which omits the
    container overhead and the frozen bytes), this is what the quantized
    file actually comes out at -- so the GUI 'Expected size' tracks reality
    instead of under-reporting by ~overhead + c0."""
    t = _load_cos("imatrix" if use_imatrix else "plain")
    if t is None:
        return None
    tier_col = {n: i for i, n in enumerate(t["ladder"])}
    pos, upg, C = t["pos"], t["upg"], t["C"]
    total = 0
    for n, tier, nel in rows:
        i = pos.get(n)
        if i is None or not upg[i]:
            continue                      # frozen: baked into c0
        k = tier_col.get(tier)
        if k is None:
            continue                      # tier not on this table's ladder
        total += int(C[i, k])
    return (total + t["c0"] + t["overhead"]) / 1e9


def size_bounds(kind="plain"):
    """(min_gb, max_gb) feasible from the table: all-quantizable at the
    per-tensor smallest tier vs at F16 (frozen c0 + overhead always
    included). The floor is the per-tensor min (NOT raw column 0): a tier's
    column can be larger than another tier's for a given tensor (shape
    fallback, or an uncovered-tier Q4_K column), so the true achievable
    floor is each quantizable tensor at its own smallest column -- the same
    floor the per-tensor solver uses."""
    t = _load_cos(kind)
    if t is None:
        return None, None
    K = len(t["ladder"])
    upg = t["upg"]
    cmin = float(t["C"][upg].min(axis=1).sum()) + t["c0"] + t["overhead"]
    cmax = float(t["C"][upg, K - 1].sum()) + t["c0"] + t["overhead"]
    return cmin / 1e9, cmax / 1e9


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python qmodel.py <source.gguf>")
    set_table_source(sys.argv[1], None)
    t = _load_cos("plain")
    print("groups:", GROUPS, " n_layer:", n_layer)
    if t:
        lo, hi = size_bounds()
        print(f"feasible size: {lo:.2f} .. {hi:.2f} GB")
    rows = build_assignment({g: "Q4_K" for g in GROUPS}, set(), 0)
    print(f"{len(rows)} tensors, {gb(expected_bytes(rows)):.2f} GB all-Q4_K")
