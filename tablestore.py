"""tablestore.py -- table format v2: fingerprinting, naming, discovery, saving.

File naming:  tables/cos_<srcsha8>[_im].npz   (+ tiny tables/index.json
mapping source path -> file so the GUI can find a table by source path).

Fingerprint (v2, no imatrix-REQUIRED tier ambiguity: the ladder itself is
baked into the file and derived from the imatrix choice at build time):
  fp = {src_size, src_mtime, src_sha1_1mb, quant_dll_mtime,
        imatrix: null | {size, mtime, sha1_1mb}}

v2 npz keys:  v, arch, model_name, n_layer, names, ndims, nel, groups,
  frozen, src_type, ladder, S, Q, C, A, s0, q0, c0, overhead, fp
  * frozen rows: S/Q/C all-zero; their source values are baked into
    s0/q0 and their source bytes (ceil32) into c0.
  * c0/overhead as ints; overhead is the (type-independent) GGUF container
    size = source file size - sum(ceil32 source tensor bytes), read once
    at build time (no Q4_K re-quantize -- see tablebuild.Build).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
TABLES_DIR = APP_DIR / "tables"
INDEX_PATH = TABLES_DIR / "index.json"
TABLE_VERSION = 2


def sha1_1mb(path, chunk=1 << 20):
    """SHA-1 of the first 1 MB of a (potentially huge) file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
    return h.hexdigest()


def fp_of(source, imatrix, dll):
    """Fingerprint dict for a (source, imatrix, dll) build configuration."""
    s = Path(source).stat()
    fp = {"src_size": s.st_size, "src_mtime": int(s.st_mtime),
          "src_sha1_1mb": sha1_1mb(source),
          "quant_dll_mtime": int(Path(dll).stat().st_mtime)}
    if imatrix:
        i = Path(imatrix).stat()
        fp["imatrix"] = {"size": i.st_size, "mtime": int(i.st_mtime),
                         "sha1_1mb": sha1_1mb(imatrix)}
    else:
        fp["imatrix"] = None
    return fp


def _suffix(use_imatrix, kind):
    if kind == "k":
        return "_k"            # K-only fast ladder (imatrix not involved)
    return "_im" if use_imatrix else ""


def table_filename(source, use_imatrix=False, kind="full"):
    # cos_<srcsha8>[_im|_k].npz -- one file per (source, imatrix, ladder)
    return f"cos_{sha1_1mb(source)[:8]}{_suffix(use_imatrix, kind)}.npz"


def _fp_matches(fp, source, imatrix):
    """True if a stored fingerprint still describes (source, imatrix)."""
    try:
        s = Path(source).stat()
    except OSError:
        return False
    if (int(fp.get("src_size", -1)) != s.st_size
            or int(fp.get("src_mtime", -2)) != int(s.st_mtime)
            or fp.get("src_sha1_1mb") != sha1_1mb(source)):
        return False
    im = fp.get("imatrix")
    if (im is None) != (imatrix is None):
        return False
    if im is not None:
        try:
            i = Path(imatrix).stat()
        except OSError:
            return False
        if (int(im.get("size", -1)) != i.st_size
                or int(im.get("mtime", -2)) != int(i.st_mtime)
                or im.get("sha1_1mb") != sha1_1mb(imatrix)):
            return False
    return True


def _load_index():
    try:
        return json.loads(INDEX_PATH.read_text())
    except Exception:      # noqa: BLE001
        return {}


def _save_index(idx):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=1))
    os.replace(tmp, INDEX_PATH)


def _index_key(use_imatrix, kind):
    if kind == "k":
        return "k"
    return "im" if use_imatrix else "plain"


def find_table(source, use_imatrix=False, imatrix=None, kind="full"):
    """Return the path of a valid v2 table for
    (source, imatrix choice, ladder kind), or None. A table is valid iff
    its stored fingerprint still matches. imatrix: the imatrix path
    (required when use_imatrix, for fp check). kind: 'k' or 'full'."""
    source = str(Path(source).resolve())
    if not Path(source).exists():
        return None
    if use_imatrix and not (imatrix and Path(imatrix).exists()):
        return None
    im_path = Path(imatrix) if use_imatrix else None
    if kind == "k" and im_path is not None:
        im_path = None            # K tables never carry imatrix
    candidates = []
    idx = _load_index()
    entry = idx.get(source) or idx.get(source.lower())
    if entry:
        if entry.get(_index_key(use_imatrix, kind)):
            candidates.append(TABLES_DIR / entry[_index_key(use_imatrix, kind)])
    # also try the canonical name directly (index may be stale/missing)
    p = TABLES_DIR / table_filename(source, use_imatrix, kind)
    if p not in candidates:
        candidates.append(p)
    for p in candidates:
        if not p.exists():
            continue
        try:
            z = np.load(str(p), allow_pickle=True)
            if int(z["v"]) != TABLE_VERSION:
                z.close()
                continue
            if not bool(np.isfinite(z["A"]).all()):
                z.close()
                continue      # in-progress checkpoint: not a complete table
            ok = _fp_matches(json.loads(str(z["fp"])), source, im_path)
            z.close()
        except Exception:      # noqa: BLE001
            continue
        if ok:
            return p
    return None


def register_table(source, use_imatrix, table_path, fp, imatrix=None,
                   kind="full"):
    """Record source -> table file in tables/index.json (atomic)."""
    source = str(Path(source).resolve())
    idx = _load_index()
    entry = idx.setdefault(source, {})
    entry[_index_key(use_imatrix, kind)] = Path(table_path).name
    entry["fp"] = fp
    if imatrix:
        entry["imatrix"] = str(Path(imatrix).resolve())
    _save_index(idx)
