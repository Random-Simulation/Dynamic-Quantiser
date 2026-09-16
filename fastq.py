"""fastq.py -- ctypes bindings for fastq.dll (fused table-build kernel).

FastQ loads fastq.dll and (through it) the project's ggml-base.dll.
  segment(a, npr, types, imx)   -> (A, S, Q) fused f64 stats for one
                                   256M-elem-or-smaller segment, all tiers
Enums and sizes mirror tablebuild.py (same ggml.h values).
"""
import ctypes
import sys
from ctypes import (POINTER, c_float, c_int, c_int64, c_uint8, c_double)
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent


def _find_fastq():
    """Locate fastq.dll: next to the module, next to the frozen .exe (and
    its _internal / binaries subdirs), or in a project binaries dir."""
    cands = [APP_DIR / "fastq.dll"]
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        cands += [exe.parent / "fastq.dll",
                  exe.parent / "_internal" / "fastq.dll",
                  exe.parent / "binaries" / "fastq.dll"]
    cands += [APP_DIR.parent / "fastq.dll",
              APP_DIR / "binaries" / "fastq.dll"]
    for c in cands:
        if c.exists():
            return c
    raise RuntimeError(
        "fastq.dll not found. Searched: "
        + ", ".join(str(c) for c in cands)
        + ". Rebuild the .exe with fastq.dll bundled, or copy fastq.dll "
          "next to the app.")

# ggml enums (ggml.h, stable) -- superset: ladder + fallback types
ENUM = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7,
        "Q8_0": 8, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13,
        "Q6_K": 14, "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18,
        "IQ1_S": 19, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22,
        "IQ4_XS": 23, "IQ1_M": 29}


class FastQ:
    def __init__(self, ggml_dll):
        lib = ctypes.CDLL(str(_find_fastq()))
        self._lib = lib
        lib.fq_init.argtypes = [ctypes.c_char_p]
        lib.fq_init.restype = c_int
        lib.fq_blck_size.argtypes = [c_int]
        lib.fq_blck_size.restype = c_int
        lib.fq_type_size.argtypes = [c_int]
        lib.fq_type_size.restype = c_int
        lib.fq_segment.argtypes = [POINTER(c_float), c_int64, c_int64,
                                   POINTER(c_int), c_int, POINTER(c_float),
                                   POINTER(c_double), POINTER(c_double),
                                   POINTER(c_double), POINTER(c_uint8),
                                   POINTER(c_float), c_int]
        lib.fq_segment.restype = c_int
        if lib.fq_init(str(ggml_dll).encode()) != 0:
            raise RuntimeError(f"fastq: failed to load {ggml_dll}")
        self.blck = {t: lib.fq_blck_size(e) for t, e in ENUM.items()}
        self.rowb = {t: lib.fq_type_size(e) for t, e in ENUM.items()}

    def segment(self, a, npr, tnames, imx):
        """Fused stats for one segment. tnames: list of ggml type names
        (already effective types, no F16). Returns
          (A, S, Q, (S16, Q16, f16_ok))  -- F16 column computed in-kernel;
          f16_ok False when bf16->f16 overflowed (engine zeros the column)."""
        n = a.size
        types = np.array([ENUM[t] for t in tnames], np.int32)
        # raw capacity: max bytes/elem among the requested types
        # (Q5_1 = 1.75 B/elem is the worst; q8_0 = 1.0625). tnames may be
        # empty (every tier fell through to the F16 terminus).
        cap = (max((n // self.blck[t]) * self.rowb[t] for t in tnames)
               if tnames else 0)
        raw = np.empty(cap, np.uint8)
        bb = np.empty(n, np.float32)
        A = np.zeros(1, np.float64)
        imxp = (imx.ctypes.data_as(POINTER(c_float))
                if imx is not None else None)
        S = np.zeros(len(tnames) + 1, np.float64)
        Q = np.zeros_like(S)
        types_p = (types.ctypes.data_as(POINTER(c_int))
                   if len(tnames) else None)
        raw_p = (raw.ctypes.data_as(POINTER(c_uint8)) if cap else None)
        rc = self._lib.fq_segment(a.ctypes.data_as(POINTER(c_float)), n,
                                  c_int64(npr), types_p,
                                  len(tnames), imxp,
                                  A.ctypes.data_as(POINTER(c_double)),
                                  S.ctypes.data_as(POINTER(c_double)),
                                  Q.ctypes.data_as(POINTER(c_double)),
                                  raw_p,
                                  bb.ctypes.data_as(POINTER(c_float)), 1)
        if rc not in (0, -4):
            raise RuntimeError(f"fastq.segment rc={rc} "
                               f"(imx-missing={rc == -3}, n={n}, npr={npr})")
        f16_ok = rc == 0
        return float(A[0]), S[:-1], Q[:-1], (float(S[-1]), float(Q[-1]),
                                             f16_ok)
