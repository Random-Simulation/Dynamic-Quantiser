# Quant Maker

Model-agnostic custom quantizer for llama.cpp GGUF models (bf16/f16/f32
sources). Pick a quant type per tensor group, optionally bump whole layers
a tier, watch the expected file size and whole-model weight-space cosine
update live, then either solve for group tiers or a full per-tensor
assignment under a target size / target deviation, and hand the resulting
`schema.txt` to `llama-quantize`. No model is hardcoded.

## Run

    python quant_maker.py

## Pipeline

1. **modeldef.py** -- reads only the GGUF header (tensor inventory, group
   classification, `n_layer`); ports llama.cpp's
   `tensor_allows_quantization`.
2. **tablebuild2.py** (fast path, fused C kernel in `fastq.c`/`fastq.dll`)
   or **tablebuild.py** (numpy reference engine) -- build a per-tensor,
   per-tier cosine table (S/Q/C columns) into `tables/`. Resumable.
3. **qsolve.py** (group-level) / **qfullsolve.py** (full per-tensor) --
   solve `max cosine s.t. target size` or `min size s.t. target cosine`.
4. **quant_maker.py** -- tkinter GUI tying it together: table build, live
   size/cosine estimates, `schema.txt`, `llama-quantize`, imatrix
   generation.

## CLI entry points

    # fast table (K ladder) / full table
    python tablebuild2.py --source m.gguf [--ladder-kind k|full]
    # numpy reference table (+ --check-tier byte-exactness gate)
    python tablebuild.py --source m.gguf [--imatrix i.gguf] [--check-tier T]
    # compile the C kernel (one-off, MSVC)
    python build_fastq.py

## Requirements

- Python 3.10+ with `numpy` (see `requirements.txt`).
- `ggml-base.dll` (llama.cpp) and `llama-quantize` / `llama-imatrix`,
  searched in `<project>/binaries`, `../binaries`, `../app`, then `PATH`.
- `gguf-py/` -- vendored copy of the `gguf` library (imported directly via
  a `sys.path` insert; no pip install needed).

## Layout

- `quant_maker.py` -- GUI entry point
- `modeldef.py` / `qmodel.py` -- structure + size/cosine model
- `qsolve.py` / `qfullsolve.py` -- solvers
- `tablebuild.py` / `tablebuild2.py` / `tablestore.py` -- table engines +
  store (format v2, `tables/`, indexed by `tables/index.json`)
- `fastq.py` / `fastq.c` / `fastq.dll` -- fused build kernel + bindings
- `build_fastq.py` -- compiles the kernel
- `QuantMaker.spec` -- PyInstaller spec (`build/` + `dist/` are outputs)
