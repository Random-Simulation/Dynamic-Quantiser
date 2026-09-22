# Dynamic Quantiser

Allows anyone to make their own dynamic quantisations of LLM .gguf files.
Automatic, fast and fairly optimal - considerably more accurate than
standard K quants. Should work with any llama.cpp compatible .gguf file
(bf16/f16/f32). Chooses tensors at mixed quantisation levels to minimise
whole-model cosine deviation. Also lets you make your own custom 
quantisations by picking layers/groups at different quantisation levels.

## Motivation

Many local LLM users are VRAM constrained, and this tool lets anyone make
exact-size quantised models that are considerably better in accuracy than
standard Qx_K_M quants. Whole-model cosine deviation is highly correlated
with KLD (a 0.99 Pearson correlation) - a good measure of quant quality.
Once the initial table has been built, solutions are just seconds.

## QuickStart – Windows

1. Unzip and double click `Dynamic-Quantiser.exe` (or run `python quant_maker.py`)
2. Enter the path for a bf16 model in **Source**
3. Enter the path to the llama.cpp binaries in **Binaries**
   (the `llama-quantize.exe`, `llama-imatrix.exe` and `ggml-base.dll`
   from your llama.cpp build)
4. Optionally enter an imatrix path (or make one)
5. Click **Build Table** (this takes a while, ~30s per GB)
6. Click **Make Dynamic Quant**

The table is a one-off build. After that, solutions are just seconds.

## QuickStart – Linux

1. **Install Python 3.10+ and numpy** (if you don't have them already):
   ```bash
   sudo apt install python3 python3-pip
   pip3 install numpy
   ```

2. **Install a C compiler** (one-time, to build the fast kernel):
   ```bash
   sudo apt install build-essential
   ```

3. **Build the fast table-build kernel** (one-time):
   ```bash
   python3 build_fastq.py
   ```
   This compiles `fastq.c` into `libfastq.so` using `gcc`. You only
   need to do this once, or if you update the source code.

4. **Get the llama.cpp Linux binaries** – `llama-quantize`,
   `llama-imatrix` and `libggml-base.so`. Drop them in a `binaries`
   folder next to the app, or put them on your `PATH`.

5. **Run the app:**
   ```bash
   python3 quant_maker.py
   ```
   A window opens – the same one as Windows.

6. Enter the path for a bf16 model in **Source**
7. Enter the path to the llama.cpp binaries in **Binaries**
8. Optionally enter an imatrix path (or make one)
9. Click **Build Table** (this takes a while, ~30s per GB)
10. Click **Make Dynamic Quant**

## QuickStart – macOS

> **Note for Apple Silicon Macs (M1/M2/M3/M4):** the fast C kernel uses
> x86-specific instructions and won't compile on Apple Silicon. The app
> still works fine – it just uses the (slower) pure-Python table builder
> instead. **Intel Macs** can build the fast kernel as normal.

1. **Install Python 3.10+** (if you don't have it already):
   ```bash
   brew install python
   ```
   (If you don't have Homebrew, install it first from
   [brew.sh](https://brew.sh). Or download Python from
   [python.org](https://python.org) – the installer includes tkinter.)

2. **Install numpy:**
   ```bash
   pip3 install numpy
   ```

3. **Build the fast table-build kernel** – Intel Macs only, one-time:
   ```bash
   python3 build_fastq.py
   ```
   This compiles `fastq.c` into `libfastq.dylib` using `clang`
   (built-in with macOS, no extra install needed).
   *Skip this step on Apple Silicon Macs – the app will use the
   pure-Python table builder instead.*

4. **Get the llama.cpp macOS binaries** – `llama-quantize`,
   `llama-imatrix` and `libggml-base.dylib`. Drop them in a `binaries`
   folder next to the app, or put them on your `PATH`.

5. **Run the app:**
   ```bash
   python3 quant_maker.py
   ```
   A window opens – the same one as Windows.

6. Enter the path for a bf16 model in **Source**
7. Enter the path to the llama.cpp binaries in **Binaries**
8. Optionally enter an imatrix path (or make one)
9. Click **Build Table** (this takes a while, ~30s per GB)
10. Click **Make Dynamic Quant**

## How it works

There are 3 different algorithms in the program:

1. **Make Dynamic Quant** - the most accurate, full solution. It minimises
   whole-model cosine deviation for a fixed file size, choosing the best
   quantisation level for every individual tensor. Requires a target file
   size.
2. **Minimise cosine deviation** - minimises whole-model cosine deviation,
   but not at tensor level; instead it uses a much more approximate solution
   at layer/group level. Useful for making your own custom quants - easy to
   tweak from a decent base solution. Also requires a target file size.
3. **Minimise disk size** - again at layer/group level (not tensor level),
   this does the same thing as 2, but minimises disk size instead, subject
   to a fixed cosine deviation. Requires a target cosine deviation.

## Requirements

- Python 3.10+ with `numpy` (see `requirements.txt`).
- A C compiler, to build the fast table-build kernel one-off
  (`python build_fastq.py`). *Optional on Apple Silicon Macs – the app
  falls back to the pure-Python table builder.*
- The llama.cpp binaries, in a `binaries` folder next to the app
  (the app also looks in a couple of other spots and your system `PATH`):
  - **Windows**: `llama-quantize.exe`, `llama-imatrix.exe`, `ggml-base.dll`
  - **Linux**: `llama-quantize`, `llama-imatrix`, `libggml-base.so`
  - **macOS**: `llama-quantize`, `llama-imatrix`, `libggml-base.dylib`
  [Prebuilt binaries are available on the llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases).
- A starting `.gguf` model - a bf16 one works best.

`gguf-py/` is a bundled copy of the `gguf` library, so there's nothing
extra to install.

## Files

1. **modeldef.py** - peeks at the model so it knows what tensors are in
   there, how they group together, and which ones are safe to quantise.
2. **tablebuild2.py** - the fast table builder. It works out, for every
   tensor, what it looks like at each quant size and roughly how much
   quality you'd lose, then saves that to `tables/`. If it gets interrupted
   you can pick up where it left off. (`tablebuild.py` does the same thing
   in pure Python, a little slower.)
3. **qsolve.py** / **qfullsolve.py** - given a target file size, work out
   the best mix of tensor sizes so the model stays as accurate as possible.
4. **quant_maker.py** - The GUI. It builds the table,
   shows live size/accuracy estimates as you tweak things, and runs the
   final quantisation.

## Command line

    # fast table (K ladder) / full table
    python tablebuild2.py --source m.gguf [--ladder-kind k|full]
    # numpy reference table (+ --check-tier byte-exactness gate)
    python tablebuild.py --source m.gguf [--imatrix i.gguf] [--check-tier T]
    # compile the C kernel (one-off)
    python build_fastq.py

## Project layout

- `quant_maker.py` - the app you run (the window)
- `modeldef.py` / `qmodel.py` - understand the model, estimate size/accuracy
- `qsolve.py` / `qfullsolve.py` - the solvers that pick the best mix of sizes
- `tablebuild.py` / `tablebuild2.py` / `tablestore.py` - build and save the
  size/accuracy table
- `fastq.py` / `fastq.c` - the fast C kernel that speeds up table building
  (compiled to `fastq.dll` on Windows, `libfastq.so` on Linux,
  `libfastq.dylib` on macOS)
- `build_fastq.py` - compiles the C kernel (only needed once, cross-platform)
- `Dynamic-Quantiser.spec` - recipe for building the Windows `Dynamic-Quantiser.exe`

## Acknowledgements

Made with local Qwen 3.8 27b inside Bare agentic harness.
