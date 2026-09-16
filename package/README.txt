DYNAMIC QUANTISER
==================

Make your own dynamic quantisations of LLM .gguf files. Fast, automatic and
considerably more accurate than standard K quants. Works with any
llama.cpp-compatible .gguf file (bf16/f16/f32). It picks tensors at mixed
quantisation levels to minimise whole-model cosine deviation (which is
highly correlated with KLD, a good proxy for quant quality).

WHAT'S IN THIS FOLDER
---------------------
  Dynamic-Quantiser.exe   the program - double-click to run
  _internal\              bundled runtime - KEEP this next to the exe
  README.txt              this file

Do NOT move or delete the _internal folder; the exe needs it to start.

SETUP (one-off)
---------------
You need the llama.cpp binaries in one folder:
  - llama-quantize(.exe)
  - llama-imatrix(.exe)
  - ggml-base.dll

(They ship with any standard llama.cpp build.) Put them in a folder, then in
the app's **Binaries** field enter the path to that folder.

YOU NEED
--------
  - A starting .gguf model - a bf16 one works best.
  - (Optional) an imatrix .gguf for better-calibrated quants.

QUICK START
-----------
  1. Double-click Dynamic-Quantiser.exe
  2. In **Source**, enter the path to your bf16 .gguf model
  3. In **Binaries**, enter the path to the folder with the llama.cpp binaries
  4. (Optional) enter an imatrix path, or click to make one
  5. Click **Build Table**  (takes a while, ~30 seconds per GB, one-off)
  6. Click **Make Dynamic Quant**

The table is built once. After that, each new quant is just a few seconds.

THREE MODES
-----------
  - Make Dynamic Quant: the most accurate. For a target file size, chooses
    the best quant level per tensor. Requires a target size.
  - Minimise cosine deviation: approximate layer/group-level solution for a
    target size. Handy for making custom quants you can tweak.
  - Minimise disk size: same as above but minimises size for a fixed
    cosine deviation. Requires a target deviation.

TROUBLESHOOTING
---------------
  - "Binaries not found": check the **Binaries** folder actually contains
    llama-quantize.exe, llama-imatrix.exe and ggml-base.dll.
  - Antivirus warning: PyInstaller-built exes can trigger false positives.
    If you trust the source, allow it. (A code-signed build avoids this.)
  - Table build interrupted: just run Build Table again - it resumes from
    where it left off (saved in tables\).
