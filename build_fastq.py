"""build_fastq.py -- compile fastq.c into a platform shared library.

Usage: python build_fastq.py
  Windows: finds VS via vswhere, runs cl through vcvars64 -> fastq.dll
  Linux:   gcc -shared -fPIC -> libfastq.so
  macOS:   clang -shared -fPIC -> libfastq.dylib
No CMake, no Python API.
"""
import platform
import shutil
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def find_vcvars():
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio"
                   r"\Installer\vswhere.exe")
    if vswhere.exists():
        out = subprocess.run([str(vswhere), "-latest", "-products", "*",
                              "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools."
                              "x86.x64", "-property", "installationPath"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            p = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if p.exists():
                return p
    for c in [r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
              r"C:\Program Files\Microsoft Visual Studio\2022\Community",
              r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"]:
        p = Path(c) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if p.exists():
            return p
    sys.exit("ERROR: vcvars64.bat not found (install VS C++ build tools)")


def build_windows():
    vcvars = find_vcvars()
    src = APP_DIR / "fastq.c"
    out = APP_DIR / "fastq.dll"
    if out.exists():
        out.unlink()
    cl = (f'cl /nologo /O2 /W3 /arch:AVX2 /LD /EHsc "{src}" /Fe:"{out}" '
          f'/link /DLL')
    cmd = f'call "{vcvars}" >nul && {cl}'
    r = subprocess.run(cmd, shell=True, cwd=str(APP_DIR),
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    # cl leaves build artifacts behind
    for p in ("fastq.obj", "fastq.lib", "fastq.exp"):
        (APP_DIR / p).unlink(missing_ok=True)
    return out


def build_unix(cc="gcc"):
    """Build fastq.c as a shared library on Linux (gcc) or macOS (clang).
    Returns the output path."""
    src = APP_DIR / "fastq.c"
    if platform.system() == "Darwin":
        out = APP_DIR / "libfastq.dylib"
        cc = cc or "clang"
    else:
        out = APP_DIR / "libfastq.so"
    if out.exists():
        out.unlink()
    cmd = f'{cc} -O2 -shared -fPIC -o "{out}" "{src}"'
    r = subprocess.run(cmd, shell=True, cwd=str(APP_DIR),
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if not out.exists():
        sys.exit("ERROR: build failed")
    return out


def main():
    sysname = platform.system()
    if sysname == "Windows":
        out = build_windows()
    else:
        out = build_unix()
    print(f"OK -> {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
