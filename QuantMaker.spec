# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['quant_maker.py'],
    pathex=['gguf-py'],
    binaries=[('fastq.dll', '_internal'), ('E:/llama/binaries/ggml-base.dll', '_internal/binaries'), ('E:/llama/binaries/llama-quantize.exe', '_internal/binaries'), ('E:/llama/binaries/llama-imatrix.exe', '_internal/binaries')],
    datas=[],
    hiddenimports=['gguf'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QuantMaker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='QuantMaker',
)
