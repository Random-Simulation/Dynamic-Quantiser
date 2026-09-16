# -*- mode: python ; coding: utf-8 -*-
# Bundle only fastq.dll (the fused table-build kernel, ~108 KB). The user
# points Quant Maker at their own llama.cpp build via the "Binaries dir"
# field in the Model panel; the runtime finds llama-quantize.exe,
# llama-imatrix.exe and ggml-base.dll there (quant_maker._find_bin +
# tablebuild.tool_search_dirs). fastq.dll is loaded next to the .exe and
# initialised with the user's ggml-base.dll at runtime.

a = Analysis(
    ['quant_maker.py'],
    pathex=['gguf-py'],
    binaries=[
        ('fastq.dll', '.'),
    ],
    datas=[],
    hiddenimports=['gguf'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchtext',
        'nvidia', 'nvidia.cublas', 'nvidia.cudnn', 'nvidia.cufft',
        'nvidia.cuda_runtime', 'nvidia.cuda_nvrtc', 'nvidia.nccl',
        'nvidia.curand', 'nvidia.cusolver', 'nvidia.nvtx',
        'nvidia.nvjitlink', 'cuda',
        'datasets', 'diffusers', 'transformers', 'accelerate',
        'librosa', 'lightning', 'pytorch_lightning',
        'matplotlib', 'contourpy', 'kiwisolver', 'PIL',
        'IPython', 'jedi', 'av', 'aiohttp',
        'babel', 'lxml', 'msgpack', 'dill',
        'huggingface_hub', 'hf_xet', 'filelock',
        'fsspec', 'attrs', 'certifi', 'charset_normalizer',
        'click', 'dateutil', 'jsonschema', 'jinja2',
        'google', 'hydra', 'blis', 'cymem',
        'scipy', 'pandas', 'sympy',
        'numpy.fft', 'numpy.testing',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Dynamic-Quantiser',
    icon='dynamic-quantiser.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='Dynamic-Quantiser',
)
