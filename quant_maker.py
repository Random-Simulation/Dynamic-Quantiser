"""quant_maker.py -- model-agnostic custom quant maker GUI (tkinter).

Works with any llama.cpp-compatible GGUF (bf16/f16/f32 source). Pick a
quant type per tensor group (the groups, layer range and feasible-size
range are all discovered from the selected model + its cosine table),
optionally bump whole layers up a tier or two, watch the expected size and
whole-model weight-space cosine update live, then:

  * "Make custom quant"   -> writes schema.txt and runs llama-quantize
  * "Make dynamic quant"  -> per-tensor solver, then runs llama-quantize
  * "Minimise cosine deviation" / "Minimise disk size" -> group-level solvers
  * "Build table"         -> builds the fast K-ladder cosine table in a
                             worker thread

No model is hardcoded: the source path drives table lookup and the
"Build table" button builds the fast K-ladder table (Q2_K to Q8_0 + F16,
NO IQ_X tiers) in a worker thread (resumable, cancellable). Only the fast
K quants are ever built or used, so the per-tensor / group-level solvers
can only pick fast tiers -- they never see IQ_X. Only cosine + deviation
are shown (there is no KLD). Run:  python quant_maker.py
"""
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import modeldef
import qmodel as q
import tablebuild
import tablebuild2
import tablestore

try:
    import qsolve
    import qfullsolve
except ImportError:          # pragma: no cover (needs numpy)
    qsolve = None
    qfullsolve = None

APP_DIR = q.APP_DIR
SCHEMA = q.SCHEMA_PATH
STATE_PATH = APP_DIR / "state.json"     # paths/settings, no model defaults
MINIMISE_BTN_WIDTH_PX = 170              # shared width for the two minimise buttons


def _find_bin(name, extra_dir=None):
    """Find `name` (+ .exe) in: the user's Binaries dir, a binaries/ folder
    next to the app / .exe, and finally PATH."""
    import shutil
    names = [name + ".exe", name]
    cands = []
    if extra_dir:
        cands.append(Path(extra_dir))
    cands += [APP_DIR / "binaries"]
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        cands += [exe.parent, exe.parent / "binaries"]
    for c in cands:
        for n in names:
            if (c / n).exists():
                return c / n
    p = shutil.which(names[0])
    return Path(p) if p else None


# startup lookup (no user dir yet); the GUI re-resolves dynamically via
# _quantize_exe/_imatrix_exe once the Binaries dir field is available.
QUANTIZE_EXE = _find_bin("llama-quantize")
IMATRIX_EXE = _find_bin("llama-imatrix")

# no model-specific defaults: the user always selects a source
DEFAULTS = {"source": "", "output": str(APP_DIR / "custom_quant.gguf"),
            "imatrix": "", "calib": "",
            "imx_out": str(APP_DIR / "imatrix.dat"), "binaries": ""}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.logq = queue.Queue()
        self.jobq = queue.Queue()      # worker -> main-thread jobs
        self.proc_running = False
        self.tbl_running = False
        self.tbl_cancel = False
        self._tbl_key = None
        self._model_src = None         # source the widgets are built for
        self._saved_tiers = {}         # group -> tier (survives rebuild)
        self._src_after = None         # debounce timer for the source field
        self._loading = False          # suppress rebuilds while restoring state
        self._numpy_warned = False     # only warn about missing numpy once

        root.title("Quant Maker")
        root.geometry("500x1000")
        root.minsize(500, 1000)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Hint.TLabel", foreground="#555555",
                        font=("Segoe UI", 8))
        style.configure("Small.TButton", font=("Segoe UI", 8), padding=(4, 2))

        body = ttk.Frame(root, padding=10)
        body.pack(fill="both", expand=True)
        # Top row: quant types on the left, tuning controls beside them.
        cols = ttk.Frame(body)
        cols.pack(fill="both", expand=True)
        cols.columnconfigure(0, weight=0)   # group sits tight to its content
        cols.columnconfigure(1, weight=1)   # tuning fills the rest
        cols.rowconfigure(0, weight=1)

        g_cell = ttk.Frame(cols)
        t_cell = ttk.Frame(cols)
        g_cell.grid(row=0, column=0, sticky="nsw")
        t_cell.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self._build_group_frame(g_cell)
        self._build_tune_frame(t_cell)

        # Full-width column below: model, expected stats, table build,
        # imatrix and the quant buttons.
        bottom = ttk.Frame(body)
        bottom.pack(fill="x", pady=(8, 0))
        self._build_right_frame(bottom)

        self._build_log(body.pack(fill="x", pady=(8, 0)))

        self._load_state()
        self._on_model_changed()
        self._table_ready = False
        # sizes can only be measured once the window is drawn; defer & retry
        self._equalise_button_widths()
        self._match_table_btn_width()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll_log)
        root.after(100, self._poll_jobs)

    # ----------------------------------------------- equalise button width
    def _equalise_button_widths(self, *_):
        # ttk buttons size to their text, so make the two minimise buttons
        # the same width as the larger one.  The width option is expressed in
        # character cells, so measure the cell size + padding at runtime.
        # The very first call runs before the window is drawn, so defer and
        # retry (via root.after) until the widgets are actually realised.
        self.root.update_idletasks()
        btns = [self.btn_min, self.btn_minsize]
        if any(not b.winfo_viewable() for b in btns):
            return self.root.after(50, self._equalise_button_widths)
        big = max(btns, key=lambda b: b.winfo_reqwidth())
        w1, w2 = 10, 20
        big.config(width=w1); self.root.update_idletasks(); p1 = big.winfo_reqwidth()
        big.config(width=w2); self.root.update_idletasks(); p2 = big.winfo_reqwidth()
        cell = (p2 - p1) / (w2 - w1)
        if cell <= 0:
            return self.root.after(50, self._equalise_button_widths)
        pad = p1 - cell * w1
        chars = int(round((MINIMISE_BTN_WIDTH_PX - pad) / cell))
        for b in btns:
            b.pack_propagate(False)
            b.config(width=chars)

    # -------------------------------------- match Build table to make-quant
    def _match_table_btn_width(self, *_):
        # Make the "Build table" button as wide as the make-quant buttons
        # (which fill their grid columns).  Read the reference's displayed
        # width and convert the target to a character width for ttk.
        self.root.update_idletasks()
        ref = self.btn_quant
        if not ref.winfo_viewable() or ref.winfo_width() < 40:
            return self.root.after(50, self._match_table_btn_width)
        target = ref.winfo_width()
        b = self.btn_table
        if not b.winfo_viewable():
            return self.root.after(50, self._match_table_btn_width)
        w1, w2 = 10, 20
        b.config(width=w1); self.root.update_idletasks(); p1 = b.winfo_reqwidth()
        b.config(width=w2); self.root.update_idletasks(); p2 = b.winfo_reqwidth()
        cell = (p2 - p1) / (w2 - w1)
        if cell <= 0:
            return self.root.after(50, self._match_table_btn_width)
        pad = p1 - cell * w1
        chars = int(round((target - pad) / cell))
        b.pack_propagate(False)
        b.config(width=chars)
        self._table_ready = True

    # ---------------------------------------------------- state persistence
    def _collect_state(self):
        return {
            "groups": {g: v.get() for g, v in self.group_vars.items()},
            "layers": self.layers_var.get(),
            "imx": self.imx_on.get(),
            "target": self.target_var.get(),
            "dev": self.dev_var.get(),
            "source": self.src_var.get(),
            "output": self.out_var.get(),
            "imatrix": self.imx_var.get(),
            "calib": self.calib_var.get(),
            "imx_out": self.imx_out_var.get(),
            "binaries": self.bin_var.get(),
        }

    def _save_state(self):
        try:
            STATE_PATH.write_text(json.dumps(self._collect_state(), indent=1))
        except Exception:      # pragma: no cover
            pass

    def _load_state(self):
        # Populate widgets WITHOUT triggering the model rebuild: several of
        # these vars fire a trace/command that (synchronously) runs
        # _on_model_changed, which rebuilds the group comboboxes. Doing that
        # before _saved_tiers is populated would reset every chosen tier to
        # Q4_K -- exactly the 'selection does not persist' bug. The guard
        # lets _on_model_changed run once, cleanly, after load() returns
        # (see __init__).
        self._loading = True
        try:
            st = json.loads(STATE_PATH.read_text())
        except Exception:
            self._loading = False
            self.logq.put("[state] could not load saved state "
                          "(state.json missing or corrupted) -- "
                          "using defaults")
            return
        tiers = st.get("groups", {})
        if isinstance(st.get("layers"), str):
            self.layers_var.set(st["layers"])
        if isinstance(st.get("imx"), bool):
            self.imx_on.set(st["imx"])
        for key, var in (("target", self.target_var),
                         ("dev", self.dev_var),
                         ("source", self.src_var),
                         ("output", self.out_var),
                         ("imatrix", self.imx_var),
                         ("calib", self.calib_var),
                         ("imx_out", self.imx_out_var),
                         ("binaries", self.bin_var)):
            if isinstance(st.get(key), str):
                var.set(st[key])
        # remember per-group tiers; applied on the next group rebuild
        self._saved_tiers = {g: t for g, t in tiers.items()}
        # allow the explicit _on_model_changed() in __init__ to rebuild now
        self._loading = False

    def _on_close(self):
        self.tbl_cancel = True
        if self.proc_running:
            if not messagebox.askyesno(
                    "Quit",
                    "A process is still running.\n"
                    "Quit anyway? The background process\n"
                    "will continue in the console."):
                return
        self._save_state()
        self.root.destroy()

    # ------------------------------------------------------------------ UI
    def _build_group_frame(self, parent):
        f = ttk.LabelFrame(parent, text="Group                Quant type  ",
                           padding=6)
        f.pack(fill="both", expand=True)
        self.grp_frame = ttk.Frame(f)
        self.grp_frame.pack(fill="both", expand=True)
        self.grp_frame.columnconfigure(1, weight=1)
        self.group_vars = {}
        self.group_cbs = []

    def _build_tune_frame(self, parent):
        f = ttk.LabelFrame(parent,
                           text="                     Tuning                    ",
                           padding=8)
        f.pack(fill="both", expand=True)

        ov = ttk.LabelFrame(f, text="  Layer upgrades  ", padding=6)
        ov.pack(fill="x", pady=(0, 8))
        # Row 0: layers label, entry and the (none) count hint side by side,
        # kept together and left-aligned in their own sub-frame.
        r0 = ttk.Frame(ov)
        r0.grid(row=0, column=0, sticky="w")
        ttk.Label(r0, text="Layers:").pack(side="left")
        self.layers_var = tk.StringVar(value="")
        le = ttk.Entry(r0, textvariable=self.layers_var, width=12)
        le.pack(side="left", padx=(6, 6))
        le.bind("<KeyRelease>", lambda e: self._recompute())
        self._layer_hint = ttk.Label(r0, text="(none)", style="Hint.TLabel")
        self._layer_hint.pack(side="left", padx=(6, 0))
        # Row 1: explanation hint, full width underneath.
        hf = ttk.Frame(ov)
        hf.grid(row=1, column=0, columnspan=3, sticky="we")
        ttk.Label(hf, text="Bumps all tensors in specified layers by a "
                           "quant tier", style="Hint.TLabel").pack(
            side="left", anchor="w")

        ttk.Label(f, text="Target size on disk (GB):").pack(anchor="w",
                                                           pady=(6, 0))
        tf = ttk.Frame(f)
        tf.pack(fill="x", pady=(2, 0))
        self.target_var = tk.StringVar(value="")
        te = ttk.Entry(tf, textvariable=self.target_var, width=8)
        te.pack(side="left")
        te.bind("<KeyRelease>", lambda e: self._save_state())
        self.btn_min = ttk.Button(tf, text="Minimise cosine deviation",
                                  command=self.minimise_cos)
        self.btn_min.pack(side="left", padx=(8, 0))
        self._target_hint = ttk.Label(f, text="", style="Hint.TLabel")
        self._target_hint.pack(anchor="w")

        ttk.Label(f, text="Target cosine deviation:").pack(anchor="w",
                                                          pady=(6, 0))
        kf = ttk.Frame(f)
        kf.pack(fill="x", pady=(2, 0))
        self.dev_var = tk.StringVar(value="0.001")
        ke = ttk.Entry(kf, textvariable=self.dev_var, width=8)
        ke.pack(side="left")
        ke.bind("<KeyRelease>", lambda e: self._save_state())
        self.btn_minsize = ttk.Button(kf, text="Minimise disk size",
                                      command=self.minimise_size)
        self.btn_minsize.pack(side="left", padx=(8, 0))
        ttk.Label(f, text="Minimises byte size of groups/layers for cosine\ndeviation", style="Hint.TLabel").pack(
            anchor="w", pady=(4, 0))

    def _rebuild_groups(self, groups, ladder):
        """(Re)create one combobox per present group, preserving any tier the
        user already chose for a group that still exists."""
        for w in self.grp_frame.winfo_children():
            w.destroy()
        self.group_vars = {}
        self.group_cbs = []
        for i, g in enumerate(groups):
            ttk.Label(self.grp_frame, text=g).grid(
                row=i, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=self._saved_tiers.get(g, "Q4_K"))
            if var.get() not in ladder:
                var.set("Q4_K")
            self.group_vars[g] = var
            self._saved_tiers[g] = var.get()
            cb = ttk.Combobox(self.grp_frame, textvariable=var, width=8,
                              values=ladder, state="readonly")
            cb.grid(row=i, column=1, sticky="we", padx=(14, 0), pady=1)
            cb.bind("<<ComboboxSelected>>", lambda e: self._recompute())
            self.group_cbs.append(cb)

    def _set_ladder(self, ladder):
        for cb in self.group_cbs:
            cb.config(values=ladder)
            if cb["textvariable"].get() not in ladder:
                cb["textvariable"].set("Q4_K")

    def _build_right_frame(self, parent):
        f = ttk.LabelFrame(parent, text="  Model  ", padding=10)
        f.pack(fill="both", expand=True)
        f.columnconfigure(0, weight=0)   # widened by the long target labels
        f.columnconfigure(1, weight=1)
        f.columnconfigure(2, weight=0)

        # The three path rows (Source/Output/Imatrix) live in their own
        # sub-grid whose label column is only as wide as those (short)
        # labels -- so the labels sit at the left edge and their boxes are
        # flush beside them, unaffected by the longer 'Target size ...'
        # labels in the panel below.
        p = ttk.Frame(f)
        p.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 6))
        p.columnconfigure(0, weight=0)
        p.columnconfigure(1, weight=1)
        p.columnconfigure(2, weight=0)

        def path_row(row, label, default):
            var = tk.StringVar(value=default)
            ttk.Label(p, text=label + ":").grid(
                row=row, column=0, sticky="w", pady=2)
            e = ttk.Entry(p, textvariable=var)
            e.grid(row=row, column=1, sticky="we", padx=(6, 0), pady=2)
            return var

        self.src_var = path_row(0, "Source", DEFAULTS["source"])
        self.src_var.trace_add("write", lambda *a: self._schedule_model())
        ttk.Button(p, text="...", width=3, style="Small.TButton",
                   command=lambda: self._browse_gguf(self.src_var,
                                                     "GGUF source model"
                                                     )).grid(
            row=0, column=2, sticky="e", padx=(4, 0), pady=2)
        self.out_var = path_row(1, "Output", DEFAULTS["output"])
        self.imx_var = path_row(2, "Imatrix", DEFAULTS["imatrix"])
        self.imx_var.trace_add("write", lambda *a: self._on_table_changed())
        ttk.Button(p, text="...", width=3, style="Small.TButton",
                   command=lambda: self._browse_gguf(self.imx_var,
                                                     "GGUF imatrix"
                                                     )).grid(
            row=2, column=2, sticky="e", padx=(4, 0), pady=2)
        # Folder holding the llama.cpp binaries (ggml-base.dll, llama-
        # quantize.exe, llama-imatrix.exe). Empty = auto-search.
        self.bin_var = path_row(3, "Binaries", DEFAULTS["binaries"])
        ttk.Button(p, text="...", width=3, style="Small.TButton",
                   command=self._browse_bin_dir).grid(
            row=3, column=2, sticky="e", padx=(4, 0), pady=2)
        ttk.Label(p, text="ggml-base.dll + llama-quantize.exe + "
                          "llama-imatrix.exe",
                  style="Hint.TLabel").grid(
            row=4, column=1, sticky="w", pady=(0, 2))

        ttk.Separator(f, orient="horizontal").grid(row=1, column=0,
                                                   columnspan=2,
                                                   sticky="we", pady=10)

        self.size_lbl = tk.Label(f, text="Expected size: -",
                                 font=("Segoe UI", 13, "bold"),
                                 wraplength=340, justify="left")
        self.size_lbl.grid(row=2, column=0, columnspan=2, sticky="w")
        self.dev_lbl = tk.Label(f, text="Expected cosine deviation: -",
                                font=("Segoe UI", 13, "bold"),
                                wraplength=340, justify="left")
        self.dev_lbl.grid(row=4, column=0, columnspan=2, sticky="w",
                          pady=(6, 0))
        self.err_lbl = ttk.Label(f, text="", foreground="#b00020",
                                 wraplength=340, justify="left")
        self.err_lbl.grid(row=5, column=0, columnspan=2, sticky="w")
        # Fast K-ladder table only (Q2_K to Q8_0 + F16, no IQ_X tiers):
        # there is no table-type choice to make, just the build button.
        bk = ttk.Frame(f)
        bk.grid(row=7, column=0, columnspan=2, sticky="we", pady=(8, 0))
        self.btn_table = ttk.Button(bk, text="Build table",
                                    style="Small.TButton",
                                    command=self.build_table)
        self.btn_table.pack(side="left")
        self.tbl_lbl = ttk.Label(bk,
                                 text="", style="Hint.TLabel",
                                 justify="left", wraplength=340)
        self.tbl_lbl.pack(side="left", padx=(8, 0))

        # Imatrix generation section
        ig = ttk.LabelFrame(f, text="  Imatrix generation (requires calibration data) ", padding=6)
        ig.grid(row=9, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ig.columnconfigure(1, weight=1)

        ttk.Label(ig, text="Calib:").grid(row=0, column=0, sticky="w", pady=2)
        self.calib_var = tk.StringVar(value=DEFAULTS["calib"])
        ttk.Entry(ig, textvariable=self.calib_var).grid(
            row=0, column=1, sticky="we", padx=(6, 0), pady=2)
        ttk.Button(ig, text="...", width=3, style="Small.TButton",
                   command=self._browse_calib).grid(
            row=0, column=2, sticky="e", padx=(4, 0), pady=2)

        ttk.Label(ig, text="Output:").grid(row=1, column=0, sticky="w", pady=2)
        self.imx_out_var = tk.StringVar(value=DEFAULTS["imx_out"])
        ttk.Entry(ig, textvariable=self.imx_out_var).grid(
            row=1, column=1, sticky="we", padx=(6, 0), pady=2)
        ttk.Button(ig, text="...", width=3, style="Small.TButton",
                   command=self._browse_imx_out).grid(
            row=1, column=2, sticky="e", padx=(4, 0), pady=2)

        ir = ttk.Frame(ig)
        ir.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.btn_imatrix = ttk.Button(ir, text="Make imatrix",
                                      style="Small.TButton",
                                      command=self.make_imatrix)
        self.btn_imatrix.pack(side="left")

        imx_frame = ttk.Frame(f)
        imx_frame.grid(row=10, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.imx_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(imx_frame, text="Use imatrix during quantisation (recommended)",
                        variable=self.imx_on,
                        command=self._on_table_changed).pack(side="left")

        bf = ttk.Frame(f)
        bf.grid(row=11, column=0, columnspan=2, sticky="we", pady=(6, 0))
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(2, weight=1)
        self.btn_quant = ttk.Button(bf, text="Make custom quant",
                                    style="Small.TButton",
                                    command=self.make_quant)
        self.btn_quant.grid(row=0, column=0, sticky="we", padx=3)
        self.btn_min_quant = ttk.Button(bf, text="Make dynamic quant",
                                        style="Small.TButton",
                                        command=self.make_min_cos_quant)
        self.btn_min_quant.grid(row=0, column=2, sticky="we", padx=3)
        self.all_btns = [self.btn_quant,
                         self.btn_min_quant,
                         self.btn_min, self.btn_minsize, self.btn_table,
                         self.btn_imatrix]

        hb = ttk.Frame(f)
        hb.grid(row=13, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ln1 = ttk.Frame(hb)
        ln1.pack(fill="x", anchor="w")
        ttk.Label(ln1, text="Custom quant - as specified above in Quant type/Tuning.",
                  style="Hint.TLabel").pack(side="left", anchor="w")
        ln2 = ttk.Frame(hb)
        ln2.pack(fill="x", anchor="w", pady=(2, 0))
        ttk.Label(ln2, text="Dynamic quant - solves dynamically at the "
                            "tensor level for target size (most accurate). "
                            , style="Hint.TLabel").pack(
            side="left", anchor="w")

    def _build_log(self, parent):
        # Bottom container holding the log panel + the show/hide toggle
        bottom = ttk.Frame(parent)
        bottom.pack(fill="x")
        self.log = scrolledtext.ScrolledText(bottom, height=12, width=90,
                                             state="disabled",
                                             font=("Consolas", 9),
                                             bg="#1e1e1e", fg="#d4d4d4",
                                             insertbackground="#d4d4d4")
        self.log.pack(fill="both", expand=True)
        self._log_visible = False         # hidden until the user opts in
        self.log.pack_forget()            # start collapsed
        self.btn_term = ttk.Button(bottom, text="Show terminal",
                                   style="Small.TButton",
                                   command=self._toggle_log)
        self.btn_term.pack(fill="x", pady=(4, 0))

    def _toggle_log(self):
        """Pop the terminal out at / hide it again."""
        self._log_visible = not self._log_visible
        if self._log_visible:
            self.log.pack(fill="both", expand=True)
            self.btn_term.config(text="Hide terminal")
        else:
            self.log.pack_forget()
            self.btn_term.config(text="Show terminal")

    # ------------------------------------------------------------- browse
    def _browse_gguf(self, var, title):
        """Native file picker (Windows Explorer / macOS panel / GTK on
        Linux -- one call, no per-platform code). Setting the var fires
        the existing trace, so the model loads exactly as if typed."""
        cur = var.get().strip()
        curdir = os.path.dirname(cur)
        init = curdir if (cur and os.path.isdir(curdir)) else str(APP_DIR)
        p = filedialog.askopenfilename(
            parent=self.root, title=title, initialdir=init,
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")])
        if p:
            var.set(p)

    def _browse_calib(self):
        cur = self.calib_var.get().strip()
        curdir = os.path.dirname(cur)
        init = curdir if (cur and os.path.isdir(curdir)) else str(APP_DIR)
        p = filedialog.askopenfilename(
            parent=self.root, title="Select calibration data file",
            initialdir=init, filetypes=[("Text files", "*.txt"),
                                         ("All files", "*.*")])
        if p:
            self.calib_var.set(p)

    def _browse_bin_dir(self):
        p = filedialog.askdirectory(
            parent=self.root,
            title="Select the folder with the llama.cpp binaries",
            initialdir=str(APP_DIR))
        if p:
            self.bin_var.set(p)

    def _browse_imx_out(self):
        cur = self.imx_out_var.get().strip()
        curdir = os.path.dirname(cur)
        init = curdir if (cur and os.path.isdir(curdir)) else str(APP_DIR)
        p = filedialog.asksaveasfilename(
            parent=self.root, title="Save imatrix as",
            initialdir=init, defaultextension=".dat",
            filetypes=[("dat files", "*.dat"), ("All files", "*.*")])
        if p:
            self.imx_out_var.set(p)

    # ------------------------------------------- model + table switching
    def _active_kind(self):
        return "imatrix" if self.imx_on.get() else "plain"

    def _active_ladder(self):
        t = q._load_cos(self._active_kind())
        if t is not None:
            return t["ladder"]
        # Fast K-ladder only: the app never builds/uses a full (IQ_X) table.
        return q.ladder_for(self._active_kind() == "imatrix")

    def _schedule_model(self):
        """Debounce the source field so the GGUF is only read once per
        settled path (loading is the one slow step)."""
        if self._src_after is not None:
            self.root.after_cancel(self._src_after)
        self._src_after = self.root.after(450, self._on_model_changed)

    def _on_model_changed(self):
        # no-op while restoring persisted state -- see _load_state
        if self._loading:
            return
        src = self.src_var.get().strip()
        key = (src, self.imx_var.get().strip(), self.imx_on.get())
        if src and Path(src).exists():
            st = Path(src).stat()
            key = key + (int(st.st_mtime),)
        if key == self._model_src:
            self._recompute()
            return
        self._model_src = key
        imx = self.imx_var.get().strip()
        # Fast K-ladder only (the app never selects a full/IQ_X table).
        q.set_build_kind("k")
        m = q.set_table_source(src if (src and Path(src).exists()) else None,
                               imx if (imx and Path(imx).exists()) else None)
        # the model changed: clear any stale display + validate the saved
        # layer range against the new layer count (it is model-specific)
        self.size_lbl.config(text="Expected size: -")
        self.dev_lbl.config(text="Expected cosine deviation: -")
        self.err_lbl.config(text="")
        if m is not None:
            self.root.title("Dynamic Quant Maker" )
            self._rebuild_groups(m["GROUPS"], self._active_ladder())
            self._layer_hint.config(text=f"(0-{m['n_layer']}) e.g. 0,5,7,12")
            lv = self.layers_var.get().strip()
            if lv:
                try:
                    q.parse_layers(lv)
                except ValueError:
                    self.layers_var.set("")   # out of range for this model
        else:
            self.root.title("Quant Maker")
            self._rebuild_groups([], [])
            self._layer_hint.config(text="(none)  ")
        self._update_size_hint()
        self._table_status()
        self._recompute()

    def _on_table_changed(self):
        self._model_src = None          # force the ladder/values refresh
        self._on_model_changed()

    def _update_size_hint(self):
        lo, hi = q.size_bounds(self._active_kind())
        self._target_hint.config(
            text=(f"Minimises cosine deviation for groups/layers\nfor a target GB size"
                  if lo is not None else "no table yet -- 'Build table'"))

    def _table_status(self):
        if not Path(self.src_var.get().strip()).exists():
            self.tbl_lbl.config(text="No source -- fill in the Source "
                                     "field, then 'Build table'")
            return
        p = q.find_cos_table(self._active_kind())
        if p is None:
            self.tbl_lbl.config(text="No cosine table for this source -- "
                                     "press 'Build table' (then solve)")
        else:
            self.tbl_lbl.config(text=f"Built cosine table: {p.name}")

    # ------------------------------------------------------------- sizing
    def _current_config(self):
        """(rows, error_message); error is '' when ok."""
        tiers = {g: v.get() for g, v in self.group_vars.items()}
        try:
            layers = q.parse_layers(self.layers_var.get())
            rows = q.build_assignment(tiers, layers, 1)
            return rows, ""
        except (ValueError, tk.TclError) as e:
            return None, str(e)

    def _recompute(self, _=None):
        if not Path(self.src_var.get().strip()).exists():
            self.size_lbl.config(text="Expected size: -")
            self.dev_lbl.config(text="Expected cosine deviation: -")
            return
        rows, err = self._current_config()
        self.err_lbl.config(text=err)
        if rows is None:
            return
        # size_estimate uses the table's MEASURED per-tensor bytes (C) plus
        # the baked-in frozen bytes (c0) and the container overhead -- so it
        # matches the actual llama-quantize output. expected_bytes() would
        # show only the pure weight and under-report by ~the overhead, which
        # is why quantized files used to come out 'slightly larger than'
        # expected.
        size_gb = q.size_estimate(rows, self.imx_on.get())
        if size_gb is None:
            self.size_lbl.config(text="Expected size: - (build table first)")
        else:
            self.size_lbl.config(text=f"Expected size: {size_gb:.3f} GB")
        cos, dev, note = q.cos_estimate(rows, self.imx_on.get())
        if cos is None:
            self.dev_lbl.config(text="Expected cosine deviation: n/a")
            self.err_lbl.config(text=note or self.err_lbl.cget("text"))
            return
        self.dev_lbl.config(text=f"Expected cosine deviation: {dev:.9f}")
        self._save_state()

    # ------------------------------------------------------------- schema
    def _log(self, msg):
        self.logq.put(msg)

    def _write_schema(self, rows):
        try:
            SCHEMA.write_text(q.schema_text(rows))
            return True
        except OSError as e:
            self.err_lbl.config(
                text=f"Could not write {SCHEMA.name}: {e}")
            return False

    # ------------------------------------------------- validation helpers
    def _confirm_overwrite(self, path, label):
        """Ask before overwriting an existing file. True = proceed."""
        if not Path(path).exists():
            return True
        return messagebox.askyesno(
            "Overwrite file",
            f"{label} already exists:\n{path}\n\nOverwrite?")

    def _show_busy(self, what):
        """Tell the user a process is already running."""
        messagebox.showinfo(
            "Already running",
            f"{what} is already in progress.\n"
            f"Please wait for it to finish\n(see the log panel).")

    def _check_numpy(self, solver_name):
        """Check if the numpy-backed solvers are available. Returns True."""
        if qsolve is None or qfullsolve is None:
            if not self._numpy_warned:
                self._numpy_warned = True
                messagebox.showinfo(
                    "NumPy not installed",
                    f"{solver_name} requires numpy.\n\n"
                    f"Install it with:\n    pip install numpy\n\n"
                    f"Then restart Quant Maker.")
            return False
        return True

    # ---------------------------------------------------------- minimise
    def _target_range_ok(self):
        lo, hi = q.size_bounds(self._active_kind())
        if lo is None:
            messagebox.showwarning(
                "No cosine table",
                "No cosine table for this source --\n"
                "press 'Build table' first.")
            return None
        return lo, hi

    def minimise_cos(self):
        if self.proc_running:
            self._show_busy("Minimisation")
            return
        if not self._check_numpy("The minimiser solver"):
            return
        rng = self._target_range_ok()
        if rng is None:
            return
        lo, hi = rng
        raw = self.target_var.get().strip()
        try:
            target = float(raw)
        except ValueError:
            self.err_lbl.config(text=(
                f"'{raw}' is not a valid number.\n"
                f"Enter a target size in GB between "
                f"{lo:.2f} and {hi:.2f}."))
            return
        if not (lo <= target <= hi):
            self.err_lbl.config(text=(
                f"Target {target:g} GB is outside the feasible range "
                f"({lo:.2f}\u2013{hi:.2f} GB).\n"
                f"Min = all tensors at lowest tier,\n"
                f"max = all tensors at highest tier."))
            return
        self.err_lbl.config(text="")
        imx = self.imx_on.get()
        self._log(f"[solve ] target {target:g} GB "
                  f"({'imatrix' if imx else 'no imatrix'}) ...")
        try:
            res, err = qsolve.solve(target, imx)
        except Exception as e:      # noqa: BLE001
            self.err_lbl.config(text=f"solve error: {e}")
            return
        if res is None:
            self.err_lbl.config(text=err)
            return
        for g, tier in res["tiers"].items():
            if g in self.group_vars:
                self.group_vars[g].set(tier)
        self.layers_var.set(", ".join(str(L) for L in res["layers"]))
        self._recompute()
        self._log(f"[solve ] result {res['size_gb']:.3f} GB  "
                  f"cos {res['cos']:.9f}  dev {res['dev']:.9f}")
        self._log("[solve ] " + "  ".join(
            f"{g}={res['tiers'][g]}" for g in self.group_vars
            if g in res["tiers"]))
        if res["layers"]:
            self._log("[solve ] bump layers (1 step): "
                      + ", ".join(map(str, res["layers"])))

    def minimise_size(self):
        """Find the SMALLEST build reaching a target deviation (dev =
        1 - cosine). Runs in a worker thread."""
        if self.proc_running:
            self._show_busy("Minimisation")
            return
        if not self._check_numpy("The minimiser solver"):
            return
        if self._target_range_ok() is None:
            return
        raw = self.dev_var.get().strip()
        try:
            dev = float(raw)
        except ValueError:
            self.err_lbl.config(text=(
                f"'{raw}' is not a valid number.\n"
                f"Enter a cosine deviation between 0.00001 and 0.2,\n"
                f"e.g. 0.001 for 0.1% deviation."))
            return
        if not (1e-5 <= dev <= 0.2):
            self.err_lbl.config(text=(
                f"Target deviation {dev} is outside 0.00001\u20130.2.\n"
                f"Deviation = 1 \u2212 cosine similarity.\n"
                f"0.001 = 0.1% deviation (typical)."))
            return
        target_cos = 1.0 - dev
        self.err_lbl.config(text="")
        for b in self.all_btns:
            b.config(state="disabled")
        self.btn_minsize.config(text="Solving ...")
        imx = self.imx_on.get()
        self._log(f"[solve ] minimising size for dev {dev:g} "
                  f"(cos {target_cos:.9f}, "
                  f"{'imatrix' if imx else 'no imatrix'}) ...")

        def work():
            res, err = qsolve.solve_min_size(target_cos, imx)
            self.jobq.put(("minds", res, err))
        threading.Thread(target=work, daemon=True).start()

    def _finish_min_size(self, res, err):
        self.btn_minsize.config(text="Minimise size")
        if res is None:
            self.err_lbl.config(text=err)
            self._log(f"[solve ] FAILED: {err}")
            for b in self.all_btns:
                b.config(state="normal")
            return
        for b in self.all_btns:
            b.config(state="normal")
        for g, tier in res["tiers"].items():
            if g in self.group_vars:
                self.group_vars[g].set(tier)
        self.layers_var.set(", ".join(str(L) for L in res["layers"]))
        self._recompute()
        self._log(f"[solve ] result {res['size_gb']:.3f} GB  "
                  f"cos {res['cos']:.9f}  dev {res['dev']:.9f}  "
                  f"({res['method']})")
        self._log("[solve ] " + "  ".join(
            f"{g}={res['tiers'][g]}" for g in self.group_vars
            if g in res["tiers"]))
        self._log("[solve ] bump " +
                  (", ".join(map(str, res["layers"])) if res["layers"]
                   else "no layer bumps") + " (1 step)")

    # ------------------------------------------------- full per-tensor solve
    def _full_solve_then(self):
        if self.proc_running:
            self._show_busy("Per-tensor solve")
            return
        if not self._check_numpy("The per-tensor solver"):
            return
        out = self.out_var.get().strip()
        if not self._confirm_overwrite(out, "Output file"):
            return
        rng = self._target_range_ok()
        if rng is None:
            return
        lo, hi = rng
        raw = self.target_var.get().strip()
        if not raw:
            messagebox.showerror(
                "No target size",
                "The 'Target size on disk (GB)' field is empty.\n\n"
                f"Enter a target size in GB between {lo:.2f} and "
                f"{hi:.2f}.\n\n"
                "Tip: fill it in first with 'Minimise cosine\n"
                "deviation', then press 'Make dynamic quant'.")
            return
        try:
            target = float(raw)
        except ValueError:
            self.err_lbl.config(
                f"enter a target size in GB ({lo:.2f}..{hi:.2f})")
            return
        if not (lo <= target <= hi):
            self.err_lbl.config(f"target outside {lo:.2f}..{hi:.2f} GB")
            return
        if not self._validate_quant_paths():
            return
        self.err_lbl.config(text="")
        for b in self.all_btns:
            b.config(state="disabled")
        self.btn_min_quant.config(text="Solving ...")
        imx = self.imx_on.get()
        self._log(f"[full  ] solving {target:g} GB per-tensor "
                  f"({'imatrix' if imx else 'no imatrix'}) ...")

        def work():
            res, err = qfullsolve.solve(target, imx)
            self.jobq.put(("full", res, err))
        threading.Thread(target=work, daemon=True).start()

    def _finish_full_solve(self, res, err):
        self.btn_min_quant.config(text="Make dynamic quant")
        if res is None:
            self.err_lbl.config(text=err)
            self._log(f"[full  ] FAILED: {err}")
            for b in self.all_btns:
                b.config(state="normal")
            return
        rows = [(n, tier, q.NAME_NEL.get(n, 0))
                for n, tier in res["assignment"].items()]
        if not self._write_schema(rows):
            for b in self.all_btns:
                b.config(state="normal")
            return
        self._log(f"[schema] wrote {SCHEMA}  ({len(rows)} tensors, "
                  f"{q.gb(q.expected_bytes(rows)):.3f} GB)")
        out = self.out_var.get().strip()
        if not self._confirm_overwrite(out, "Output file"):
            for b in self.all_btns:
                b.config(state="normal")
            return
        self._show_per_tensor(rows, res)
        self._start_quantize()

    def make_min_cos_quant(self):
        self._full_solve_then()

    def _show_per_tensor(self, rows, res):
        size_gb = q.size_estimate(rows, self.imx_on.get())
        if size_gb is None:
            self.size_lbl.config(text="Expected size: - (build table first)")
        else:
            self.size_lbl.config(
                text=f"Expected size: {size_gb:.3f} GB")
        cos, dev, note = q.cos_estimate(rows, self.imx_on.get())
        if cos is not None:
            self.dev_lbl.config(text=f"Expected cosine deviation: {dev:.9f}")
        self._log(f"[full  ] {res['size_gb']:.3f} GB  cos {res['cos']:.9f}  "
                  f"dev {res['dev']:.9f}  method {res['method']}")
        for g in q.GROUPS:
            if g in res["group_counts"]:
                dist = " ".join(
                    f"{t}x{c}" for t, c in sorted(
                        res["group_counts"][g].items(),
                        key=lambda kv: (q.LADDER.index(kv[0])
                                        if kv[0] in q.LADDER else 99)))
                self._log(f"[full  ] {g:12s} {dist}")

    # --------------------------------------------------------------- quant
    def _bin_dir(self):
        d = self.bin_var.get().strip()
        return d if d else None

    def _quantize_exe(self):
        return _find_bin("llama-quantize", self._bin_dir())

    def _imatrix_exe(self):
        return _find_bin("llama-imatrix", self._bin_dir())

    def _build_command(self):
        cmd = [str(self._quantize_exe() or "llama-quantize")]
        if self.imx_on.get():
            cmd += ["--imatrix", self.imx_var.get().strip()]
        cmd += ["--tensor-type-file", str(SCHEMA),
                self.src_var.get().strip(), self.out_var.get().strip(),
                "Q4_K"]     # placeholder; the schema sets every tensor
        return cmd

    def _validate_quant_paths(self):
        imx = self.imx_var.get().strip()
        if self.imx_on.get():
            if not imx:
                self.err_lbl.config(text=(
                    "'Use imatrix' is checked but no imatrix file is "
                    "specified.\nFill in the Imatrix field or untick "
                    "the checkbox."))
                return False
            if not Path(imx).exists():
                self.err_lbl.config(
                    text=f"imatrix file not found: {imx}")
                return False
        src = self.src_var.get().strip()
        if not src:
            self.err_lbl.config(text="No source model selected")
            return False
        if not Path(src).exists():
            self.err_lbl.config(text=f"source file not found: {src}")
            return False
        out = self.out_var.get().strip()
        if not out:
            self.err_lbl.config(text="No output path set")
            return False
        if Path(out).is_dir():
            self.err_lbl.config(
                text=f"output path is a directory: {out}")
            return False
        out_parent = Path(out).parent
        if not out_parent.exists():
            self.err_lbl.config(
                text=f"output directory does not exist:\n{out_parent}")
            return False
        if self._quantize_exe() is None:
            self.err_lbl.config(text=(
                "llama-quantize not found.\n"
                "Set the 'Binaries dir' field (or put the binaries in "
                "<project>\\binaries / add them to PATH)."))
            return False
        return True

    def make_quant(self, rows=None):
        if self.proc_running:
            self._show_busy("Quantisation")
            return
        src = self.src_var.get().strip()
        if not src:
            messagebox.showwarning(
                "No source model",
                "Select a source GGUF model file first.\n"
                "Fill in the Source field or click '...' to browse.")
            return
        out = self.out_var.get().strip()
        if not self._confirm_overwrite(out, "Output file"):
            return
        if rows is None:
            rows, err = self._current_config()
            if rows is None:
                return
        if not self._validate_quant_paths():
            return
        if not self._write_schema(rows):
            return
        self._log(f"[schema] wrote {SCHEMA}  ({len(rows)} tensors, "
                  f"{q.gb(q.expected_bytes(rows)):.3f} GB)")
        self._start_quantize()

    def _start_quantize(self):
        cmd = self._build_command()
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self._log(f"[quant ] {cmd_str}")
        for b in self.all_btns:
            b.config(state="disabled")
        self.proc_running = True
        threading.Thread(target=self._run_quantize, args=(cmd,),
                         daemon=True).start()

    def _run_quantize(self, cmd):
        try:
            proc = subprocess.Popen(cmd, cwd=str(APP_DIR),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    errors="replace")
            for line in proc.stdout:
                self._log(line.rstrip())
            rc = proc.wait()
            self._log(f"[quant ] exit code {rc}")
            if rc == 0:
                self._log(f"[quant ] done -> {self.out_var.get()}")
            else:
                self._log(f"[quant ] FAILED (rc={rc})")
                self.jobq.put(("quant_fail", rc))
        except Exception as e:      # noqa: BLE001
            self._log(f"[quant ] ERROR: {e}")
            self.jobq.put(("quant_fail", str(e)))
        finally:
            self.proc_running = False
            self.jobq.put(("quant_done",))

    # ---------------------------------------------------------- imatrix gen
    def _build_imatrix_cmd(self):
        cmd = [str(self._imatrix_exe() or "llama-imatrix")]
        cmd += ["-m", self.src_var.get().strip()]
        cmd += ["-f", self.calib_var.get().strip()]
        cmd += ["-o", self.imx_out_var.get().strip()]
        cmd += ["--output-format", "dat"]
        return cmd

    def make_imatrix(self):
        if self.proc_running:
            self._show_busy("Imatrix generation")
            return
        src = self.src_var.get().strip()
        calib = self.calib_var.get().strip()
        out = self.imx_out_var.get().strip()
        if not src:
            messagebox.showwarning(
                "No source model",
                "Select a source GGUF model file first.\n"
                "Fill in the Source field or click '...' to browse.")
            return
        if not Path(src).exists():
            self.err_lbl.config(text=f"source file not found: {src}")
            return
        if not calib:
            messagebox.showwarning(
                "No calibration file",
                "Select a calibration text file first.\n"
                "Fill in the Calib field or click '...' to browse.")
            return
        if not Path(calib).exists():
            self.err_lbl.config(
                text=f"calibration file not found: {calib}")
            return
        try:
            if Path(calib).stat().st_size == 0:
                self.err_lbl.config(
                    text="calibration file is empty (0 bytes)")
                return
        except OSError:
            pass
        if not out:
            self.err_lbl.config(text=(
                "No imatrix output path set.\n"
                "Fill in the Imatrix Output field "
                "or click '...' to browse."))
            return
        if Path(out).is_dir():
            self.err_lbl.config(
                text=f"imatrix output path is a directory: {out}")
            return
        out_parent = Path(out).parent
        if not out_parent.exists():
            self.err_lbl.config(
                text=f"imatrix output directory does not exist:\n"
                     f"{out_parent}")
            return
        if self._imatrix_exe() is None:
            self.err_lbl.config(text=(
                "llama-imatrix not found.\n"
                "Set the 'Binaries dir' field (or put the binaries in "
                "<project>\\binaries / add them to PATH)."))
            return
        if not self._confirm_overwrite(out, "Imatrix output file"):
            return
        cmd = self._build_imatrix_cmd()
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self._log(f"[imatrix] {cmd_str}")
        self.err_lbl.config(text="")
        for b in self.all_btns:
            b.config(state="disabled")
        self.btn_imatrix.config(text="Making ...")
        self.proc_running = True
        threading.Thread(target=self._run_imatrix, args=(cmd,),
                         daemon=True).start()

    def _run_imatrix(self, cmd):
        try:
            proc = subprocess.Popen(cmd, cwd=str(APP_DIR),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    errors="replace")
            for line in proc.stdout:
                self._log(line.rstrip())
            rc = proc.wait()
            self._log(f"[imatrix] exit code {rc}")
            if rc == 0:
                out = self.imx_out_var.get().strip()
                try:
                    sz = Path(out).stat().st_size / (1024 * 1024)
                    self._log(f"[imatrix] done -> {out} ({sz:.1f} MB)")
                    # auto-fill the imatrix path for quantisation
                    self.imx_var.set(out)
                except OSError:
                    self._log(f"[imatrix] done -> {out}")
            else:
                self._log(f"[imatrix] FAILED (rc={rc})")
                self.jobq.put(("imatrix_fail", rc))
        except Exception as e:      # noqa: BLE001
            self._log(f"[imatrix] ERROR: {e}")
            self.jobq.put(("imatrix_fail", str(e)))
        finally:
            self.proc_running = False
            self.jobq.put(("imatrix_done",))

    # ----------------------------------------------------------- table build
    def build_table(self):
        """Build the fast K-ladder cosine table (Q2_K to Q8_0 + F16, no
        IQ_X tiers) for the current source, in a worker thread. The K
        ladder never uses imatrix for the table (imatrix applies to the
        final llama-quantize step). Resumable, cancellable."""
        if self.proc_running:
            self._show_busy("Process")
            return
        if self.tbl_running:
            self._show_busy("Table build")
            return
        src = self.src_var.get().strip()
        # Fast K-ladder table only (Q2_K to Q8_0 + F16, no IQ_X tiers).
        # The K ladder never uses imatrix for the table -- imatrix applies
        # to the final llama-quantize step, not the K-ladder build.
        if not src:
            messagebox.showwarning(
                "No source model",
                "No source file selected.\n"
                "Fill in the Source field or click '...' to browse,\n"
                "then press Build table.")
            return
        if not Path(src).exists():
            self.err_lbl.config(text=f"source file not found: {src}")
            return
        try:
            with open(src, "rb") as f:
                magic = f.read(4)
            if magic != b"GGUF":
                messagebox.showwarning(
                    "Not a GGUF file",
                    f"The selected file does not look like a GGUF model.\n"
                    f"File: {src}\n\n"
                    f"Please select a .gguf model file.")
                return
        except OSError as e:
            self.err_lbl.config(text=f"cannot read source file: {e}")
            return
        kind = "k"
        out = tablestore.TABLES_DIR / tablestore.table_filename(
            src, False, kind)
        bin_dir = self._bin_dir()
        tablebuild.EXTRA_BIN_DIRS = [bin_dir] if bin_dir else []
        self.err_lbl.config(text="")
        for b in self.all_btns:
            b.config(state="disabled")
        self.btn_table.config(text="Building ...")
        self.tbl_running = True
        self.tbl_cancel = False
        self._log(f"[table ] building {out.name} (fast K ladder) "
                  f"from {src} -- resumable, see log")
        threading.Thread(target=self._run_table_build, args=(src, out),
                         daemon=True).start()

    def _run_table_build(self, src, out):
        # BaseException on purpose: besides ordinary errors this catches
        # SystemExit (a sys.exit inside a worker thread used to die here
        # silently, leaving the UI stuck on 'Building ...' with zero CPU)
        # and KeyboardInterrupt. table_done is ALWAYS queued so the UI
        # recovers either way. Fast K ladder only (no imatrix tiers).
        try:
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            dll = tablebuild.find_tool(["ggml-base.dll", "libggml-base.so",
                                        "libggml-base.dylib"], "ggml-base")
            lad = modeldef.LADDER_K
            n_thr = os.cpu_count() or 4
            self.logq.put(
                f"[table ] C++ fused engine, {n_thr} threads, "
                f"{len(lad)} tiers: {', '.join(lad)}")
            b = tablebuild2.Build(src, None, dll, lad, out, workers=n_thr,
                                  log=lambda m: self.logq.put(m))
            ok = b.run(cancel=lambda: self.tbl_cancel)
            self.jobq.put(("table_done", ok, out.name))
        except BaseException as e:      # noqa: BLE001
            import traceback
            self.logq.put(f"[table ] ERROR: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines()[-4:]:
                self.logq.put("    " + line)
            self.jobq.put(("table_done", False, out.name))

    # ------------------------------------------------------------ job pump
    def _poll_jobs(self):
        try:
            while True:
                job = self.jobq.get_nowait()
                if job[0] == "full":
                    self._finish_full_solve(job[1], job[2])
                elif job[0] == "minds":
                    self._finish_min_size(job[1], job[2])
                elif job[0] == "quant_fail":
                    self.err_lbl.config(text=(
                        f"Quantize failed ({job[1]}).\n"
                        f"Check the log panel for details "
                        f"(click 'Show terminal')."))
                elif job[0] == "quant_done":
                    self.btn_min_quant.config(text="Make dynamic quant")
                    for b in self.all_btns:
                        b.config(state="normal")
                elif job[0] == "imatrix_fail":
                    self.err_lbl.config(text=(
                        f"Imatrix generation failed ({job[1]}).\n"
                        f"Check the log panel for details "
                        f"(click 'Show terminal')."))
                elif job[0] == "imatrix_done":
                    self.btn_imatrix.config(text="Make imatrix")
                    for b in self.all_btns:
                        b.config(state="normal")
                elif job[0] == "table_done":
                    ok, name = job[1], job[2]
                    self.btn_table.config(text="Build table")
                    self.tbl_running = False
                    self._model_src = None     # re-discover the new table
                    q.set_table_source(self.src_var.get().strip(),
                                       self.imx_var.get().strip())
                    if ok:
                        self._log(f"[table ] done -> tables/{name}")
                    else:
                        self._log("[table ] FAILED -- checkpoint kept, "
                                  "press 'Build table' to resume")
                        self.err_lbl.config(
                            text="Table build failed -- press "
                                 "'Show terminal' for the error, then "
                                 "'Build table' to resume")
                    for b in self.all_btns:
                        b.config(state="normal")
                    self._on_model_changed()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_jobs)

    def _poll_log(self):
        try:
            while True:
                self.log.config(state="normal")
                self.log.insert("end", self.logq.get_nowait() + "\n")
                self.log.see("end")
                self.log.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
