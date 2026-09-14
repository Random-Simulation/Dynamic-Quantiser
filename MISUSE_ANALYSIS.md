# Quant Maker — Misuse Analysis & Error-Handling Design

## How errors are shown today
- **`err_lbl`** — a red inline label in the Model panel. Set by several handlers.
- **Silent `return`** — several guards (`proc_running`, `tbl_running`,
  `qsolve is None`, `qfullsolve is None`) just `return` with no feedback.
- **Log panel** — subprocess output appears here, but only if the user
  clicked "Show terminal" first.

---

## CATEGORY 1 — "Click a button, nothing happens" (silent guards)

These are the most dangerous: the user thinks nothing is wrong.

### 1.1 Clicking any action while a process is running
**Where:** `make_quant`, `make_imatrix`, `minimise_cos`, `minimise_size`,
`_full_solve_then`, `build_table` — all start with `if self.proc_running: return`.

**Problem:** User double-clicks "Make custom quant", or clicks another
button mid-build. The app silently ignores it. User thinks the app hung.

**Fix:** Show a `messagebox.showinfo` popup:
```
Title:  "Already running"
Body:   "A process is already in progress.
         Please wait for it to finish (see the log panel)."
```
Alternatively (less intrusive): flash the relevant button briefly, or
change the log line: `[busy ] another process is still running`.

---

### 1.2 "Minimise cosine deviation" / "Minimise disk size" clicked without numpy
**Where:** `minimise_cos` and `minimise_size` check `qsolve is None` and
`return` silently.

**Problem:** If numpy isn't installed, the buttons do nothing. The user
has no idea why.

**Fix:**
```
Title:  "NumPy not installed"
Body:   "The minimiser solvers require numpy.
         Install it with:  pip install numpy
         then restart Quant Maker."
```
(Only show this once — set a flag after the first occurrence so the user
isn't spammed.)

---

### 1.3 "Make dynamic quant" clicked without numpy
**Where:** `_full_solve_then` checks `qfullsolve is None` and `return`s
silently.

**Fix:** Same as 1.2, but mention "the per-tensor solver".

---

### 1.4 "Build table" clicked while a table is already building
**Where:** `build_table` checks `self.tbl_running` and `return`s.

**Fix:**
```
Title:  "Table build in progress"
Body:   "A table build is already running.
         It is resumable — you can cancel it by restarting the app,
         or wait for it to finish."
```

---

## CATEGORY 2 — Missing path / file validation

### 2.1 "Build table" with an empty Source field
**Where:** `build_table` — `Path(src).exists()` returns `False` for `""`,
so the error label shows `source not found: ` (empty string after the
colon). Confusing.

**Fix:** Distinguish the two cases:
```
if not src:
    # popup or err_lbl:
    "No source file selected. Fill in the Source field or click '...'
     to browse, then press Build table."
elif not Path(src).exists():
    "Source file not found:
     {src}
     Check the path and try again."
```

---

### 2.2 "Build table" with Source pointing to a non-GGUF file
**Where:** `build_table` only checks `Path(src).exists()`. A user could
point it at a `.txt` file.

**Problem:** The C++ table builder will crash or produce garbage.

**Fix:** Add a magic-byte check on the first 4 bytes of the file.
GGUF files start with `GGUF`. If it doesn't:
```
Title:  "Not a GGUF file"
Body:   "The selected source does not look like a GGUF model.
         File: {src}
         Please select a .gguf model file."
```

---

### 2.3 "Make custom quant" / "Make dynamic quant" with empty Output field
**Where:** `_validate_quant_paths` checks source and imatrix but never
checks that the output path is non-empty or that its parent directory
exists.

**Problem:** `llama-quantize` gets an empty output argument and either
crashes or writes to an unexpected location.

**Fix:** Add to `_validate_quant_paths`:
```
out = self.out_var.get().strip()
if not out:
    "No output path set. Fill in the Output field."
elif not Path(out).parent.exists():
    "Output directory does not exist:
     {Path(out).parent}
     Create it first or pick a different path."
elif Path(out).exists() and not messagebox.askyesno(
         "Overwrite", f"{out} already exists. Overwrite?"):
    return False   # user cancelled
```

---

### 2.4 "Make imatrix" with empty Calib field
**Where:** `make_imatrix` checks `not calib` and shows
`calib file not found: ` — same empty-string problem as 2.1.

**Fix:**
```
if not calib:
    "No calibration file selected. Fill in the Calib field."
elif not Path(calib).exists():
    "Calibration file not found: {calib}"
```

---

### 2.5 "Make imatrix" with empty output path
**Where:** `make_imatrix` checks `not out` — good. But the message
`enter an output imatrix path` is terse.

**Fix (minor):**
```
"No imatrix output path set. Fill in the Imatrix Output field
 or click '...' to browse."
```

---

### 2.6 "Make imatrix" output directory doesn't exist
**Where:** `make_imatrix` checks the output path is non-empty but not
that its parent directory exists.

**Fix:**
```
out_dir = Path(out).parent
if not out_dir.exists():
    "Imatrix output directory does not exist:
     {out_dir}"
```

---

### 2.7 "Make custom quant" with imatrix checkbox ON but imatrix path empty
**Where:** `_validate_quant_paths` checks
`Path(self.imx_var.get().strip()).exists()` — for an empty string this
returns `False`, so the error shows `imatrix not found: ` (empty).

**Fix:**
```
imx = self.imx_var.get().strip()
if self.imx_on.get():
    if not imx:
        "The 'Use imatrix' checkbox is on but no imatrix file is
         specified. Fill in the Imatrix field or untick the checkbox."
    elif not Path(imx).exists():
        "Imatrix file not found: {imx}"
```

---

### 2.8 "Make dynamic quant" with imatrix checkbox ON but no imatrix file
**Where:** `_validate_quant_paths` is called from `_full_solve_then`,
so this is partially covered. But the error message is the same
ambiguous `imatrix not found: ` (empty).

**Fix:** Same as 2.7.

---

### 2.9 "Build table" with "Full" build kind + imatrix ON but no imatrix file
**Where:** `build_table` checks `use_im and not Path(imx).exists()`.
Good, but the message is `imatrix not found: {imx}` with possibly
empty `{imx}`.

**Fix:** Same distinction as 2.7.

---

## CATEGORY 3 — Numeric field validation

### 3.1 "Minimise cosine deviation" — empty or non-numeric Target size
**Where:** `minimise_cos` catches `ValueError` from `float(...)`. Good.
But the message `enter a target size in GB ({lo:.2f}..{hi:.2f})`
doesn't say what the user actually typed.

**Fix:**
```
f"Target size '{self.target_var.get()}' is not a valid number.
   Enter a value in GB between {lo:.2f} and {hi:.2f}."
```

---

### 3.2 "Minimise cosine deviation" — target outside feasible range
**Where:** `minimise_cos` checks `lo <= target <= hi`. Good. But the
message doesn't explain *why* the range is what it is.

**Fix (minor):**
```
f"Target {target:g} GB is outside the feasible range
   ({lo:.2f}–{hi:.2f} GB for this model/table).
   The minimum is all-tensors-at-lowest-tier;
   the maximum is all-tensors-at-highest-tier."
```

---

### 3.3 "Minimise disk size" — empty or non-numeric dev value
**Where:** `minimise_size` catches `ValueError`. Message is
`enter a target dev, e.g. 0.001`.

**Fix:**
```
f"'{self.dev_var.get()}' is not a valid number.
   Enter a cosine deviation between 0.00001 and 0.2,
   e.g. 0.001 for 0.1% deviation."
```

---

### 3.4 "Minimise disk size" — dev value outside 1e-5..0.2
**Where:** `minimise_size` checks the range. Good.

**Fix (minor — explain the meaning):**
```
f"Target deviation {dev} is outside 0.00001–0.2.
   Deviation = 1 - cosine similarity.
   0.001 = 0.1% deviation (recommended for most models)."
```

---

### 3.5 Layers field — invalid syntax
**Where:** `parse_layers` raises `ValueError` on bad input like `abc`,
`-1`, or `5.5`. In `_current_config` this is caught and shown in
`err_lbl`. But the message from `parse_layers` is
`layers must be 0..63, got [70]` — technical.

**Fix:** Wrap with a friendlier message:
```
f"Layers field has invalid values: '{self.layers_var.get()}'.
   Use comma-separated integers or ranges, e.g. '0, 5-9, 12'.
   Valid range for this model: 0–{n_layer}."
```

Also: `parse_layers` calls `int(tok)` which will raise `ValueError` on
`abc` — but the error message would be `invalid literal for int()`.
Add a try/except inside the parse or validate before calling.

---

### 3.6 Layers field — layer number exceeds model's layer count
**Where:** `parse_layers` raises `ValueError` with
`layers must be 0..63, got [70]`.

**Fix:** Same as 3.5 — include the valid range hint.

---

### 3.7 Target size = 0 or negative
**Where:** `minimise_cos` — the range check `lo <= target <= hi` catches
negative and zero values (lo is always > 0 for a real model). But the
error message would be `target outside 0.50..4.20 GB` which is
confusing for `target = -5`.

**Fix:** Add an explicit check before the range check:
```
if target <= 0:
    "Target size must be a positive number of GB."
```

---

## CATEGORY 4 — State / context errors

### 4.1 "Make custom quant" with no model loaded (empty Source)
**Where:** `make_quant` → `_current_config()` → `build_assignment`
raises `ValueError("no source model loaded")` → caught in
`_current_config` → `rows` is `None` → `make_quant` returns silently.

**Problem:** The user clicks "Make custom quant" without ever selecting
a source. Nothing happens. No error message.

**Fix:** Before calling `_current_config`, check:
```
if not self.src_var.get().strip():
    messagebox.showwarning(
        "No source model",
        "Select a source GGUF model file first.\n"
        "Fill in the Source field or click '...' to browse.")
    return
```
Or at minimum set `err_lbl` to `"No source model selected."`

---

### 4.2 "Make custom quant" with source set but table not built
**Where:** `make_quant` doesn't require a table — it just writes the
schema and runs `llama-quantize`. But `_current_config` calls
`build_assignment` which calls `q.build_assignment` which needs the
model loaded (not the table). The size/cosine estimates will show
"(build table first)" but the quantize itself will work.

**This is actually fine** — the table is only needed for size/cosine
estimates and solvers, not for the actual quantize. But the user might
be confused why the size label says "- (build table first)" while the
quantize still proceeds.

**Fix (optional — informational):** Log a note:
```
[quant ] Note: no cosine table built yet — size estimates are unavailable.
         The quantization will proceed using the schema directly.
```

---

### 4.3 "Make dynamic quant" (full solve) with no table
**Where:** `_full_solve_then` → `_target_range_ok()` returns `None` if
no table → sets `err_lbl` to `no cosine table -- 'Build table' first`.
Good — this is handled.

But the message could be clearer:
```
"No cosine table exists for this model yet.
   Press 'Build table' first, wait for it to finish,
   then try again."
```

---

### 4.4 "Minimise cosine deviation" with no table
**Where:** `minimise_cos` → `_target_range_ok()` → same as 4.3. Good.

---

### 4.5 Switching model while a table build is in progress
**Where:** `_on_model_changed` is called when source changes. The table
build thread captured the source at start time, so it continues building
the old model's table. When it finishes, `_poll_jobs` calls
`_on_model_changed` again, which would re-discover the (now stale)
table.

**Problem:** The user changed the source mid-build. The completed table
is for the *old* model. The UI would show the old table as if it's for
the new model.

**Fix:** Track which source the build was started for. In
`_poll_jobs`'s `table_done` handler:
```
if ok and build_src == self.src_var.get().strip():
    # normal completion
else:
    self._log(f"[table ] finished building table for a previous "
              f"source — not applied to current model")
```
Or simpler: disable the Source field while a table is building
(`src_var` entry becomes read-only).

---

### 4.6 Changing imatrix checkbox while a table is building
**Where:** `imx_on` change triggers `_on_table_changed` →
`_on_model_changed`. But the build thread is still running with the
old setting.

**Fix:** Disable the imatrix checkbox while `tbl_running` is True.
Or, in the `table_done` handler, check if the current settings match
what was built:
```
if self.imx_on.get() != build_imx_on:
    self._log("[table ] table built with different imatrix setting "
              "— rebuild with current settings if needed")
```

---

## CATEGORY 5 — Binary / environment errors

### 5.1 `llama-quantize` not found
**Where:** `_validate_quant_paths` checks `QUANTIZE_EXE is None`.
Good. Message: `llama-quantize not found -- put it in <project>\binaries or PATH`.

**Fix (minor — more helpful):**
```
"llama-quantize was not found.

To fix:
  1. Download the llama.cpp binaries for your platform.
  2. Place 'llama-quantize' (or .exe) in the 'binaries' folder
     next to this app, OR add it to your system PATH.
  3. Restart Quant Maker."
```

---

### 5.2 `llama-imatrix` not found
**Where:** `make_imatrix` checks `IMATRIX_EXE is None`. Same pattern.

**Fix:** Same as 5.1, but for `llama-imatrix`.

---

### 5.3 Subprocess crash (segfault, missing DLL, etc.)
**Where:** `_run_quantize` and `_run_imatrix` catch `Exception` and
log it. But if the process segfaults, `proc.wait()` returns a non-zero
code and the log shows `exit code -11` or similar. The user sees this
in the log panel (if visible).

**Problem:** If the log panel is hidden (default), the user sees the
buttons re-enable but no explanation of what went wrong.

**Fix:** After `proc.wait()`, if `rc != 0`:
```
# In addition to the log line, also set err_lbl:
self.err_lbl.config(
    text=f"Process failed (exit code {rc}). "
         f"Check the log panel for details (click 'Show terminal').")
```
Or show a `messagebox.showerror` with the last N lines of output.

---

### 5.4 Output file already exists
**Where:** Not checked anywhere. `llama-quantize` will silently
overwrite an existing file.

**Fix:** Before starting quantize:
```
out = self.out_var.get().strip()
if Path(out).exists():
    if not messagebox.askyesno("Overwrite file",
            f"{out} already exists.\nOverwrite it?"):
        return
```

---

### 5.5 Output path is a directory (not a file)
**Where:** Not checked. If the user types a directory path as output,
`llama-quantize` will fail with a confusing error.

**Fix:**
```
if Path(out).is_dir():
    "Output path is a directory, not a file."
```

---

## CATEGORY 6 — Imatrix-specific edge cases

### 6.1 "Make imatrix" — calib file is empty (0 bytes)
**Where:** Not checked. `llama-imatrix` would fail with a cryptic error.

**Fix:**
```
calib_size = Path(calib).stat().st_size
if calib_size == 0:
    "Calibration file is empty (0 bytes)."
```

---

### 6.2 "Make imatrix" — source and calib are the same file
**Where:** Not checked. Makes no sense but `llama-imatrix` might handle
it or crash.

**Fix:**
```
if Path(src).resolve() == Path(calib).resolve():
    "Calibration file and source model are the same file.
     The calibration file should be a text file of sample
     prompts/utterances for the model."
```

---

### 6.3 "Make imatrix" output overwrites an existing imatrix
**Where:** Not checked. Silently overwrites.

**Fix:**
```
if Path(out).exists():
    if not messagebox.askyesno("Overwrite",
            f"{out} already exists. Overwrite?"):
        return
```

---

### 6.4 Imatrix file exists but is for a different model
**Where:** Not checked. The imatrix GGUF might have been generated
from a different source model.

**Fix (optional — hard to detect reliably):** Add a warning log line
when the imatrix file's mtime is older than the source file's mtime:
```
self._log("[warn ] imatrix file is older than the source model — "
          "it may have been generated from a different model")
```

---

### 6.5 "Use imatrix" checkbox ON but build kind is "k" (fast)
**Where:** `build_table` sets `use_im = imx_on and kind == "full"`.
So with "k" build kind, the imatrix checkbox is silently ignored.
The user might think the imatrix is being used.

**Fix:** When build kind is "k", either:
- Disable the imatrix checkbox (with a tooltip: "Imatrix is only used
  with the Full build kind"), or
- Show a hint next to the checkbox: "(not used with Fast build kind)"

In `make_quant` and `make_dynamic_quant`, the `imx_on` checkbox IS
respected (it adds `--imatrix` to the command). So there's an
inconsistency: table build ignores it for "k" but quantize uses it.
Document this or make it consistent.

---

## CATEGORY 7 — Concurrent / race conditions

### 7.1 User types in a field while a process is running
**Where:** No guard. The user could change the source field while
`llama-quantize` is running. The process captured the source at start
time, so it's fine functionally, but the UI labels (size, dev) would
update to reflect the new source while the old quantize is still
running. Confusing.

**Fix:** Disable the Source entry and the imatrix entry while
`proc_running` or `tbl_running` is True.

---

### 7.2 User closes the window while a process is running
**Where:** `_on_close` sets `tbl_cancel = True` and destroys the root.
The daemon threads will be killed. But `llama-quantize` (a subprocess)
is NOT killed — it continues running in the background.

**Fix:**
```
def _on_close(self):
    self.tbl_cancel = True
    if self.proc_running:
        if not messagebox.askyesno("Quit",
                "A process is still running. Quit anyway?
                 (The background process will continue.)"):
            return
    self._save_state()
    self.root.destroy()
```

---

### 7.3 User clicks "Build table" then immediately clicks "Make custom quant"
**Where:** `make_quant` checks `proc_running` but not `tbl_running`.
These are separate flags. If a table build is running, `proc_running`
is False, so `make_quant` proceeds. This is actually fine — the
quantize and table build are independent. But the user might be
confused.

**No fix needed** — this is correct behaviour.

---

## CATEGORY 8 — State persistence

### 8.1 `state.json` is corrupted (invalid JSON)
**Where:** `_load_state` has a broad `try/except` — good. Falls back
to defaults silently.

**Fix (minor):** Log a warning to the log panel:
```
[self ] Could not load saved state (state.json is corrupted).
       Using defaults.
```

---

### 8.2 `state.json` contains a path that no longer exists
**Where:** `_load_state` populates widgets with the saved paths.
`_on_model_changed` is called, which checks `Path(src).exists()`.
If the file is gone, the model isn't loaded. The UI shows "Expected
size: -".

**Fix (minor):** In `_on_model_changed`, if the saved source no longer
exists, clear the source field:
```
if src and not Path(src).exists():
    self._log(f"[state ] saved source no longer exists: {src}")
    # optionally: self.src_var.set("")
```

---

### 8.3 `state.json` writes fail (read-only filesystem, etc.)
**Where:** `_save_state` has a broad `try/except: pass`. Silent.

**Fix (minor):** Log a warning once:
```
[self ] Could not save state to {STATE_PATH}: {e}.
       Your settings will not be preserved.
```

---

## CATEGORY 9 — Schema / tensor edge cases

### 9.1 `schema.txt` write fails (read-only, disk full)
**Where:** `_write_schema` has no error handling. If the write fails,
the exception propagates up to `make_quant` which has no try/except
around it. The app would crash.

**Fix:**
```
def _write_schema(self, rows):
    try:
        SCHEMA.write_text(q.schema_text(rows))
        return True
    except OSError as e:
        self.err_lbl.config(text=f"Could not write schema file: {e}")
        return False
```

---

### 9.2 Empty group assignment (no model → no groups)
**Where:** `make_quant` with no model loaded → `_current_config`
returns `None` → silent return. Covered in 4.1.

But also: what if the model has groups but the user somehow has an
empty `group_vars` dict? This shouldn't happen in normal operation,
but as a safety check:
```
if not self.group_vars:
    "No tensor groups found for this model. The source file may
     be corrupted or not a valid GGUF model."
```

---

## CATEGORY 10 — Disk / resource issues

### 10.1 Not enough disk space for output
**Where:** Not checked. `llama-quantize` would fail with a cryptic
error.

**Fix (optional — hard to check portably):** After the quantize
process fails, add a hint to the error:
```
if rc != 0:
    # In addition to the exit code, add:
    self.err_lbl.config(
        text=f"Quantize failed (exit {rc}). "
             f"Check disk space and the log for details.")
```

---

### 10.2 Table build produces a very large .npz file
**Where:** Not checked. The table stores S, Q, C matrices for every
tensor × every tier. For a large model this could be several GB.

**Fix (optional):** Before starting the build, estimate the size:
```
# Rough estimate: n_tensors * n_tiers * 3 * 8 bytes (float64)
# Plus overhead
```
Show a warning if it exceeds, say, 5 GB.

---

## CATEGORY 11 — UI / workflow confusion

### 11.1 User doesn't know they need to build a table first
**Where:** The table status label says `No cosine table for this source
-- press 'Build table' (then solve)` but it's easy to miss.

**Fix:** When the user clicks a solve button and there's no table,
show a more prominent message (popup, not just the small label):
```
Title:  "Build table first"
Body:   "The solvers need a cosine table, which hasn't been built
         for this model yet.

         1. Press 'Build table'
         2. Wait for it to finish (see the log)
         3. Press the solve button again"
```

---

### 11.2 User doesn't understand the imatrix checkbox interaction
**Where:** The checkbox says "Use imatrix during quantisation
(recommended)" but:
- With "k" build kind, it's ignored for table building but used for
  quantize
- The imatrix file field is above the checkbox, not visually connected

**Fix:**
- Add a hint label below the checkbox:
  "(Requires an imatrix file generated by 'Make imatrix'.
   Only affects table building when 'Full' kind is selected.)"
- Or: when the checkbox is ticked but the imatrix field is empty,
  show a yellow warning label next to the imatrix field:
  "No imatrix file specified"

---

### 11.3 "Make custom quant" vs "Make dynamic quant" — user confusion
**Where:** The hint labels explain the difference but they're small
and at the bottom of the panel.

**Fix (minor):** Add tooltips to the buttons:
- "Make custom quant": "Uses the tier assignment from the Group
  panel above. Fast — just runs llama-quantize with your choices."
- "Make dynamic quant": "Solves the optimal per-tensor assignment
  to hit your target size. Slower but more accurate."

---

## CATEGORY 12 — Specific error message designs (summary table)

| # | Trigger | Current behaviour | Proposed fix |
|---|---------|-------------------|-------------|
| 1 | Any button while process running | Silent return | `showinfo` popup |
| 2 | Minimise buttons without numpy | Silent return | `showinfo` popup (once) |
| 3 | Dynamic quant without numpy | Silent return | `showinfo` popup (once) |
| 4 | Build table while table building | Silent return | `showinfo` popup |
| 5 | Build table, empty source | `err_lbl`: "source not found: " | Distinguish empty vs missing |
| 6 | Build table, non-GGUF file | C++ crash / garbage | Magic-byte check + popup |
| 7 | Quant, empty output | `llama-quantize` crash | Validate before running |
| 8 | Quant, output dir missing | `llama-quantize` crash | Validate parent dir |
| 9 | Quant, output exists | Silent overwrite | `askyesno` popup |
| 10 | Quant, imatrix ON, empty path | `err_lbl`: "imatrix not found: " | Distinguish empty vs missing |
| 11 | Imatrix, empty calib | `err_lbl`: "calib file not found: " | Distinguish empty vs missing |
| 12 | Imatrix, output dir missing | `llama-imatrix` crash | Validate parent dir |
| 13 | Imatrix, output exists | Silent overwrite | `askyesno` popup |
| 14 | Minimise, bad target number | `err_lbl` (ok) | Add user's input to message |
| 15 | Minimise, target out of range | `err_lbl` (ok) | Add explanation of range |
| 16 | Min-size, bad dev number | `err_lbl` (ok) | Add user's input to message |
| 17 | Layers, invalid syntax | `err_lbl` (ok) | Add valid-range hint |
| 18 | Quant, no source model | Silent return | `showwarning` popup |
| 19 | Solve, no table | `err_lbl` (ok) | Add "press Build table" hint |
| 20 | Source changed mid-build | Stale table applied | Track build source, warn |
| 21 | Imatrix ON + "k" build kind | Silently ignored | Disable checkbox or show hint |
| 22 | Close window mid-process | Daemon killed, subprocess orphaned | `askyesno` popup |
| 23 | schema.txt write fails | App crash | try/except + `err_lbl` |
| 24 | Subprocess crash, log hidden | User sees nothing | Set `err_lbl` on failure |
| 25 | Calib file is 0 bytes | `llama-imatrix` cryptic error | Check file size |
| 26 | Calib = source (same file) | Nonsense / crash | Check path equality |
| 27 | State file corrupted | Silent fallback | Log warning |
| 28 | Saved source no longer exists | UI shows "-" | Log warning, clear field |
| 29 | Output path is a directory | `llama-quantize` crash | Check `is_dir()` |
| 30 | Not enough disk space | `llama-quantize` cryptic error | Add hint on failure |

---

## CATEGORY 13 — Suggested implementation priority

**High priority (user-facing crashes / data loss):**
1. #5, #7, #10, #11 — empty path validations (prevent crashes)
2. #18 — no source before quant (prevent silent no-op)
3. #23 — schema write crash (prevent app crash)
4. #22 — close window mid-process (prevent orphaned subprocess)

**Medium priority (confusing UX):**
5. #1, #4, #22 — silent "already running" guards
6. #2, #3 — numpy missing (silent no-op)
7. #9, #13, #26 — overwrite confirmations
8. #6 — non-GGUF source file

**Low priority (polish):**
9. #14–17 — improve error message wording
10. #20–21 — mid-build state changes
11. #25, #27, #28, #30 — edge cases

---

## Suggested helper function

Most of these checks boil down to a few patterns. Consider adding:

```python
import tkinter.messagebox as mb

def _require(self, var, label, popup_title="Missing path"):
    """Validate a path field. Returns the stripped path or None."""
    p = var.get().strip()
    if not p:
        mb.showwarning(popup_title,
            f"No {label} specified.\nFill in the {label} field.")
        return None
    if not Path(p).exists():
        mb.showwarning(f"{label} not found",
            f"{label} file does not exist:\n{p}")
        return None
    return p

def _confirm_overwrite(self, path, label="Output file"):
    """Ask before overwriting. Returns True if OK to proceed."""
    if not Path(path).exists():
        return True
    return mb.askyesno("Overwrite",
        f"{label} already exists:\n{path}\n\nOverwrite?")
```

This would make each handler much shorter and more consistent.
