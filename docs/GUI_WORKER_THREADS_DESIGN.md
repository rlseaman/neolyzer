# GUI Worker Threads for Blocking I/O — Design (plan item 5.4)

Status: **implemented** 2026-07-16 (steps 1–3 below; see
`src/gui_workers.py`). One correction found during implementation:
there is **no in-GUI ephemeris selector** — `set_configured_ephemeris`
is called only by the CLI setup wizard, so "user switches ephemeris in
settings" below is not a live trigger. The real GUI scenario is a
missing `.bsp` at startup (deleted cache, hand-edited
`~/.neolyzer/ephemeris.json`, or interrupted setup), now handled in
`NEOVisualizer.initialize_data` via `download_with_dialog()`. The
synchronous `ensure_ephemeris()` calls in the render paths remain as a
safety net and are no-ops once the file exists. If an in-GUI ephemeris
selector is ever added, it should reuse the same pieces.

## Inventory — what actually blocks the GUI thread (verified by grep)

The plan assumed the JPL SBDB MOID fetch was the likeliest freeze;
**that turned out to be wrong** — `fetch_moid_batch` is called only
from scripts (`setup_database.py`, `update_catalog.py`,
`load_alt_catalog.py`), never from the running GUI.

What the GUI thread actually does synchronously:

| operation | trigger | worst case |
|---|---|---|
| **Ephemeris download** (`ensure_ephemeris`/`load_ephemeris`, GUI call sites at `neolyzer.py:1617, 1986, 9129` plus module init) | first run without the file; user switches ephemeris in settings | **de441.bsp is 3.5 GB** — the window freezes for the entire download (minutes to an hour on slow links). de440 is 115 MB. |
| Initial catalog load + cache open (`initialize_data`, `neolyzer.py:16313` area) | startup | seconds; splash-adjacent, lower priority |
| Full-catalog position computation when cache misses (`--no-cache` or out-of-range dates) | animation | per-frame cost, a performance topic (plan 6.6), not a worker topic |

Conclusion: **the ephemeris download is the one real freeze**, and it
is guaranteed to hit any user who selects de441 (or a fresh install
without bundled kernels).

## Design

One small generic worker, not a per-task zoo:

```python
class IoWorker(QThread):
    progress = pyqtSignal(int, str)   # percent (-1 = indeterminate), message
    finished_ok = pyqtSignal(object)  # result
    failed = pyqtSignal(str)          # error text

    def __init__(self, fn, *args, **kwargs): ...
    def run(self):
        try:
            self.finished_ok.emit(self.fn(*self.args, **self.kwargs))
        except Exception as e:
            self.failed.emit(str(e))
```

- `net_utils.download_file()` gains an optional
  `progress_callback(bytes_done, bytes_total)` parameter (tqdm stays
  for CLI use; the callback feeds `IoWorker.progress` for the GUI).
- The ephemeris-switch flow becomes: disable the affected controls →
  start worker → modal-less progress dialog with Cancel → on
  `finished_ok`, reload ephemeris object and refresh; on `failed`,
  restore previous ephemeris selection and show the error.
- Cancellation: a `threading.Event` checked in the download chunk
  loop (needs a `cancel_event` parameter on `download_file`); the
  atomic `.part` rename already guarantees no truncated file on
  cancel.

## Rules (to prevent the classic Qt threading bugs)

1. Workers never touch Qt widgets; results cross back only via
   signals (queued connections handle thread affinity).
2. One worker at a time per resource (the ephemeris flow disables its
   trigger while running).
3. Skyfield objects are constructed on the GUI thread *after* the
   file lands — the worker only moves bytes.

## Implementation order

1. `download_file(progress_callback=, cancel_event=)` in net_utils
   (+ tests with a fake response).
2. `IoWorker` + progress dialog; wire the ephemeris-switch call site.
3. First-run ephemeris download at startup (same worker, splash text).
4. Later, if catalog update ever moves in-app, it reuses the same
   pieces.
