"""
Worker threads for blocking I/O in the GUI — design in
docs/GUI_WORKER_THREADS_DESIGN.md.

Rules (see the design doc):
- Workers never touch Qt widgets; results cross back only via signals.
- Workers only move bytes / compute; Qt-side objects are constructed on
  the GUI thread after the worker finishes.
"""

import logging
import threading

try:
    from PyQt6.QtCore import Qt, QThread, QEventLoop, pyqtSignal
    from PyQt6.QtWidgets import QProgressDialog
    _WINDOW_MODAL = Qt.WindowModality.WindowModal
except ImportError:
    from PyQt5.QtCore import Qt, QThread, QEventLoop, pyqtSignal
    from PyQt5.QtWidgets import QProgressDialog
    _WINDOW_MODAL = Qt.WindowModal

logger = logging.getLogger(__name__)


class IoWorker(QThread):
    """Run a blocking callable off the GUI thread.

    Signals:
        progress(int, str) — percent (-1 = indeterminate), message
        finished_ok(object) — the callable's return value
        failed(str) — exception text
    """

    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self.finished_ok.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:
            logger.warning(f"Worker failed: {e}")
            self.failed.emit(str(e))


class DownloadWorker(IoWorker):
    """IoWorker specialized for net_utils.download_file with progress
    reporting and cancellation."""

    def __init__(self, url, filepath, *, min_size=None, parent=None):
        from net_utils import download_file
        self._cancel_event = threading.Event()

        def report(done, total):
            if total:
                self.progress.emit(int(done * 100 / total),
                                   f"{done / 1e6:.0f} / {total / 1e6:.0f} MB")
            else:
                self.progress.emit(-1, f"{done / 1e6:.0f} MB")

        super().__init__(download_file, url, filepath,
                         min_size=min_size, show_progress=False,
                         progress_callback=report,
                         cancel_event=self._cancel_event,
                         parent=parent)

    def cancel(self):
        self._cancel_event.set()


def download_with_dialog(parent, url, filepath, *, title,
                         min_size=None):
    """
    Download url to filepath in a worker thread while showing a
    window-modal progress dialog with a Cancel button. Blocks (spinning
    a local event loop, so the GUI stays responsive) until the download
    finishes, fails, or is cancelled.

    Returns the Path on success, or None on cancel/failure (failure is
    logged; the caller decides what to tell the user).
    """
    dialog = QProgressDialog(title, "Cancel", 0, 100, parent)
    dialog.setWindowModality(_WINDOW_MODAL)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

    worker = DownloadWorker(url, filepath, min_size=min_size, parent=parent)
    outcome = {}
    loop = QEventLoop()

    def on_progress(pct, msg):
        if pct >= 0:
            dialog.setValue(pct)
        dialog.setLabelText(f"{title}\n{msg}")

    worker.progress.connect(on_progress)
    worker.finished_ok.connect(lambda r: (outcome.setdefault('ok', r),
                                          loop.quit()))
    worker.failed.connect(lambda e: (outcome.setdefault('err', e),
                                     loop.quit()))
    dialog.canceled.connect(worker.cancel)

    worker.start()
    dialog.show()
    loop.exec() if hasattr(loop, 'exec') else loop.exec_()

    worker.wait(5000)
    dialog.close()

    if 'ok' in outcome:
        return outcome['ok']
    if 'err' in outcome:
        logger.error(f"Download failed: {outcome['err']}")
    return None
