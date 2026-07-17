"""Headless tests for src/gui_workers.py (IoWorker/DownloadWorker)."""

import os
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pytest.importorskip("PyQt6.QtCore")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def run_worker(worker, timeout_ms=5000):
    """Start a worker and spin an event loop until it signals."""
    from PyQt6.QtCore import QEventLoop, QTimer
    outcome = {}
    loop = QEventLoop()
    worker.finished_ok.connect(lambda r: (outcome.setdefault('ok', r),
                                          loop.quit()))
    worker.failed.connect(lambda e: (outcome.setdefault('err', e),
                                     loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)  # safety net
    worker.start()
    loop.exec()
    worker.wait(1000)
    return outcome


class TestIoWorker:
    def test_result_crosses_thread(self, qapp):
        from gui_workers import IoWorker
        worker = IoWorker(lambda x, y: x + y, 20, 22)
        outcome = run_worker(worker)
        assert outcome == {'ok': 42}

    def test_exception_becomes_failed_signal(self, qapp):
        from gui_workers import IoWorker

        def boom():
            raise ValueError("kaput")

        outcome = run_worker(IoWorker(boom))
        assert outcome == {'err': 'kaput'}


class TestDownloadWorker:
    def test_progress_and_result(self, qapp, monkeypatch, tmp_path):
        import net_utils

        def fake_download(url, filepath, *, progress_callback=None,
                          cancel_event=None, **kw):
            for done in (50, 100):
                progress_callback(done, 100)
            return Path(filepath)

        monkeypatch.setattr(net_utils, "download_file", fake_download)
        from gui_workers import DownloadWorker
        worker = DownloadWorker("https://example.test/f",
                                tmp_path / "out.bsp")
        seen = []
        worker.progress.connect(lambda p, m: seen.append(p))
        outcome = run_worker(worker)
        assert outcome['ok'] == tmp_path / "out.bsp"
        assert seen and seen[-1] == 100

    def test_cancel_sets_event(self, qapp, monkeypatch, tmp_path):
        import net_utils

        started = threading.Event()

        def fake_download(url, filepath, *, progress_callback=None,
                          cancel_event=None, **kw):
            started.set()
            cancel_event.wait(5)
            raise net_utils.DownloadCancelled("cancelled")

        monkeypatch.setattr(net_utils, "download_file", fake_download)
        from gui_workers import DownloadWorker
        worker = DownloadWorker("https://example.test/f",
                                tmp_path / "out.bsp")
        worker.started.connect(lambda: None)
        threading.Timer(0.1, worker.cancel).start()
        outcome = run_worker(worker)
        assert 'err' in outcome and 'cancel' in outcome['err'].lower()
