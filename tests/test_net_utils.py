"""Tests for the shared HTTP helper (src/net_utils.py) — retry/backoff,
SSL-fallback ladder, and download integrity guarantees. All network
activity is faked; no real connections are made."""

import pytest
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import net_utils
from net_utils import http_get, download_file


class FakeResponse:
    def __init__(self, content=b"data", status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers if headers is not None else {
            'content-length': str(len(content))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code}", response=self)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retries must not actually wait during tests."""
    monkeypatch.setattr(net_utils.time, "sleep", lambda s: None)


def patch_get(monkeypatch, fake):
    calls = []

    def wrapper(url, **kwargs):
        calls.append(kwargs)
        return fake(url, **kwargs)

    monkeypatch.setattr(net_utils.requests, "get", wrapper)
    return calls


class TestHttpGet:
    def test_success_first_attempt(self, monkeypatch):
        calls = patch_get(monkeypatch, lambda url, **kw: FakeResponse(b"ok"))
        r = http_get("https://example.test/x")
        assert r.content == b"ok"
        assert len(calls) == 1

    def test_retries_transient_then_succeeds(self, monkeypatch):
        state = {"n": 0}

        def flaky(url, **kw):
            state["n"] += 1
            if state["n"] < 3:
                raise requests.exceptions.ConnectionError("transient")
            return FakeResponse(b"ok")

        calls = patch_get(monkeypatch, flaky)
        r = http_get("https://example.test/x", retries=3)
        assert r.content == b"ok"
        assert len(calls) == 3

    def test_gives_up_after_retries(self, monkeypatch):
        def always_fail(url, **kw):
            raise requests.exceptions.ConnectionError("down")

        calls = patch_get(monkeypatch, always_fail)
        with pytest.raises(requests.exceptions.ConnectionError):
            http_get("https://example.test/x", retries=3)
        assert len(calls) == 3

    def test_no_retry_on_client_error(self, monkeypatch):
        calls = patch_get(monkeypatch,
                          lambda url, **kw: FakeResponse(b"", status_code=404))
        with pytest.raises(requests.exceptions.HTTPError):
            http_get("https://example.test/missing", retries=3)
        assert len(calls) == 1  # 4xx must not be retried

    def test_retries_server_error(self, monkeypatch):
        state = {"n": 0}

        def flaky(url, **kw):
            state["n"] += 1
            if state["n"] == 1:
                return FakeResponse(b"", status_code=503)
            return FakeResponse(b"ok")

        calls = patch_get(monkeypatch, flaky)
        r = http_get("https://example.test/x", retries=3)
        assert r.content == b"ok"
        assert len(calls) == 2

    def test_ssl_fallback_ladder(self, monkeypatch, caplog):
        """SSLError on verified rungs → succeeds unverified, loudly."""

        def ssl_broken(url, **kw):
            if kw.get("verify") is False:
                return FakeResponse(b"ok")
            raise requests.exceptions.SSLError("bad CA store")

        calls = patch_get(monkeypatch, ssl_broken)
        with caplog.at_level("WARNING", logger="net_utils"):
            r = http_get("https://example.test/x")
        assert r.content == b"ok"
        # normal, certifi, then verify=False
        assert calls[-1].get("verify") is False
        assert any("TLS VERIFICATION DISABLED" in m for m in caplog.messages)


class TestDownloadFile:
    def test_writes_file_atomically(self, monkeypatch, tmp_path):
        patch_get(monkeypatch,
                  lambda url, **kw: FakeResponse(b"x" * 1000))
        dest = tmp_path / "out.dat"
        result = download_file("https://example.test/f", dest,
                               show_progress=False)
        assert result == dest
        assert dest.read_bytes() == b"x" * 1000
        assert not dest.with_name("out.dat.part").exists()

    def test_rejects_undersized_download(self, monkeypatch, tmp_path):
        patch_get(monkeypatch, lambda url, **kw: FakeResponse(b"404 page"))
        dest = tmp_path / "out.dat"
        with pytest.raises(IOError, match="suspiciously small"):
            download_file("https://example.test/f", dest,
                          min_size=1000, show_progress=False)
        assert not dest.exists()
        assert not dest.with_name("out.dat.part").exists()

    def test_failed_download_leaves_no_partial(self, monkeypatch, tmp_path):
        class ExplodingResponse(FakeResponse):
            def iter_content(self, chunk_size=1):
                yield b"partial"
                raise requests.exceptions.ConnectionError("dropped")

        patch_get(monkeypatch, lambda url, **kw: ExplodingResponse())
        dest = tmp_path / "out.dat"
        with pytest.raises(requests.exceptions.ConnectionError):
            download_file("https://example.test/f", dest,
                          show_progress=False)
        assert not dest.exists()
        assert not dest.with_name("out.dat.part").exists()
