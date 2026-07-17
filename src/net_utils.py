"""
Shared HTTP helpers — one timeout, retry, and SSL-fallback policy for
every network operation in NEOlyzer.

Previously each call site had its own policy (timeouts of none/60/120/300 s,
no retries anywhere) and two modules carried verbatim copies of the
SSL-fallback ladder. All downloads and API calls should go through
http_get() / download_file().

The SSL ladder (normal verification → certifi bundle → unverified) exists
for platforms with incomplete CA certificate stores (notably Raspberry Pi).
The unverified rung is deliberately loud: it logs a prominent warning every
time, and download_file() applies a minimum-size sanity check so an
interposed error page can't silently replace a data file.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import requests

logger = logging.getLogger(__name__)

# (connect, read) seconds. The read timeout applies between bytes of the
# response, so it also bounds a stalled streaming download.
DEFAULT_TIMEOUT: Tuple[float, float] = (10, 120)
DEFAULT_RETRIES = 3
RETRY_BACKOFF_S = 2.0  # first retry delay; doubles per attempt


def _get_with_ssl_fallback(url: str, *, params=None, stream=False,
                           timeout=DEFAULT_TIMEOUT) -> requests.Response:
    """Single GET attempt, walking the SSL-fallback ladder on SSL errors."""
    try:
        response = requests.get(url, params=params, stream=stream,
                                timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError as e:
        logger.warning(f"SSL verification failed for {url}: {e}")

    # Second rung: certifi CA bundle
    try:
        import certifi
        logger.info("Retrying with certifi certificate bundle...")
        response = requests.get(url, params=params, stream=stream,
                                timeout=timeout, verify=certifi.where())
        response.raise_for_status()
        return response
    except ImportError:
        logger.info("certifi not installed")
    except requests.exceptions.SSLError:
        logger.warning("SSL verification still failing with certifi")

    # Last rung: no verification. Loud on purpose — the transfer is
    # unauthenticated and could be tampered with in transit.
    logger.warning(f"*** TLS VERIFICATION DISABLED for {url} — transfer is "
                   f"unauthenticated. Fix the system CA store to restore "
                   f"verified downloads. ***")
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    response = requests.get(url, params=params, stream=stream,
                            timeout=timeout, verify=False)
    response.raise_for_status()
    return response


def http_get(url: str, *, params=None, stream=False,
             timeout=DEFAULT_TIMEOUT,
             retries: int = DEFAULT_RETRIES) -> requests.Response:
    """
    GET with retry/backoff and SSL fallback. Raises the last
    requests exception if all attempts fail.

    Client errors (HTTP 4xx) are not retried — they won't heal.
    Timeouts, connection errors, and server errors (5xx) are.
    """
    for attempt in range(1, retries + 1):
        try:
            return _get_with_ssl_fallback(url, params=params, stream=stream,
                                          timeout=timeout)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500:
                raise  # 4xx: retrying won't help
            if attempt == retries:
                raise
        except requests.exceptions.RequestException:
            if attempt == retries:
                raise
        wait = RETRY_BACKOFF_S * 2 ** (attempt - 1)
        logger.warning(f"GET {url} failed (attempt {attempt}/{retries}); "
                       f"retrying in {wait:.0f}s")
        time.sleep(wait)


def download_file(url: str, filepath: Union[str, Path], *,
                  desc: Optional[str] = None,
                  timeout=DEFAULT_TIMEOUT,
                  retries: int = DEFAULT_RETRIES,
                  min_size: Optional[int] = None,
                  show_progress: bool = True) -> Path:
    """
    Stream url to filepath with a tqdm progress bar.

    Downloads to filepath.part and renames on success, so an interrupted
    download never leaves a truncated file that a later exists() check
    would mistake for a good one. If min_size is given and the result is
    smaller, the download is rejected (guards against error pages served
    with HTTP 200, especially on the unverified-TLS ladder rung).
    """
    from tqdm import tqdm

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    partial = filepath.with_name(filepath.name + '.part')

    response = http_get(url, stream=True, timeout=timeout, retries=retries)
    total_size = int(response.headers.get('content-length', 0))

    try:
        with open(partial, 'wb') as f, \
             tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=desc or filepath.name,
                  disable=not show_progress) as pbar:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                pbar.update(len(chunk))

        size = partial.stat().st_size
        if min_size is not None and size < min_size:
            raise IOError(
                f"Download of {url} is suspiciously small "
                f"({size:,} bytes < required {min_size:,}); rejecting")
        partial.replace(filepath)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    logger.info(f"Downloaded {filepath.name} "
                f"({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
    return filepath
