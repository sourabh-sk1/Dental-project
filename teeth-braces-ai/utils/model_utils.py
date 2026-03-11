"""Model utilities: download helper for model weights.

This module provides a small, dependency-free helper to download model
weights from a URL to a destination path. It uses the standard library so
tests can mock the network call safely.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from typing import Optional


def download_weights(url: str, dest_path: str, chunk_size: int = 8192) -> Optional[str]:
    """Download a file from `url` to `dest_path`.

    Uses urllib.request.urlretrieve under the hood but writes to a temporary
    file first and moves it into place atomically.

    Args:
        url: URL to download from
        dest_path: local filesystem path to write the downloaded file
        chunk_size: not used currently but kept for future streaming logic

    Returns:
        The path to the downloaded file on success, or None on failure.
    """
    try:
        # Ensure destination dir exists
        dest_dir = os.path.dirname(os.path.abspath(dest_path))
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        # Download to a temporary path first
        tmp_path = dest_path + ".part"

        # Use urllib.request.urlretrieve which is easy to mock in tests
        urllib.request.urlretrieve(url, tmp_path)

        # Move into place
        shutil.move(tmp_path, dest_path)
        return dest_path
    except Exception:
        # Do not raise — callers handle fallback behavior. Return None to
        # indicate failure.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None
