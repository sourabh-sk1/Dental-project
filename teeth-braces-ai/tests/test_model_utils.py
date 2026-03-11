import os
import tempfile
import urllib.request
import pytest

from utils.model_utils import download_weights


class DummyURLRetriever:
    """Helper to monkeypatch urllib.request.urlretrieve"""

    @staticmethod
    def fake_urlretrieve(url, filename):
        # Write a small dummy file to simulate download
        with open(filename, 'wb') as f:
            f.write(b"dummy-weights-content")
        return (filename, None)


def test_download_weights_creates_file(monkeypatch):
    monkeypatch.setattr(urllib.request, 'urlretrieve', DummyURLRetriever.fake_urlretrieve)

    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, 'weights.pt')
        result = download_weights('http://example.com/weights.pt', dest)
        assert result is not None
        assert os.path.exists(dest)
        with open(dest, 'rb') as f:
            data = f.read()
        assert data == b"dummy-weights-content"

*** End Patch