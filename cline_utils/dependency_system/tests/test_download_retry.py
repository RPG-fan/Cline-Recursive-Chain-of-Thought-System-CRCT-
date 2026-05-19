import os
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from cline_utils.dependency_system.analysis.embedding_manager import _download_with_retry


class DummyHeader:
    def __init__(self, headers):
        self.headers = headers

    def get(self, name, default=None):
        return self.headers.get(name, default)


class DummyResponse:
    def __init__(self, data, status=200, headers=None):
        self.data = data
        self.status = status
        self._headers = headers or {"Content-Length": str(len(data))}
        self.read_called = 0

    def info(self):
        return DummyHeader(self._headers)

    def read(self, chunk_size=8192):
        if self.read_called > 0:
            return b""
        self.read_called += 1
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_download_success():
    """Test a simple successful download from start to finish."""
    url = "https://example.com/test.model"
    data = b"Hello, model world!"
    
    response = DummyResponse(data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = os.path.join(tmpdir, "test.model")
        
        with patch("cline_utils.dependency_system.analysis.embedding_manager.urlopen", return_value=response) as mock_urlopen, \
             patch("cline_utils.dependency_system.analysis.embedding_manager.Request") as mock_req_class:
            
            mock_req = MagicMock()
            mock_req_class.return_value = mock_req
            
            success = _download_with_retry(url, dest_path, "Test Model", max_retries=3, initial_delay=0.01)
            
            assert success is True
            assert os.path.exists(dest_path)
            with open(dest_path, "rb") as f:
                assert f.read() == data
                
            mock_urlopen.assert_called_once_with(mock_req)
            mock_req.add_header.assert_any_call("User-Agent", "LLMRPG-Downloader/1.0")


def test_download_permanent_error():
    """Test that a permanent HTTP error (e.g. 404) aborts immediately without retrying."""
    url = "https://example.com/missing.model"
    
    # HTTPError signature: url, code, msg, hdrs, fp
    http_error = urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = os.path.join(tmpdir, "missing.model")
        
        with patch("cline_utils.dependency_system.analysis.embedding_manager.urlopen", side_effect=http_error) as mock_urlopen:
            success = _download_with_retry(url, dest_path, "Missing Model", max_retries=5, initial_delay=0.01)
            
            assert success is False
            assert not os.path.exists(dest_path)
            # Should only be called once, no retries
            assert mock_urlopen.call_count == 1


def test_download_transient_retry_success():
    """Test that transient HTTP errors are retried and eventually succeed."""
    url = "https://example.com/transient.model"
    data = b"Successful after retry"
    
    transient_error = urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
    response = DummyResponse(data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = os.path.join(tmpdir, "transient.model")
        
        # First call raises 503, second call succeeds
        with patch("cline_utils.dependency_system.analysis.embedding_manager.urlopen", side_effect=[transient_error, response]) as mock_urlopen, \
             patch("time.sleep") as mock_sleep:
            
            success = _download_with_retry(url, dest_path, "Transient Model", max_retries=3, initial_delay=0.01)
            
            assert success is True
            assert os.path.exists(dest_path)
            with open(dest_path, "rb") as f:
                assert f.read() == data
                
            assert mock_urlopen.call_count == 2
            assert mock_sleep.call_count == 1


def test_download_resume_success():
    """Test that download resumes using Range header after a mid-download network failure."""
    url = "https://example.com/large.model"
    part1_data = b"First half "
    part2_data = b"Second half"
    total_data = part1_data + part2_data
    
    # First attempt: successfully returns response, but reading raises connection reset midway
    class FailingResponse(DummyResponse):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.read_count = 0

        def read(self, chunk_size=8192):
            if self.read_count == 0:
                self.read_count += 1
                return self.data
            raise ConnectionResetError("Connection dropped mid-stream")

    response1 = FailingResponse(part1_data, status=200, headers={"Content-Length": str(len(total_data))})
    
    # Second attempt: returns the remaining data with status 206 Partial Content
    response2 = DummyResponse(part2_data, status=206, headers={"Content-Length": str(len(part2_data))})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = os.path.join(tmpdir, "large.model")
        
        # Mocks urlopen to return FailingResponse on first call, DummyResponse on second
        with patch("cline_utils.dependency_system.analysis.embedding_manager.urlopen", side_effect=[response1, response2]) as mock_urlopen, \
             patch("time.sleep") as mock_sleep, \
             patch("cline_utils.dependency_system.analysis.embedding_manager.Request") as mock_req_class:
            
            req1 = MagicMock()
            req2 = MagicMock()
            mock_req_class.side_effect = [req1, req2]
            
            success = _download_with_retry(url, dest_path, "Large Model", max_retries=3, initial_delay=0.01)
            
            assert success is True
            assert os.path.exists(dest_path)
            with open(dest_path, "rb") as f:
                assert f.read() == total_data
                
            assert mock_urlopen.call_count == 2
            assert mock_sleep.call_count == 1
            
            # Verify the second request asked for the Range header starting at len(part1_data)
            req2.add_header.assert_any_call("Range", f"bytes={len(part1_data)}-")


def test_download_resume_ignored_starts_over():
    """Test that if the server ignores Range request (returns 200), we start download from scratch."""
    url = "https://example.com/non_resumable.model"
    part1_data = b"Stale part data"
    full_data = b"Fresh complete download data"
    
    class FailingResponse(DummyResponse):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.read_count = 0

        def read(self, chunk_size=8192):
            if self.read_count == 0:
                self.read_count += 1
                return self.data
            raise ConnectionResetError("Connection dropped mid-stream")

    response1 = FailingResponse(part1_data, status=200, headers={"Content-Length": "1000"})
    
    # Second attempt: server returns 200 OK (ignores Range header)
    response2 = DummyResponse(full_data, status=200, headers={"Content-Length": str(len(full_data))})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = os.path.join(tmpdir, "non_resumable.model")
        
        with patch("cline_utils.dependency_system.analysis.embedding_manager.urlopen", side_effect=[response1, response2]) as mock_urlopen, \
             patch("time.sleep") as mock_sleep, \
             patch("cline_utils.dependency_system.analysis.embedding_manager.Request") as mock_req_class:
            
            req1 = MagicMock()
            req2 = MagicMock()
            mock_req_class.side_effect = [req1, req2]
            
            success = _download_with_retry(url, dest_path, "Non-resumable Model", max_retries=3, initial_delay=0.01)
            
            assert success is True
            assert os.path.exists(dest_path)
            with open(dest_path, "rb") as f:
                # Content should only be full_data, the stale part1_data should have been overwritten!
                assert f.read() == full_data
                
            assert mock_urlopen.call_count == 2
            assert mock_sleep.call_count == 1
            
            # Verify range header was requested on second call
            req2.add_header.assert_any_call("Range", f"bytes={len(part1_data)}-")
