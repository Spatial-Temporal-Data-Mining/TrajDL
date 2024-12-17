# Copyright 2024 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import requests

from trajdl.datasets.open_source.downloader import FileDownloader


# Mocking responses for testing
class MockResponse:
    def __init__(self, content_length, content):
        self.headers = {"content-length": str(content_length)}
        self.content = content

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


# Mocking requests methods
def mock_requests_head(url):
    if url == "http://invalid.url/file":
        raise requests.exceptions.RequestException("Invalid URL")
    elif url.endswith("range_support"):
        return MockResponse(1000, b"")
    else:
        return MockResponse(1000, b"")


def mock_requests_get(url, headers=None, stream=False):
    if url == "http://invalid.url/file":
        raise requests.exceptions.RequestException("Invalid URL")
    if headers and "Range" in headers:
        return MockResponse(1000, b"x" * 1000)  # 每个请求返回1000字节的内容
    return MockResponse(1000, b"x" * 1000)


# Replacing requests methods with mock methods
requests.head = mock_requests_head
requests.get = mock_requests_get


# Test class for the FileDownloader
class TestFileDownloader:
    @pytest.fixture
    def setup_downloader(self):
        return FileDownloader(num_threads=4)

    def test_download_range_supported(self, setup_downloader):
        url = "http://example.com/file/range_support"
        output_path = "/tmp/trajdl/test_output_range_supported"
        downloader = setup_downloader

        downloader.download(url, output_path)

        # Check if the file was created and its content is valid
        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            content = f.read()
            assert content == b"x" * 1000  # Verify content is correct

        os.remove(output_path)

    def test_download_range_not_supported(self, setup_downloader):
        url = "http://example.com/file/no_range_support"
        output_path = "/tmp/trajdl/test_output_range_not_supported"
        downloader = setup_downloader

        downloader.download(url, output_path)

        # Check if the file was created and its content is valid
        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            content = f.read()
            assert content == b"x" * 1000  # Verify content is correct

        os.remove(output_path)

    def test_download_invalid_url(self, setup_downloader):
        url = "http://invalid.url/file"
        output_path = "/tmp/trajdl/test_output_invalid_url"
        downloader = setup_downloader

        with pytest.raises(requests.exceptions.RequestException):
            downloader.download(url, output_path)

    def test_output_path_exists(self, setup_downloader):
        url = "http://example.com/file/range_support"
        output_path = "/tmp/trajdl/test_output_path_exists"
        downloader = setup_downloader

        # Pre-create the output file
        with open(output_path, "wb") as f:
            f.write(b"original_content")

        downloader.download(url, output_path)

        # Verify the file was overwritten and content is correct
        with open(output_path, "rb") as f:
            content = f.read()
            assert content == b"x" * 1000  # Verify content is correct

        os.remove(output_path)
