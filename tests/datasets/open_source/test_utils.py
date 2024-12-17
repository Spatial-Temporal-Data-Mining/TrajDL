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

import gzip
import os
import sys
import zipfile
from io import StringIO
from unittest.mock import patch

import pytest

from trajdl.datasets.open_source.utils import (
    decompress_gz,
    download_file,
    remove_path,
    unzip_file,
)


def test_decompress_gz(tmp_path):
    gz_file = tmp_path / "test.gz"
    decompressed_file = tmp_path / "decompressed.txt"

    # Create a sample .gz file
    with gzip.open(gz_file, "wb") as f:
        f.write(b"Hello, World!")

    # Test decompression
    decompress_gz(str(gz_file), str(decompressed_file))

    assert decompressed_file.exists()
    with open(decompressed_file, "rb") as f:
        content = f.read()

    assert content == b"Hello, World!"


def test_unzip_file(tmp_path):
    zip_file = tmp_path / "test.zip"
    output_folder = tmp_path / "output"

    # Create a sample zip file
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.writestr("test.txt", "Hello, World!")

    # Test unzipping
    unzip_file(str(zip_file), str(output_folder))

    assert (output_folder / "test.txt").exists()
    with open(output_folder / "test.txt", "r") as f:
        content = f.read()

    assert content == "Hello, World!"


def test_download_file(requests_mock, tmp_path):
    url = "http://test.com"
    content = b"data"
    requests_mock.get(url, content=content)

    file_path = tmp_path / "test_file"

    download_file(url, str(file_path))

    assert file_path.exists()
    with open(file_path, "rb") as f:
        content = f.read()
    assert content == content


def test_download_file_timeout(requests_mock, tmp_path):
    url = "http://test.com"
    requests_mock.get(url, status_code=404)

    file_path = tmp_path / "test_file"

    with pytest.raises(
        RuntimeError, match="Downloading dataset failed! Check your network."
    ):
        download_file(url, str(file_path))


def test_remove_path_file(tmp_path):
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("Hello, World!")

    assert file_path.exists()

    remove_path(str(file_path))
    assert not file_path.exists()


def test_remove_path_directory(tmp_path):
    dir_path = tmp_path / "test_dir"
    os.makedirs(dir_path)

    assert dir_path.exists()

    remove_path(str(dir_path))
    assert not dir_path.exists()


def test_remove_path_non_existent(tmp_path):
    non_existent_path = tmp_path / "does_not_exist.txt"

    remove_path(str(non_existent_path))


def test_remove_path_exception_handling():
    # Redirect stdout to capture print statements
    captured_output = StringIO()
    sys.stdout = captured_output

    # Patch os.path.isfile and os.path.isdir to simulate raising an exception
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.remove", side_effect=Exception("File not found")),
        patch("os.path.isdir", return_value=False),
    ):

        remove_path("dummy_file.txt")

    # Reset redirect.
    sys.stdout = sys.__stdout__

    # Check the printed output
    assert "Error while deleting" in captured_output.getvalue()


# Additional test for directories
def test_remove_directory_exception_handling():
    captured_output = StringIO()
    sys.stdout = captured_output

    with (
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch("shutil.rmtree", side_effect=Exception("Directory not empty")),
    ):

        remove_path("dummy_directory")

    sys.stdout = sys.__stdout__

    assert "Error while deleting" in captured_output.getvalue()
