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
import shutil
import zipfile


def decompress_gz(gz_path: str, output_path: str) -> None:
    """Decompress a .gz file.

    Parameters
    ----------
    gz_path : str
        The path to the .gz file.
    output_path : str
        The path where the decompressed file will be saved.
    """
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def unzip_file(zip_file_path: str, output_folder: str) -> None:
    """Unzip a zip file to a specified folder.

    Parameters
    ----------
    zip_file_path : str
        The path to the zip file.
    output_folder : str
        The directory where the zip file will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)


def remove_path(path):
    """Remove the specified file or directory, making it non-existent.

    Parameters
    ----------
    path : str
        The path to the file or directory to be deleted.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If an error occurs while deleting the path.

    Notes
    -----
    - If the specified path does not exist, a message indicating that will be printed.
    - If the path is a directory, it will be removed recursively.
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"File '{path}' has been removed.")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Directory '{path}' has been removed.")
        else:
            print(f"Path '{path}' does not exist.")
    except Exception as e:
        print(f"Error while deleting '{path}': {e}")
