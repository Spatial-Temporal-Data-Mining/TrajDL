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

import hashlib
from typing import Iterable

import mmh3
import pyarrow as pa
import pyarrow.parquet as pq


class Hasher:
    def __init__(self, hasher_type: str):
        """Initialize the Hasher with a specified hashing algorithm.

        Parameters
        ----------
        hasher_type : str
            The type of hasher to use. Options are "sha256" and "mmh3".

        Raises
        ------
        ValueError
            If the specified hasher_type is not supported.
        """
        self.hasher_type = hasher_type
        self.init_hasher()

    def init_hasher(self):
        """Initialize the hashing function based on the hasher_type."""
        if self.hasher_type == "sha256":
            self.hasher = hashlib.sha256()
        elif self.hasher_type == "mmh3":
            self.hasher = mmh3.mmh3_x64_128()
        else:
            raise ValueError(
                f"hasher_type should be one of {'sha256', 'mmh3'}, not '{self.hasher_type}'"
            )

    def _get_hash(self, iterable: Iterable[bytes]) -> str:
        """Compute the hash for a given iterable of byte data.

        Parameters
        ----------
        iterable : Iterable[bytes]
            An iterable yielding byte objects to hash.

        Returns
        -------
        str
            The hexadecimal representation of the hash value.
        """
        for b in iterable:
            self.hasher.update(b)
        result = self.hasher.digest().hex()
        # Re-initialize the hasher for future use
        self.init_hasher()
        return result

    def digest_file(self, path: str) -> str:
        """Digest a file and produce its hash.

        Parameters
        ----------
        path : str
            The path to the file to be hashed.

        Returns
        -------
        str
            The hexadecimal representation of the file hash.
        """
        with open(path, "rb") as f:
            return self._get_hash(iter(lambda: f.read(1 << 20), b""))

    def digest_arrow(self, table: pa.Table, max_chunksize: int = 8192) -> str:
        """Digest a PyArrow Table and produce its hash.

        Parameters
        ----------
        table : pa.Table
            The PyArrow table to be hashed.
        max_chunksize : int, optional
            The maximum size of each chunk for processing (default is 8192).

        Returns
        -------
        str
            The hexadecimal representation of the table hash.
        """
        return self._get_hash(
            batch.serialize().hex()
            for batch in table.to_batches(max_chunksize=max_chunksize)
        )

    def digest_parquet(self, path: str, max_chunksize: int = 8192) -> str:
        """Digest a Parquet file and produce its hash.

        Parameters
        ----------
        path : str
            The path to the Parquet file to be hashed.
        max_chunksize : int, optional
            The maximum size of each chunk for processing (default is 8192).

        Returns
        -------
        str
            The hexadecimal representation of the Parquet file hash.
        """
        table = pq.read_table(path)
        return self.digest_arrow(table=table, max_chunksize=max_chunksize)
