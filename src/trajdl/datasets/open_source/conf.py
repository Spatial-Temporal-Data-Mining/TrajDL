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
import tempfile
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ...common.enum import OpenSourceDatasetEnum
from ...config import CACHE_DATASET_DIR
from .hasher import Hasher
from .utils import decompress_gz, download_file, remove_path, unzip_file


@dataclass
class OpenSourceDataset(ABC):
    """Base class for open-source datasets.

    Provides functionality for downloading, validating, and loading datasets.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset.
    size : int
        The expected size of the dataset.
    url : Optional[str]
        The URL to download the dataset from.
    sha256_original : str
        The expected SHA-256 hash of the original dataset.
    mmh3_cache : str
        The expected MMH3 hash of the cached dataset.

    Methods
    -------
    check_original_dataset(path: str) -> bool
        Validates the downloaded dataset.
    download(chunk_size) -> str
        Downloads the dataset if it does not exist or is invalid.
    load_cache(table: pa.Table, return_as: str = "pl") -> Union[pl.DataFrame, pd.DataFrame, pa.Table]
        Loads the cached dataset in the specified format.
    load(return_as: str = "pl") -> Union[pl.DataFrame, pd.DataFrame, pa.Table]
        Loads the dataset, attempting to use the cache first.
    cache()
        Abstract method to be implemented by subclasses for caching behavior.
    set_path(path: str) -> None
        Set the original dataset path.
    """

    dataset_name: str
    size: int
    url: Optional[str]
    sha256_original: str
    mmh3_cache: str

    def __post_init__(self):
        self._path = os.path.join(CACHE_DATASET_DIR, self.dataset_name)
        self._cache_path = os.path.join(
            CACHE_DATASET_DIR, f"cache_{self.dataset_name}.parquet"
        )

    @property
    def path(self) -> str:
        """Return the path to the downloaded dataset file."""
        return self._path

    @property
    def cache_path(self) -> str:
        """Return the path to the cached Parquet file."""
        return self._cache_path

    def set_path(self, path: str) -> None:
        """Set the path of the original dataset

        If user specifies the path of the original dataset, this path will be saved into this dataset object.

        Parameters
        ----------
        path: str
            The path of the original dataset, like `~/Downloads/loc-gowalla_totalCheckins.txt.gz`
        """
        self._path = path

    def set_url(self, url: str) -> None:
        """Set the download url of the original dataset

        If user specifies a url for a dataset, the dataset will be downloaded from that link.

        Parameters
        ----------
        url: str
            The download link of the original dataset.
        """
        self.url = url

    def check_original_dataset(self, path: str) -> bool:
        """Check if the original dataset is valid.

        Parameters
        ----------
        path : str
            The path to the dataset to validate.

        Returns
        -------
        bool
            True if the dataset is valid, False otherwise.
        """
        hasher = Hasher(hasher_type="sha256")
        if os.path.exists(path):
            if os.path.getsize(path) != self.size:
                warnings.warn("[Validating datasets] Size of the dataset is invalid.")
                return False
            elif hasher.digest_file(path) != self.sha256_original:
                warnings.warn("[Validating datasets] SHA-256 check failed.")
                return False
            else:
                return True
        else:
            return False

    def download(self) -> str:
        """Download the dataset.

        The dataset is downloaded to the CACHE_DATASET_DIR. The path is determined by the dataset name.

        Parameters
        ----------
        chunk_size : int, optional
            The size of each download chunk (default is 8192).

        Returns
        -------
        str
            The path to the downloaded dataset.
        """
        if not self.check_original_dataset(self.path):
            print(f"Dataset will be downloaded from {self.url}")
            download_file(self.url, self.path)

            if self.check_original_dataset(self.path):
                print(f"Dataset has been downloaded as {self.path}")
                return self.path
            else:
                print("SHA-256 check failed.")

    def load_cache(
        self, table: pa.Table, return_as: str = "pl"
    ) -> Union[pl.DataFrame, pd.DataFrame, pa.Table]:
        """Load the cache in the specified format.

        Parameters
        ----------
        table : pa.Table
            The Arrow table to load.
        return_as : str, optional
            The format to return the table in ('pl' for Polars, 'pd' for Pandas, or 'pa' for PyArrow).

        Returns
        -------
        Union[pl.DataFrame, pd.DataFrame, pa.Table]
            The loaded table in the specified format.

        Raises
        ------
        ValueError
            If the specified return format is unsupported.
        """
        if return_as == "pl":
            return pl.from_arrow(table)
        elif return_as == "pd":
            return table.to_pandas()
        elif return_as == "pa":
            return table
        else:
            raise ValueError("return_as only supports {'pl', 'pd', 'pa'}")

    def load(
        self,
        return_as: str = "pl",
        original_dataset_path: Optional[str] = None,
        unsafe: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame, pa.Table]:
        """Load the dataset, checking the cache first.

        If no valid cached data exists, it attempts to load the original dataset from `original_dataset_path`
        or download the original dataset, cache it, and then load the data.

        Parameters
        ----------
        return_as : str, optional
            The format to return (default is 'pl').
        original_dataset_path: Optional[str], optional
            The path of the original dataset downloaded by user (default is None).
        unsafe: bool, optional
            Do not check mmh3 of cache dataset (default is False).

        Returns
        -------
        Union[pl.DataFrame, pd.DataFrame, pa.Table]
            The loaded dataset in the specified format.
        """
        print(f"load dataset: {self.dataset_name}")
        hasher = Hasher(hasher_type="mmh3")
        if os.path.isfile(self.cache_path):
            table = pq.read_table(self.cache_path)
            if unsafe or hasher.digest_arrow(table) == self.mmh3_cache:
                return self.load_cache(table, return_as=return_as)

        # Remove cache if not valid
        remove_path(self.cache_path)

        # if original_dataset_path is specified, validate this dataset, otherwise download the dataset
        need_download = True
        if original_dataset_path:
            # validate dataset, set the new path
            if self.check_original_dataset(original_dataset_path):
                self.set_path(original_dataset_path)
                print(
                    f"dataset loaded from the original_dataset_path: {original_dataset_path}",
                    flush=True,
                )
                need_download = False

        if need_download:
            self.download()

        self.cache()
        table = pq.read_table(self.cache_path)
        return self.load_cache(table, return_as=return_as)

    @abstractmethod
    def cache(self) -> None:
        """Cache the dataset.

        Implement this method to define how the dataset is cached,
        typically by extracting the downloaded file and saving it as a Parquet file.
        """
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass
class GowallaDataset(OpenSourceDataset):
    """Gowalla Dataset class extending OpenSourceDataset with specific parameters."""

    dataset_name: str = "gowalla"
    url: str = os.environ.get(
        OpenSourceDatasetEnum.GOWALLA_URL.value,
        "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
    )
    size: int = 105470044
    sha256_original: str = (
        "c1c3e19effba649b6c89aeab3c1f9459fad88cfdc2b460fc70fd54e295d83ea0"
    )
    mmh3_cache: str = "8a2eb882146b2ab51774b4bf8b1432dc"

    def cache(self) -> None:
        """Cache the Gowalla dataset by decompressing and storing it as Parquet."""
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            tmp_file_path = tmp_file.name
            print("Decompressing files...")
            decompress_gz(self.path, tmp_file_path)
            pl.read_csv(
                tmp_file_path,
                has_header=False,
                new_columns=["user_id", "check_in_time", "lat", "lng", "loc_id"],
                separator="\t",
                try_parse_dates=True,
                schema_overrides={
                    "user_id": pl.String,
                    "check_in_time": pl.Datetime,
                    "lng": pl.Float64,
                    "lat": pl.Float64,
                    "loc_id": pl.String,
                },
            ).write_parquet(self.cache_path, use_pyarrow=True)


@dataclass
class PortoDataset(OpenSourceDataset):
    """Porto Dataset class extending OpenSourceDataset with specific parameters."""

    dataset_name: str = "porto"
    url: str = os.environ.get(
        OpenSourceDatasetEnum.PORTO_URL.value,
        "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip",
    )
    size: int = 534065916
    sha256_original: str = (
        "a33e2a5e145607ae2bad0db5d21b7548c88b7e0f9db1ce15839f24c4c61f8c76"
    )
    mmh3_cache: str = "5ef83abb3cf649583f28b80f16e1f4a7"

    def cache(self) -> None:
        """Cache the Porto dataset by decompressing and storing it as Parquet."""
        with tempfile.TemporaryDirectory() as tmp_folder:
            print("Decompressing files...")
            unzip_file(self.path, tmp_folder)
            train_csv_zip_path = os.path.join(tmp_folder, "train.csv.zip")
            unzip_file(train_csv_zip_path, tmp_folder)
            csv_path = os.path.join(tmp_folder, "train.csv")

            with open(csv_path, "r") as f:
                line_count = sum(1 for _ in f)  # 计算行数
            print(f"num records: {line_count}")

            progress_bar = tqdm(total=line_count)

            reader = pl.read_csv_batched(
                csv_path,
                schema_overrides={
                    "TRIP_ID": pl.String,
                    "CALL_TYPE": pl.String,
                    "ORIGIN_CALL": pl.Int64,
                    "ORIGIN_STAND": pl.Int64,
                    "TAXI_ID": pl.Int64,
                    "TIMESTAMP": pl.Int64,
                    "DAY_TYPE": pl.String,
                    "MISSING_DATA": pl.Boolean,
                    "POLYLINE": pl.String,
                },
                batch_size=10000,
            )

            new_df = []
            batches = reader.next_batches(5)
            while batches:
                new_df.append(
                    pl.concat(batches).with_columns(
                        pl.col("POLYLINE")
                        .str.json_decode(dtype=pl.List(pl.List(pl.Float64)))
                        .cast(pl.List(pl.Array(pl.Float64, 2)))
                        .alias("POLYLINE")
                    )
                )
                progress_bar.update(new_df[-1].shape[0])
                batches = reader.next_batches(5)

            progress_bar.close()

            pl.concat(new_df).write_parquet(self.cache_path, use_pyarrow=True)
