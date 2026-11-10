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
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import orjson
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ...config import CACHE_DATASET_DIR
from .foursquare import read_tist2015, read_tsmc2014_txt
from .hasher import Hasher
from .hub import (
    HUGGINGFACE,
    HubFile,
    build_download_error_message,
    download_hub_file,
    resolve_source,
)
from .utils import decompress_gz, remove_path, unzip_file


@dataclass
class OpenSourceDataset(ABC):
    """Base class for open-source datasets.

    Provides functionality for downloading from dataset hubs, validating, and
    loading datasets. The download source is HuggingFace or ModelScope-CN
    (hub-only distribution, no fallback to original URLs).

    Attributes
    ----------
    dataset_name : str
        The name of the dataset.
    size : int
        The expected size of the dataset.
    sha256_original : str
        The expected SHA-256 hash of the original dataset.
    TABLES : Tuple[str, ...]
        Ordered names of the tables this dataset provides; the first entry is
        the primary table (returned by ``load()`` when ``table=None``).
        Multi-table datasets (e.g. UniTE) declare every table here.
    mmh3_tables : Dict[str, str]
        Expected MMH3 stable hash of every cached table, keyed by table name.
        Pins live in code (never in the sidecar) so that a tampered cache
        cannot re-validate itself by rewriting its metadata.
    hf : HubFile
        File position on HuggingFace (repo_id / filename / revision).
    ms : HubFile
        File position on ModelScope-CN (repo_id / filename / revision).

    Methods
    -------
    check_original_dataset(path: str) -> bool
        Validates the downloaded dataset.
    download(source: Optional[str] = None) -> str
        Downloads the dataset from the hub if it does not exist or is invalid.
    load_cache(table: pa.Table, return_as: str = "pl") -> Union[pl.DataFrame, pd.DataFrame, pa.Table]
        Loads the cached dataset in the specified format.
    load(return_as: str = "pl", table: Optional[str] = None, original_dataset_path: Optional[str] = None, unsafe: bool = False, source: Optional[str] = None) -> Union[pl.DataFrame, pd.DataFrame, pa.Table]
        Loads one table of the dataset, attempting to use the cache first.
    cache()
        Abstract method to be implemented by subclasses for caching behavior.
    set_path(path: str) -> None
        Set the original dataset path.
    """

    TABLES: ClassVar[tuple[str, ...]] = ()

    dataset_name: str
    size: int
    sha256_original: str
    mmh3_tables: ClassVar[dict[str, str]] = {}
    hf: HubFile
    ms: HubFile

    def __post_init__(self):
        if not self.TABLES:
            raise ValueError(
                f"{type(self).__name__} must declare TABLES (ordered table names, primary first)"
            )
        if set(self.mmh3_tables) != set(self.TABLES):
            raise ValueError(
                f"{type(self).__name__}: mmh3_tables keys {sorted(self.mmh3_tables)} "
                f"must match TABLES {list(self.TABLES)}"
            )
        self._path = os.path.join(CACHE_DATASET_DIR, self.dataset_name)
        self._cache_dir = os.path.join(
            CACHE_DATASET_DIR, f"cache_{self.dataset_name}"
        )

    @property
    def path(self) -> str:
        """Return the path to the downloaded dataset file."""
        return self._path

    @property
    def primary_table(self) -> str:
        """Return the name of the primary table (the default ``load()`` target)."""
        return self.TABLES[0]

    @property
    def cache_dir(self) -> str:
        """Return the directory holding this dataset's cached tables."""
        return self._cache_dir

    def table_path(self, name: str) -> str:
        """Return the path of one cached table's Parquet file."""
        if name not in self.TABLES:
            raise ValueError(
                f"unknown table {name!r} for {self.dataset_name}; available: {list(self.TABLES)}"
            )
        return os.path.join(self._cache_dir, f"{name}.parquet")

    @property
    def meta_path(self) -> str:
        """Return the path to the sidecar metadata file."""
        return os.path.join(self._cache_dir, "meta.json")

    def _write_sidecar(self, tables: dict[str, pa.Table]) -> None:
        """Write informational sidecar metadata alongside the cached tables.

        The sidecar is for humans and debugging only (row/column counts per
        table). It deliberately contains no hashes: integrity comes from the
        code-pinned ``mmh3_tables``, which a tampered cache cannot rewrite.
        """
        meta = {
            "dataset_name": self.dataset_name,
            "format_version": 1,
            "tables": {
                name: {"num_rows": t.num_rows, "num_columns": t.num_columns}
                for name, t in tables.items()
            },
        }
        with open(self.meta_path, "wb") as f:
            f.write(orjson.dumps(meta, option=orjson.OPT_SORT_KEYS))

    def _read_sidecar(self) -> dict | None:
        """Read the sidecar metadata file if it exists.

        Returns
        -------
        Optional[dict]
            The parsed metadata dict, or None if the file does not exist
            or is malformed.
        """
        if not os.path.isfile(self.meta_path):
            return None
        try:
            with open(self.meta_path, "rb") as f:
                return orjson.loads(f.read())
        except (orjson.JSONDecodeError, OSError):
            return None

    def _validate_table(self, name: str, table: pa.Table) -> bool:
        """Validate one cached table against its code-pinned stable hash."""
        hasher = Hasher(hasher_type="mmh3")
        return hasher.digest_arrow_stable(table) == self.mmh3_tables[name]

    def _validate_cache(self) -> bool:
        """Validate every table of the cache against the code-pinned hashes.

        The sidecar is never consulted for validation: it is informational
        only, so tampering with the cache plus rewriting the sidecar cannot
        produce a valid state (the pins live in code).
        """
        for name in self.TABLES:
            tpath = self.table_path(name)
            if not os.path.isfile(tpath):
                return False
            table = pq.read_table(tpath)
            if not self._validate_table(name, table):
                return False
        return True

    def set_path(self, path: str) -> None:
        """Set the path of the original dataset

        If user specifies the path of the original dataset, this path will be saved into this dataset object.

        Parameters
        ----------
        path: str
            The path of the original dataset, like `~/Downloads/loc-gowalla_totalCheckins.txt.gz`
        """
        self._path = path

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

    def download(self, source: str | None = None) -> str:
        """Download the dataset from the configured hub.

        The dataset is downloaded to ``CACHE_DATASET_DIR`` and validated by
        size and SHA-256. A failed download or validation raises an error with
        recovery hints; no fallback to another source is attempted.

        Parameters
        ----------
        source : str, optional
            ``"huggingface"`` or ``"modelscope"``. Defaults to the
            ``TRAJDL_DATASET_SOURCE`` env var, then ``"huggingface"``.

        Returns
        -------
        str
            The path to the downloaded dataset.

        Raises
        ------
        ValueError
            If ``source`` is not a valid source name.
        RuntimeError
            If the download or the SHA-256 validation fails.
        """
        if self.check_original_dataset(self.path):
            return self.path

        resolved = resolve_source(source)
        hub_file = self.hf if resolved == HUGGINGFACE else self.ms

        try:
            download_hub_file(hub_file, resolved, self.path)
        except ImportError:
            remove_path(self.path)
            raise
        except Exception as e:
            remove_path(self.path)
            raise RuntimeError(
                build_download_error_message(
                    resolved, hub_file, self.sha256_original, error=e
                )
            ) from e

        if not self.check_original_dataset(self.path):
            hasher = Hasher(hasher_type="sha256")
            actual = (
                hasher.digest_file(self.path)
                if os.path.exists(self.path)
                else None
            )
            remove_path(self.path)
            raise RuntimeError(
                build_download_error_message(
                    resolved, hub_file, self.sha256_original, actual_sha256=actual
                )
            )

        print(f"Dataset has been downloaded as {self.path}")
        return self.path

    def load_cache(
        self, table: pa.Table, return_as: str = "pl"
    ) -> pl.DataFrame | pd.DataFrame | pa.Table:
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
        table: str | None = None,
        original_dataset_path: str | None = None,
        unsafe: bool = False,
        source: str | None = None,
    ) -> pl.DataFrame | pd.DataFrame | pa.Table:
        """Load one table of the dataset, checking the cache first.

        If no valid cached data exists, it attempts to load the original dataset from `original_dataset_path`
        or download the original dataset from the hub, cache it, and then load the data.

        Parameters
        ----------
        return_as : str, optional
            The format to return (default is 'pl').
        table : Optional[str], optional
            The name of the table to load (default is None, meaning the
            primary table). Multi-table datasets (e.g. UniTE) expose their
            auxiliary tables through this parameter.
        original_dataset_path: Optional[str], optional
            The path of the original dataset downloaded by user (default is None).
        unsafe: bool, optional
            Do not check mmh3 of cache dataset (default is False).
        source: Optional[str], optional
            ``"huggingface"`` or ``"modelscope"``; only effective when a
            download is needed. Defaults to the ``TRAJDL_DATASET_SOURCE`` env
            var, then ``"huggingface"``.

        Returns
        -------
        Union[pl.DataFrame, pd.DataFrame, pa.Table]
            The loaded table in the specified format.
        """
        name = self.primary_table if table is None else table
        tpath = self.table_path(name)
        print(f"load dataset: {self.dataset_name} ({name})")
        if os.path.isfile(tpath):
            table = pq.read_table(tpath)
            if unsafe or self._validate_table(name, table):
                return self.load_cache(table, return_as=return_as)

        # Remove the cache directory (and any legacy flat-layout leftovers)
        # if not valid; caches are written as a whole, so any invalid table
        # rebuilds everything.
        remove_path(self._cache_dir)
        remove_path(os.path.join(CACHE_DATASET_DIR, f"cache_{self.dataset_name}.parquet"))
        remove_path(os.path.join(CACHE_DATASET_DIR, f"cache_{self.dataset_name}.meta.json"))

        # if original_dataset_path is specified, validate this dataset, otherwise download the dataset
        need_download = True
        # validate dataset, set the new path; fall back to hub download if invalid
        if original_dataset_path and self.check_original_dataset(original_dataset_path):
            self.set_path(original_dataset_path)
            print(
                f"dataset loaded from the original_dataset_path: {original_dataset_path}",
                flush=True,
            )
            need_download = False

        if need_download:
            self.download(source=source)

        self.cache()
        table = pq.read_table(tpath)
        return self.load_cache(table, return_as=return_as)

    def _write_cache(self, tables: dict[str, pa.Table]) -> None:
        """Write all tables of this dataset to the cache directory.

        Called by subclass ``cache()`` implementations after they have
        converted the raw download. Writes one Parquet file per table plus
        the informational sidecar.
        """
        unknown = set(tables) - set(self.TABLES)
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.cache() produced unknown tables {sorted(unknown)}; "
                f"expected {list(self.TABLES)}"
            )
        missing = set(self.TABLES) - set(tables)
        if missing:
            raise ValueError(
                f"{type(self).__name__}.cache() is missing tables {sorted(missing)}; "
                f"expected {list(self.TABLES)}"
            )
        os.makedirs(self._cache_dir, exist_ok=True)
        for name, arrow_table in tables.items():
            pq.write_table(arrow_table, self.table_path(name))
        self._write_sidecar(tables)
        print(
            f"cached tables -> {self._cache_dir} "
            f"({', '.join(f'{n}: {t.num_rows} rows' for n, t in tables.items())})"
        )

    @abstractmethod
    def cache(self) -> None:
        """Cache the dataset.

        Implement this method to define how the dataset is cached,
        typically by extracting the downloaded file, converting every table
        faithfully, and passing the ``{table_name: pa.Table}`` mapping to
        ``_write_cache``.
        """
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass
class GowallaDataset(OpenSourceDataset):
    """Gowalla Dataset class extending OpenSourceDataset with specific parameters."""

    TABLES: ClassVar[tuple[str, ...]] = ("checkins",)
    mmh3_tables: ClassVar[dict[str, str]] = {
        "checkins": "aaecd5d7cec82fdbdaa1e5b268b8aaf1",
    }
    dataset_name: str = "gowalla"
    size: int = 105470044
    sha256_original: str = (
        "c1c3e19effba649b6c89aeab3c1f9459fad88cfdc2b460fc70fd54e295d83ea0"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/loc-gowalla_totalCheckins",
        filename="loc-gowalla_totalCheckins.txt.gz",
        revision="b150c4e69677f408b732c2412b7ce96f866e49b8",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/loc-gowalla_totalCheckins",
        filename="loc-gowalla_totalCheckins.txt.gz",
        revision="e706922d99d9b47760058a1b3939eb29f609b598",
    )

    def cache(self) -> None:
        """Cache the Gowalla dataset by decompressing and storing it as Parquet."""
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            tmp_file_path = tmp_file.name
            print("Decompressing files...")
            decompress_gz(self.path, tmp_file_path)
            df = pl.read_csv(
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
            )
            self._write_cache({"checkins": df.to_arrow()})


@dataclass
class PortoDataset(OpenSourceDataset):
    """Porto Dataset class extending OpenSourceDataset with specific parameters."""

    TABLES: ClassVar[tuple[str, ...]] = ("trips",)
    mmh3_tables: ClassVar[dict[str, str]] = {
        "trips": "e54c8b154baed6b467aef776bb1191ce",
    }
    dataset_name: str = "porto"
    size: int = 534065916
    sha256_original: str = (
        "a33e2a5e145607ae2bad0db5d21b7548c88b7e0f9db1ce15839f24c4c61f8c76"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/Porto_Taxi_Trajectory",
        filename="taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip",
        revision="ae7c9db6e6ead1c458b724f5c88018e03060599f",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/Porto_Taxi_Trajectory",
        filename="taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip",
        revision="e56a3d627bfba4c6d41206a2f33ea1f56a7f0c4c",
    )

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
            )

            new_df = []
            batches = reader.next_batches(10)
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
                batches = reader.next_batches(10)
            self._write_cache({"trips": pl.concat(new_df).to_arrow()})

            progress_bar.close()


@dataclass
class _TSMC2014Dataset(OpenSourceDataset):
    """Raw base for the Foursquare TSMC2014 city datasets (NYC / TKY share one zip).

    The cache is a faithful conversion of the public city dump (dtype
    normalization only). Dataset-specific preprocessing — e.g. the TALE paper
    filter rules — lives in each algorithm's benchmark layer.
    """

    TABLES: ClassVar[tuple[str, ...]] = ("checkins",)
    city: str = ""
    dataset_name: str = ""
    mmh3_tables: ClassVar[dict[str, str]] = {}
    size: int = 25546284
    sha256_original: str = (
        "cbe3fdab373d24b09b5fc53509c8958c77ff72b6c1a68589ce337d4f9a80235b"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/FourSquare_NYC_TKY",
        filename="dataset_tsmc2014.zip",
        revision="96cd1044291c207b5056b6e01012df6a96b7b5fd",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/FourSquare_NYC_TKY",
        filename="dataset_tsmc2014.zip",
        revision="7dedc995f0ab3abfe3271e86f6f84e8f8d8a7698",
    )

    def cache(self) -> None:
        """Cache the raw check-in table of one city as Parquet (no filtering)."""
        with tempfile.TemporaryDirectory() as tmp_folder:
            unzip_file(self.path, tmp_folder)
            txt_path = os.path.join(
                tmp_folder, "dataset_tsmc2014", f"dataset_TSMC2014_{self.city}.txt"
            )
            df = read_tsmc2014_txt(txt_path)
            self._write_cache({"checkins": pl.from_pandas(df).to_arrow()})


@dataclass
class FoursquareNYCDataset(_TSMC2014Dataset):
    """Raw Foursquare NYC check-ins (faithful TSMC2014 conversion)."""

    city: str = "NYC"
    dataset_name: str = "foursquare_nyc"
    mmh3_tables: ClassVar[dict[str, str]] = {
        "checkins": "c63f7d5cb4771df7652714c2ef7a85a6",
    }


@dataclass
class FoursquareTKYDataset(_TSMC2014Dataset):
    """Raw Foursquare TKY check-ins (faithful TSMC2014 conversion)."""

    city: str = "TKY"
    dataset_name: str = "foursquare_tky"
    mmh3_tables: ClassVar[dict[str, str]] = {
        "checkins": "c65fd8c6e850c606fc609341a9f3ef4d",
    }


@dataclass
class _TIST2015Dataset(OpenSourceDataset):
    """Raw base for the Foursquare TIST2015 city datasets (jkt / ist share one zip).

    The cache is a faithful conversion of the public dumps: every check-in row
    (bad timestamps -> NaT) enriched with the published POI attributes. Both
    cities share one cache because the underlying dump is global; the
    city-circle selection is algorithm-level logic and lives in the benchmark
    layer.
    """

    TABLES: ClassVar[tuple[str, ...]] = ("checkins",)
    city: str = ""
    dataset_name: str = ""
    mmh3_tables: ClassVar[dict[str, str]] = {
        "checkins": "b3a1bf17f424df728662b3e81791840f",
    }
    size: int = 775746915
    sha256_original: str = (
        "a2b2a0495aa9db369bc0cd596dddd412f36640b08bae5b3166402133cab4767b"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/FourSquare_Global",
        filename="dataset_TIST2015.zip",
        revision="c83e08d467b0482a955d3addeb4814c80ccb4cca",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/FourSquare_Global",
        filename="dataset_TIST2015.zip",
        revision="526c1eb8d665a6dd576f4936bd3d8ca8b176e149",
    )

    def __post_init__(self):
        super().__post_init__()
        # the two cities consume the same global dump -> share one raw cache
        shared_dir = os.path.join(CACHE_DATASET_DIR, "cache_foursquare_tist2015")
        self._cache_dir = shared_dir

    def cache(self) -> None:
        """Cache the raw POI-enriched global check-in table as Parquet."""
        with tempfile.TemporaryDirectory() as tmp_folder:
            unzip_file(self.path, tmp_folder)
            df = read_tist2015(
                os.path.join(tmp_folder, "dataset_TIST2015_POIs.txt"),
                os.path.join(tmp_folder, "dataset_TIST2015_Checkins.txt"),
            )
            self._write_cache({"checkins": pl.from_pandas(df).to_arrow()})


@dataclass
class FoursquareJKTDataset(_TIST2015Dataset):
    """Foursquare Jakarta check-ins (raw global dump, shared TIST2015 cache)."""

    dataset_name: str = "foursquare_jkt"
    city: str = "jkt"


@dataclass
class FoursquareISTDataset(_TIST2015Dataset):
    """Foursquare Istanbul check-ins (raw global dump, shared TIST2015 cache)."""

    dataset_name: str = "foursquare_ist"
    city: str = "ist"


def _normalize_unite_road_info(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the mixed types of road_info.level: str and list[str] (e.g. ['trunk','motorway'])
    are stored together and Arrow cannot encode that; lists are joined into str with commas, str
    values are kept as is. All other columns are left untouched."""
    df = df.copy()
    df["level"] = df["level"].map(
        lambda v: ",".join(map(str, v)) if isinstance(v, list) else v
    )
    return df


@dataclass
class _UniteHDataset(OpenSourceDataset):
    """Base class for the multi-table UniTE h5 datasets.

    Cache-layer principle: **as raw as possible** — every h5 table is
    faithfully dumped into parquet (column set, row order, and dtype unchanged, ``time`` keeps
    datetime64), with no splitting/filtering/resampling/feature engineering; sample construction
    always happens in the benchmark layer
    (``scripts/benchmark/unite/scripts/samples.py``).

    The only normalization: ``road_info.level`` mixes str and list[str] in the raw dump, which
    Arrow cannot encode, so lists are comma-joined (see :func:`_normalize_unite_road_info`).

    TABLES is declared per city: chengdu includes network_info, the xian release h5 has no such
    table. Auxiliary tables are read via ``load(table="trip_info")`` etc. (a directory-style
    cache, one parquet per table).
    """

    def cache(self) -> None:
        tables: dict[str, pa.Table] = {}
        with pd.HDFStore(self.path, mode="r") as store:
            for key in store:
                name = key.lstrip("/")
                df = store[key]
                if name == "road_info":
                    df = _normalize_unite_road_info(df)
                tables[name] = pa.Table.from_pandas(df, preserve_index=False)
        self._write_cache(tables)


@dataclass
class UniteChengduDataset(_UniteHDataset):
    """UniTE Chengdu road-network matched trajectories (processed data from the Didi collaboration, not raw GPS; 2018-09-29 ~ 10-10).

    Data facts: 140,000 trips / 18,832,411 points / 4,315 roads / 8,276+ drivers; it cannot be
    strictly reconciled with the 121,394 / 3.03M in the paper's Table I. The
    splitting/filtering basis follows the UniTE data.py code.
    """

    TABLES: ClassVar[tuple[str, ...]] = (
        "trips",
        "trip_info",
        "road_info",
        "network_info",
        "pois",
    )
    mmh3_tables: ClassVar[dict[str, str]] = {
        "trips": "7f93d0947ccfc0eafbf347ca02d9a9d5",
        "trip_info": "ab8c7c265a2c07ad2d72241ea7bf4be8",
        "road_info": "cb68eaefab477c3847ea6b9f3f5558f8",
        "network_info": "690ce3699f54ebaade214c6f16d907e4",
        "pois": "e271e344343913ec674318286c3e72d1",
    }
    dataset_name: str = "unite_chengdu"
    size: int = 1221409120
    sha256_original: str = (
        "591b27e3a480a5b9346c7bfe0a3240d9ac944bdf1909ce520bb69e15fcfa219e"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/UniTE-chengdu",
        filename="chengdu.h5",
        revision="5d3f46934e88b4d3a67659a0b3dad131d18be910",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/UniTE-chengdu",
        filename="chengdu.h5",
        revision="e7c0a4329ade9d7d3b495d66d950d5749899595b",
    )


@dataclass
class UniteXianDataset(_UniteHDataset):
    """UniTE Xian road-network matched trajectories (release version: 110,000 trips / 9,514,287 points).

    Note (D2): this file is the release export, a different source from the internal version used in
    the paper experiments (210K trips / 18.27M points); experiment numbers based on this file are not
    comparable with the paper's xian numbers and also disagree with Table I (202,849 / 18,267,443),
    which must be declared in the dataset card.
    """

    TABLES: ClassVar[tuple[str, ...]] = ("trips", "trip_info", "road_info", "pois")
    mmh3_tables: ClassVar[dict[str, str]] = {
        "trips": "ff0d07968f5e3f43347328ba41775ba9",
        "trip_info": "2c00b54336cc23b9198767c91bacc9f4",
        "road_info": "47b715efd353f45ffe99a9edb803f0f8",
        "pois": "cc7351501582679ee4500a8e7cc048db",
    }
    dataset_name: str = "unite_xian"
    size: int = 618023088
    sha256_original: str = (
        "9ec29c1adde8659a9610042afb5ea88b4d086ca32dfbd0c785794ac0da9a876a"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/UniTE-xian",
        filename="xian.h5",
        revision="9a3576f306c67c6ba3377f3bff9cc650c53005b7",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/UniTE-xian",
        filename="xian.h5",
        revision="1198cac4634bbfde4d59e4dfe2c0e0b47ab07849",
    )


class _UniteEmbeddingDataset(OpenSourceDataset):
    """Embedding dataset base class shared by the two algorithms TransferTraj / TrajMamba (two npy per city).

    Data facts: POI/road text-structure embedding matrices (n, 64) float64, produced by the
    TransferTraj / TrajMamba authors' TrajCLIP pre-training pipeline; the npy files of the two
    releases are bit-for-bit identical (verified by md5), so this dataset is shared by both
    algorithms. Row numbers align one-to-one with the ``pois`` / ``road_info`` table ids of the UniTE
    city h5: chengdu poi 12439 / road 4315, xian poi 3900 / road 3392.

    Unlike the single-raw-file assumption of the ``OpenSourceDataset`` base class, the raw artifact
    of this dataset is **two npy files** (the release names carry a ``small_`` prefix, but the content
    covers the full-city POI/road sets and has nothing to do with "small"), therefore:

    - ``self.path`` is a directory (with symlinks to the npy files), not a single file
    - ``check_original_dataset`` / ``download`` validate file by file per table (size +
      sha256, pinned in ``raw_sizes`` / ``raw_sha256``)
    - ``original_dataset_path`` points to the directory containing the two npy files (e.g. the
      ``samples/`` directory of the TransferTraj release)

    The only cache-layer normalization: npy → parquet (the ``id`` column = row number, the
    ``embed_0..63`` columns = the original float64 values), with no other processing.
    """

    # table name -> hub 上原始 npy 文件名（release 命名）
    raw_files: ClassVar[dict[str, str]] = {}
    raw_sizes: ClassVar[dict[str, int]] = {}
    raw_sha256: ClassVar[dict[str, str]] = {}

    @staticmethod
    def _npy_to_embed_table(arr: np.ndarray) -> pa.Table:
        if arr.ndim != 2 or arr.dtype != np.float64:
            raise ValueError(f"expected (n, 64) float64 embedding, got {arr.shape} {arr.dtype}")
        n, d = arr.shape
        return pa.table(
            {
                "id": pa.array(range(n), type=pa.int64()),
                **{f"embed_{j}": pa.array(arr[:, j], type=pa.float64()) for j in range(d)},
            }
        )

    def _raw_file_path(self, table: str) -> str:
        return os.path.join(self.path, self.raw_files[table])

    def _check_raw_file(self, table: str, path: str) -> bool:
        hasher = Hasher(hasher_type="sha256")
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) != self.raw_sizes[table]:
            warnings.warn(f"[Validating datasets] Size of {self.raw_files[table]} is invalid.")
            return False
        if hasher.digest_file(path) != self.raw_sha256[table]:
            warnings.warn(
                f"[Validating datasets] SHA-256 check failed on {self.raw_files[table]}."
            )
            return False
        return True

    def check_original_dataset(self, path: str) -> bool:
        return all(
            self._check_raw_file(name, os.path.join(path, self.raw_files[name]))
            for name in self.TABLES
        )

    def download(self, source: str | None = None) -> str:
        if self.check_original_dataset(self.path):
            return self.path

        resolved = resolve_source(source)
        hub_file = self.hf if resolved == HUGGINGFACE else self.ms
        os.makedirs(self.path, exist_ok=True)
        for name in self.TABLES:
            target = self._raw_file_path(name)
            file_hub = replace(hub_file, filename=self.raw_files[name])
            try:
                download_hub_file(file_hub, resolved, target)
            except Exception as e:
                remove_path(self.path)
                raise RuntimeError(
                    build_download_error_message(
                        resolved, file_hub, self.raw_sha256[name], error=e
                    )
                ) from e
            if not self._check_raw_file(name, target):
                remove_path(self.path)
                raise RuntimeError(
                    build_download_error_message(resolved, file_hub, self.raw_sha256[name])
                )
        print(f"Dataset has been downloaded as {self.path}")
        return self.path

    def cache(self) -> None:
        tables = {
            name: self._npy_to_embed_table(np.load(self._raw_file_path(name)))
            for name in self.TABLES
        }
        self._write_cache(tables)


@dataclass
class UniteChengduEmbeddingDataset(_UniteEmbeddingDataset):
    """UniTE Chengdu POI/road embedding (shared by TransferTraj / TrajMamba).

    Tables: ``road_embed`` (4315×64, primary table) / ``poi_embed`` (12439×64);
    row numbers align with the ``road_info`` / ``pois`` table ids of ``UniteChengduDataset``.
    The ``small_`` file-name prefix is the release package's naming convention, not a statement
    about data scale.
    """

    TABLES: ClassVar[tuple[str, ...]] = ("road_embed", "poi_embed")
    mmh3_tables: ClassVar[dict[str, str]] = {
        "road_embed": "aa5785801bc40a6fde3a3045879ec807",
        "poi_embed": "40d30ceada3ef6428be38d9ce76b141b",
    }
    raw_files: ClassVar[dict[str, str]] = {
        "road_embed": "small_chengdu_road_embed.npy",
        "poi_embed": "small_chengdu_poi_embed.npy",
    }
    raw_sizes: ClassVar[dict[str, int]] = {
        "road_embed": 2209408,
        "poi_embed": 6368896,
    }
    raw_sha256: ClassVar[dict[str, str]] = {
        "road_embed": "f6d95c3d9910c22836c20d429372110fe765a71fd9c26b9f512446437e5b970b",
        "poi_embed": "229647f5d52010894519210d742c6fec4d11347f1e6e666ad867d630c1213979",
    }
    dataset_name: str = "unite_chengdu_embedding"
    # 基类字段在多文件数据集下仅作展示用途：size = 两 npy 字节数之和
    size: int = 8578304
    sha256_original: str = (
        "229647f5d52010894519210d742c6fec4d11347f1e6e666ad867d630c1213979"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/UniTE_chengdu_embedding",
        filename="small_chengdu_road_embed.npy",
        revision="0c5ca07f4d63f43e9264e75f217d1e2f00797d4e",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/UniTE_chengdu_embedding",
        filename="small_chengdu_road_embed.npy",
        revision="4203d4a9afb3e2ca15ccd9540b2e018d4a559e23",
    )


@dataclass
class UniteXianEmbeddingDataset(_UniteEmbeddingDataset):
    """UniTE Xian POI/road embedding (shared by TransferTraj / TrajMamba).

    Tables: ``road_embed`` (3392×64, primary table) / ``poi_embed`` (3900×64);
    row numbers align with the ``road_info`` / ``pois`` table ids of ``UniteXianDataset``.
    """

    TABLES: ClassVar[tuple[str, ...]] = ("road_embed", "poi_embed")
    mmh3_tables: ClassVar[dict[str, str]] = {
        "road_embed": "9bd37ee88c1bf2ba4ac407419c5a7eb1",
        "poi_embed": "7b97e504ecc8b2cce4c301d64a266fcd",
    }
    raw_files: ClassVar[dict[str, str]] = {
        "road_embed": "small_xian_road_embed.npy",
        "poi_embed": "small_xian_poi_embed.npy",
    }
    raw_sizes: ClassVar[dict[str, int]] = {
        "road_embed": 1736832,
        "poi_embed": 1996928,
    }
    raw_sha256: ClassVar[dict[str, str]] = {
        "road_embed": "538eb8e47cc26abc3700c0ef9f642e0d07b6776ccc8183d813405721aee3fa45",
        "poi_embed": "96f549c83a74f038c9355acd1e1ad7b12171e252c2a20ebdec6fa7f086a2f400",
    }
    dataset_name: str = "unite_xian_embedding"
    size: int = 3733760
    sha256_original: str = (
        "96f549c83a74f038c9355acd1e1ad7b12171e252c2a20ebdec6fa7f086a2f400"
    )
    hf: HubFile = HubFile(
        repo_id="Davidham3/UniTE_xian_embedding",
        filename="small_xian_road_embed.npy",
        revision="ccb22c09b93ba711faa8814fede62abc0924eef5",
    )
    ms: HubFile = HubFile(
        repo_id="Davidham3/UniTE_xian_embedding",
        filename="small_xian_road_embed.npy",
        revision="fa76cae2b094a18203d8843344a0385f12af0bfc",
    )
