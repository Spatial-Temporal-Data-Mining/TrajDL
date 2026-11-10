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
import uuid
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
import orjson
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib import tenumerate

from ... import CoordinateSystem, LocSeq, LocSeqView, Trajectory, TrajectoryView
from ... import __version__ as TRAJDL_VERSION
from ...common.enum import ArrowColName, SeqType
from .builder import make_record_batch_builder
from .schema import TRAJ_ARROW_SCHEMA

CORE_TABLE_FILENAME = "core_table.parquet"
FEATURE_TABLE_FILENAME = "feature_table.parquet"
META_FILENAME = "meta.json"


def df_to_arrow(table: pa.Table | pd.DataFrame | pl.DataFrame) -> pa.Table:
    if isinstance(table, pl.DataFrame):
        return table.to_arrow()
    elif isinstance(table, pd.DataFrame):
        return pa.Table.from_pandas(table)
    elif isinstance(table, pa.Table):
        return table
    else:
        raise ValueError(
            "`table` should be an instance of {'pyarrow.Table', 'polars.DataFrame', 'pandas.DataFrame'}"
        )


def check_columns_exist(table: pa.Table, required_columns: Iterable[str]) -> None:
    """Verify that all required columns exist in an Arrow table.

    Shared predicate for the ``check_table`` implementations of the Arrow
    dataset subclasses.

    Parameters
    ----------
    table : pa.Table
        The table to inspect.
    required_columns : Iterable[str]
        Column names that must be present in the table.

    Raises
    ------
    ValueError
        If any required column is missing; the message names the offending
        column(s).
    """
    missing = [name for name in required_columns if name not in table.column_names]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Present columns: {', '.join(table.column_names)}."
        )


def _coord_system_name(coord_system: CoordinateSystem | None) -> str | None:
    """Return the meta.json sidecar name of a coordinate system.

    PyO3 enum doesn't have .name, so we extract it from the string representation.
    """
    if coord_system is None:
        return None
    return str(coord_system).split(".")[-1]


def _coord_system_from_name(name: str | None) -> CoordinateSystem | None:
    """Recover the coordinate system from its meta.json sidecar name.

    PyO3 enum doesn't support subscripting, use getattr instead.
    """
    if name is None:
        return None
    return getattr(CoordinateSystem, name)


def _meta_payload(coord_system: CoordinateSystem | None) -> dict:
    """The single construction point for the meta.json sidecar content.

    ``save()`` (persisted to disk) and ``share()`` (written into /dev/shm)
    both produce the sidecar through this function, and ``load()`` and
    ``load_from_shm()`` both parse it through ``_coord_system_from_name`` —
    the persisted and shared-memory paths share a single basis for
    coord_system.
    """
    return {"version": TRAJDL_VERSION, "coord_system": _coord_system_name(coord_system)}


class BaseArrowDataset(ABC):
    def __init__(
        self,
        table: pa.Table,
        unsafe: bool = False,
        coord_system: CoordinateSystem | None = None,
    ):
        """
        Initialize the BaseArrowDataset.

        Parameters
        ----------
        table : pa.Table
            The input Arrow table to initialize the dataset.
        unsafe : bool, optional
            If set to True, the correctness of the input table will not be checked.
        coord_system : CoordinateSystem, optional
            The coordinate system of the trajectory data. Default is None.

        Raises
        ------
        ValueError
            If unsafe is False and the table fails the validation check.
        """
        super().__init__()
        if not unsafe:
            self.check_table(table)
        self.core_table = table
        self.feature_table = None
        self._coord_system = coord_system

    @property
    def coord_system(self) -> CoordinateSystem:
        """
        Get the coordinate system of the dataset.

        Returns
        -------
        CoordinateSystem
            The coordinate system of the trajectory data.
        """
        return self._coord_system

    @property
    def seq(self) -> pa.lib.ChunkedArray:
        """
        Get the SEQ column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The SEQ column as a ChunkedArray.
        """
        return self.core_table.column(ArrowColName.SEQ.value)

    @property
    def entity_id(self) -> pa.lib.ChunkedArray:
        """
        Get the ENTITY_ID column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The ENTITY_ID column as a ChunkedArray.
        """
        return self.core_table.column(ArrowColName.ENTITY_ID.value)

    @property
    def start_ts(self) -> pa.lib.ChunkedArray:
        """
        Get the START_TS column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The START_TS column as a ChunkedArray.
        """
        return self.core_table.column(ArrowColName.START_TS.value)

    @property
    def ts_seq(self) -> pa.lib.ChunkedArray:
        """
        Get the TS_SEQ column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The TS_SEQ column as a ChunkedArray.
        """
        return self.core_table.column(ArrowColName.TS_SEQ.value)

    @classmethod
    @abstractmethod
    def schema(cls) -> pa.lib.Schema:
        """
        Abstract method to define the Arrow schema for the dataset.

        Returns
        -------
        pa.lib.Schema
            The schema definition of the dataset.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractmethod
    def check_table(self, table: pa.Table) -> None:
        """
        Check the validity of the input table.

        Parameters
        ----------
        table : pa.Table
            The table to validate.

        Raises
        ------
        ValueError
            If the table type is not correct.
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @classmethod
    def init_from_seqs(
        cls,
        seqs: Iterable[LocSeq | Trajectory],
        chunk_size: int = 8192,
        use_rust: bool = True,
    ) -> "BaseArrowDataset":
        """
        Initialize the dataset from an iterable of location sequences.

        A single ``for`` loop drives both the LocSeq and Trajectory paths:
        each sequence is type-checked against the subclass's schema as it
        is consumed, accumulated into a per-batch buffer, and — once the
        buffer reaches ``chunk_size`` — handed to either the Rust
        ``RecordBatch`` builder (``use_rust=True``, default; ~10× faster
        on large inputs) or the legacy Python ``pa.RecordBatch.from_pydict``
        path (``use_rust=False``, kept for parity testing and as a
        fallback). Generators and lists are treated identically; nothing
        about the public API depends on the input being materialized.

        For ``TrajectoryDataset``, the coordinate system is **inferred
        from the inputs**: every ``Trajectory`` already carries its own
        ``coord_system``, so the dataset's coord_system is simply the
        single value shared by all inputs. Mixed coordinate systems
        within one call are rejected. There is no separate
        ``coord_system`` parameter — that would be a redundant second
        source of truth, and a wrong value would silently mislabel the
        dataset. To project trajectories into another coordinate system,
        do so explicitly before calling (e.g.
        ``traj.to_web_mercator()``); this method never rewrites
        coordinate values.

        ``LocSeqDataset`` has no coordinates of its own; the dataset's
        ``coord_system`` is always ``None``.

        Parameters
        ----------
        seqs : Iterable[LocSeq | Trajectory]
            Sequence objects to ingest. Generators are fully supported and
            preferred for large inputs to keep peak memory bounded by
            ``chunk_size`` items at a time.
        chunk_size : int, optional
            Number of sequences accumulated before flushing into a
            ``pa.RecordBatch``. Default: 8192.
        use_rust : bool, optional
            If ``True`` (default), each chunk is converted via the Rust
            zero-copy builder under ``trajdl.datasets.arrow.builder``.
            If ``False``, the chunk is built via Python's
            ``pa.RecordBatch.from_pydict`` — slower but useful for
            equivalence testing and as a portability fallback.

        Returns
        -------
        BaseArrowDataset
            A new dataset whose ``core_table`` is the concatenation of all
            per-chunk ``RecordBatch`` instances.

        Raises
        ------
        ValueError
            If ``chunk_size <= 0``; if any sequence's runtime type does
            not match the schema; or, on the Trajectory path, if the
            input sequences carry zero or more than one distinct
            coordinate system.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        is_traj_schema = cls.schema() == TRAJ_ARROW_SCHEMA
        schema = cls.schema()

        expected_type = Trajectory if is_traj_schema else LocSeq
        builder = make_record_batch_builder(schema, use_rust=use_rust)

        batches: List[pa.RecordBatch] = []
        # Only populated on the Trajectory path; LocSeq has no coordinates.
        all_coord_systems: set = set()

        def _flush_chunk() -> None:
            if len(builder) == 0:
                return
            batches.append(builder.build())
            builder.clear()

        for idx, seq in tenumerate(seqs, desc="constructing dataset"):
            if not isinstance(seq, expected_type):
                raise ValueError(
                    f"item at index {idx} is {type(seq).__name__}, "
                    f"expected {expected_type.__name__} for this dataset"
                )

            if is_traj_schema:
                all_coord_systems.add(seq.coord_system)
            builder.append(seq)

            if (idx + 1) % chunk_size == 0:
                _flush_chunk()

        _flush_chunk()

        arrow_table = pa.Table.from_batches(batches, schema=schema)

        # Trajectory: infer coord_system from inputs (single source of
        # truth). Empty input → 0 coord systems → error, matching the
        # historical contract.
        # LocSeq: coord_system is always None (no coordinates).
        coord_system: CoordinateSystem | None = None
        if is_traj_schema:
            if len(all_coord_systems) != 1:
                raise ValueError("only 1 coordinate system should exists in seqs")
            coord_system = next(iter(all_coord_systems))

        return cls(table=arrow_table, coord_system=coord_system)

    @classmethod
    def init_from_arrow(
        cls,
        table: pa.Table,
        coord_system: CoordinateSystem,
    ):
        """
        Initialize the dataset from an Arrow table.

        Parameters
        ----------
        table : pa.Table
            The Arrow table to initialize the dataset.
        coord_system : CoordinateSystem
            The coordinate system of the trajectory data.

        Returns
        -------
        BaseArrowDataset
            An instance of the dataset initialized from the Arrow table.

        Warns
        -----
        RuntimeWarning
            If any field in the schema does not exist in the input table.
        """
        col_names = {field.name for field in table.schema}
        arrays = []
        for field in cls.schema():
            if field.name not in col_names:
                warnings.warn(
                    f"Field {field} does not exist in the input table.", RuntimeWarning
                )
                arrays.append(pa.array([None] * len(table), type=field.type))
            else:
                arrays.append(table[field.name])
        new_table = pa.Table.from_arrays(arrays, schema=cls.schema())
        return cls(table=new_table, coord_system=coord_system)

    @classmethod
    def init_from_table(
        cls,
        table: pa.Table | pl.DataFrame | pd.DataFrame,
        coord_system: CoordinateSystem,
    ) -> "BaseArrowDataset":
        """
        Initialize the dataset from a Polars, Pandas, or Arrow table.

        Parameters
        ----------
        table : pa.Table | pl.DataFrame | pd.DataFrame
            The input table to initialize the dataset.
        coord_system : CoordinateSystem
            The coordinate system of the trajectory data.

        Returns
        -------
        BaseArrowDataset
            An instance of the dataset initialized from the provided table.

        Raises
        ------
        ValueError
            If the input is not one of the accepted table types.
        """
        return cls.init_from_arrow(
            table=df_to_arrow(table=table), coord_system=coord_system
        )

    @classmethod
    def _init_from_parquet(
        cls,
        path: str,
        coord_system: CoordinateSystem,
    ) -> "BaseArrowDataset":
        """
        Initialize the dataset from the core Parquet file (internal method).

        This method is used internally by load() to read the core table
        (core_table.parquet) of a saved dataset. The core table contains
        the essential sequence data (seq, entity_id, start_ts, ts_seq).

        Parameters
        ----------
        path : str
            The file path to the core Parquet file (e.g., dataset_dir/core_table.parquet).
            This should be an explicit file path, not a directory.
        coord_system : CoordinateSystem
            The coordinate system of the trajectory data.

        Returns
        -------
        BaseArrowDataset
            An instance of the dataset initialized from the core Parquet file.

        Notes
        -----
        For loading a complete dataset including feature tables, use load() instead.
        This method only reads the core table and does not handle feature tables.

        """
        return cls.init_from_arrow(pq.read_table(path), coord_system=coord_system)

    def __len__(self) -> int:
        """
        Get the number of rows in the dataset.

        Returns
        -------
        int
            The number of rows in the dataset.
        """
        return len(self.core_table)

    def __getitem__(self, idx: Union[int, np.int64]) -> "BaseArrowDataset":
        """
        Retrieve a single row from the dataset.

        Parameters
        ----------
        idx : Union[int, np.int64]
            The index of the row to retrieve.

        Returns
        -------
        BaseArrowDataset
            A new dataset containing the requested rows.
        """
        ds = self.__class__(
            table=self.core_table.slice(idx, length=1),
            unsafe=True,
            coord_system=self._coord_system,
        )
        if not self.feature_table:
            return ds
        ds.set_feature_table(self.feature_table.slice(idx, length=1))
        return ds

    def __getitems__(
        self, indices: List[Union[int, Iterable[int], Iterable[np.int64], np.ndarray]]
    ) -> "BaseArrowDataset":
        """
        Retrieve multiple rows from the dataset.

        Parameters
        ----------
        indices : List[Union[int, Iterable[int], Iterable[np.int64], np.ndarray]]
            A list of indices. Can be a flat list of integers or a list of iterables which will be flattened.

        Returns
        -------
        BaseArrowDataset
            A new dataset containing the requested rows.
        """
        if all(isinstance(tmp, (int, np.integer)) for tmp in indices):
            new_indices = indices
        else:
            new_indices = [idx for index_list in indices for idx in index_list]
        rows = (self.core_table.slice(idx, length=1) for idx in new_indices)
        core_table = pa.concat_tables(rows)
        ds = self.__class__(
            table=core_table, unsafe=True, coord_system=self._coord_system
        )
        if not self.feature_table:
            return ds

        rows = (self.feature_table.slice(idx, length=1) for idx in new_indices)
        ds.set_feature_table(pa.concat_tables(rows))
        return ds

    def get_feature(self, idx: int, feature_col: str) -> Any:
        """Get a single feature value by row index and column name.

        Parameters
        ----------
        idx : int
            Row index.
        feature_col : str
            Column name in the feature table.

        Returns
        -------
        Any
            The Python-native value at ``feature_table[feature_col][idx]``.

        Raises
        ------
        ValueError
            If ``feature_table`` is None or ``feature_col`` does not exist.
        """
        if not self.feature_table:
            raise ValueError(
                "feature_table is None. You should set feature_table at first."
            )

        if feature_col not in self.feature_table.column_names:
            raise ValueError(f"feature_col: {feature_col} not exists.")

        return self.feature_table[feature_col][idx].as_py()

    def iter_feature(self, feature_col: str) -> Generator[Any, None, None]:
        if not self.feature_table:
            raise ValueError(
                "feature_table is None. You should set feature_table at first."
            )

        if feature_col not in self.feature_table.column_names:
            raise ValueError(f"feature_col: {feature_col} not exists.")

        for item in self.feature_table[feature_col]:
            yield item.as_py()

    def set_feature_table(self, table: pa.Table) -> None:
        feature_table = df_to_arrow(table)

        if len(feature_table) != len(self.core_table):
            raise ValueError(
                "size of feature_table should be equal to current dataset."
            )
        self.feature_table = feature_table

    def with_feature_table(self, feature_table: pa.Table) -> "BaseArrowDataset":
        self.set_feature_table(feature_table)
        return self

    def save(self, path: str | Path, force: bool = False) -> None:
        """
        Save the dataset to a folder.

        Parameters
        ----------
        path : str | Path
            The folder to save the dataset.

        force: bool, optional
            If set to True, TrajDL will overwrite some files in this folder.

        Notes
        -----
        The dataset is stored in a folder that contains:
        - core_table.parquet: The core data (seq, entity_id, start_ts, ts_seq)
        - feature_table.parquet: Optional feature data
        - meta.json: Metadata including coordinate system
        """
        p = Path(path) if isinstance(path, str) else path
        if p.exists() and not force:
            raise FileExistsError(f"path {p} already exists.")

        os.makedirs(p, exist_ok=True)
        pq.write_table(self.core_table, p / CORE_TABLE_FILENAME)
        if self.feature_table:
            pq.write_table(self.feature_table, p / FEATURE_TABLE_FILENAME)

        # Save metadata (coord_system travels in the meta.json sidecar)
        with open(p / META_FILENAME, "wb") as f:
            f.write(orjson.dumps(_meta_payload(self._coord_system)))

    @classmethod
    def load(cls, path: str | Path) -> "BaseArrowDataset":
        """
        Load the dataset from a folder.

        Parameters
        ----------
        path : str | Path
            The folder to load the dataset.
        """
        p = Path(path) if isinstance(path, str) else path
        if not p.exists():
            raise FileNotFoundError(f"path {p} not found.")

        # Load metadata if exists (for backward compatibility)
        meta_path = p / META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"cannot find {meta_path}, consider to upgrade your TrajDL version."
            )

        with open(meta_path, "rb") as f:
            meta = orjson.loads(f.read())

        coord_system = _coord_system_from_name(meta.get("coord_system"))

        ds = cls._init_from_parquet(p / CORE_TABLE_FILENAME, coord_system=coord_system)
        if (p / FEATURE_TABLE_FILENAME).exists():
            ds.set_feature_table(pq.read_table(p / FEATURE_TABLE_FILENAME))
        return ds

    def to_polars(self) -> Tuple[pl.DataFrame]:
        if self.feature_table:
            return (
                pl.from_arrow(self.core_table),
                pl.from_arrow(self.feature_table),
            )
        else:
            return (pl.from_arrow(self.core_table),)

    @classmethod
    @abstractmethod
    def seq_type(cls) -> SeqType:
        """
        Return the trajectory sequence type of the dataset.
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def iter_as_seqs(self):
        for idx in range(len(self.core_table)):
            yield self.get_view(idx)

    @contextmanager
    def share(self):
        """Write the dataset into /dev/shm and yield a ``shm_handle``.

        The handle maps file names (``core_table.parquet`` / optional
        ``feature_table.parquet``) to their shared-memory paths, and always
        carries a ``meta.json`` sidecar entry so that the coordinate system
        travels with the handle—``load_from_shm()`` then reconstructs the
        dataset under the very same coord_system semantics as ``load()``.
        All files (meta sidecar included) are removed when the context exits.
        """
        shm_handle = None

        try:
            prefix = f"/dev/shm/TrajDLDataset.{uuid.uuid4().hex}"
            core_table_path = prefix + "_" + CORE_TABLE_FILENAME
            shm_handle = {CORE_TABLE_FILENAME: core_table_path}

            with pa.OSFile(core_table_path, "wb") as sink:
                with pa.ipc.new_file(sink, self.core_table.schema) as writer:
                    writer.write_table(self.core_table)

            if self.feature_table:
                feature_table_path = prefix + "_" + FEATURE_TABLE_FILENAME
                shm_handle[FEATURE_TABLE_FILENAME] = feature_table_path
                with pa.OSFile(feature_table_path, "wb") as sink:
                    with pa.ipc.new_file(sink, self.feature_table.schema) as writer:
                        writer.write_table(self.feature_table)

            # The meta.json sidecar mirrors the one save() writes on disk, so
            # load_from_shm() recovers coord_system through the very same
            # parser as load(). coord_system has no guessable default:
            # handles without the sidecar are rejected outright (see
            # load_from_shm).
            meta_path = prefix + "_" + META_FILENAME
            shm_handle[META_FILENAME] = meta_path
            with open(meta_path, "wb") as f:
                f.write(orjson.dumps(_meta_payload(self._coord_system)))

            yield shm_handle
        finally:
            if shm_handle:
                print(f"Cleaning up shared memory files from context: {shm_handle}")
                for path in shm_handle.values():
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except OSError as e:
                        print(f"Error removing shared memory file {path}: {e}")

    @classmethod
    def load_from_shm(cls, shm_handle: Dict[str, str]) -> "BaseArrowDataset":
        """Reconstruct the dataset from files written by ``share()``.

        The coordinate system is recovered from the ``meta.json`` sidecar
        entry of ``shm_handle`` (same sidecar content and parser as the
        on-disk ``load()``), so no coord_system parameter is needed here.

        Raises
        ------
        ValueError
            If ``shm_handle`` carries no ``meta.json`` sidecar (e.g. a stale
            handle produced by an older ``share()``); the coord_system is
            never assumed, the handle must be regenerated.
        """
        meta_path = shm_handle.get(META_FILENAME)
        if meta_path is None:
            raise ValueError(
                f"shm_handle carries no {META_FILENAME} sidecar; regenerate it "
                "with share() from this TrajDL version — coord_system is read "
                "from the sidecar and never assumed."
            )
        with open(meta_path, "rb") as f:
            meta = orjson.loads(f.read())
        coord_system = _coord_system_from_name(meta.get("coord_system"))

        with pa.memory_map(shm_handle[CORE_TABLE_FILENAME], "rb") as source:
            with pa.ipc.open_file(source) as reader:
                ds = cls.init_from_arrow(reader.read_all(), coord_system=coord_system)
        if FEATURE_TABLE_FILENAME not in shm_handle:
            return ds
        with pa.memory_map(shm_handle[FEATURE_TABLE_FILENAME], "rb") as source:
            with pa.ipc.open_file(source) as reader:
                ds.set_feature_table(reader.read_all())
        return ds

    def get_view(self, idx: int) -> LocSeqView | TrajectoryView:
        constructor = None
        if self.seq_type() == SeqType.LOC_SEQ:
            constructor = LocSeqView
            return constructor(self.core_table, idx)
        elif self.seq_type() == SeqType.TRAJECTORY:
            constructor = TrajectoryView
            return constructor(self.core_table, idx, coord_system=self._coord_system)
        else:
            raise ValueError("do not support this seq_type")
