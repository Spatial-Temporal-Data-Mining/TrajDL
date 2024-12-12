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
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ...common.enum import ArrowColName
from ..base import BaseSeq


class BaseArrowDataset(ABC):
    def __init__(self, table: pa.Table, unsafe: bool = False):
        """
        Initialize the BaseArrowDataset.

        Parameters
        ----------
        table : pa.Table
            The input Arrow table to initialize the dataset.
        unsafe : bool, optional
            If set to True, the correctness of the input table will not be checked.

        Raises
        ------
        ValueError
            If unsafe is False and the table fails the validation check.
        """
        super().__init__()
        if not unsafe:
            self.check_table(table)
        self.table = table

    @property
    def seq(self) -> pa.lib.ChunkedArray:
        """
        Get the SEQ column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The SEQ column as a ChunkedArray.
        """
        return self.table.column(ArrowColName.SEQ.value)

    @property
    def entity_id(self) -> pa.lib.ChunkedArray:
        """
        Get the ENTITY_ID column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The ENTITY_ID column as a ChunkedArray.
        """
        return self.table.column(ArrowColName.ENTITY_ID.value)

    @property
    def ts_seq(self) -> pa.lib.ChunkedArray:
        """
        Get the TS_SEQ column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The TS_SEQ column as a ChunkedArray.
        """
        return self.table.column(ArrowColName.TS_SEQ.value)

    @property
    def ts_delta(self) -> pa.lib.ChunkedArray:
        """
        Get the TS_DELTA column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The TS_DELTA column as a ChunkedArray.
        """
        return self.table.column(ArrowColName.TS_DELTA.value)

    @property
    def dis_delta(self) -> pa.lib.ChunkedArray:
        """
        Get the DIS_DELTA column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The DIS_DELTA column as a ChunkedArray.
        """
        return self.table.column(ArrowColName.DIS_DELTA.value)

    @property
    def start_ts(self) -> pa.lib.ChunkedArray:
        """
        Get the START_TS column from the Arrow table.

        Returns
        -------
        pa.lib.ChunkedArray
            The START_TS column as a ChunkedArray.
        """
        return self.table.column(ArrowColName.START_TS.value)

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
    def init_from_arrow(cls, table: pa.Table):
        """
        Initialize the dataset from an Arrow table.

        Parameters
        ----------
        table : pa.Table
            The Arrow table to initialize the dataset.

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
        return cls(table=new_table)  # Updated to return the new_table

    @classmethod
    def init_from_table(
        cls, table: Union[pa.Table, pl.DataFrame, pd.DataFrame]
    ) -> "BaseArrowDataset":
        """
        Initialize the dataset from a Polars, Pandas, or Arrow table.

        Parameters
        ----------
        table : Union[pa.Table, pl.DataFrame, pd.DataFrame]
            The input table to initialize the dataset.

        Returns
        -------
        BaseArrowDataset
            An instance of the dataset initialized from the provided table.

        Raises
        ------
        ValueError
            If the input is not one of the accepted table types.
        """
        if isinstance(table, pl.DataFrame):
            arrow_table = table.to_arrow()
        elif isinstance(table, pd.DataFrame):
            arrow_table = pa.Table.from_pandas(table)
        elif isinstance(table, pa.Table):
            arrow_table = table
        else:
            raise ValueError(
                "`table` should be an instance of {'pyarrow.Table', 'polars.DataFrame', 'pandas.DataFrame'}"
            )
        return cls.init_from_arrow(table=arrow_table)

    @classmethod
    def init_from_parquet(cls, path: str) -> "BaseArrowDataset":
        """
        Initialize the dataset from a Parquet file.

        Parameters
        ----------
        path : str
            The file path to the Parquet file.

        Returns
        -------
        BaseArrowDataset
            An instance of the dataset initialized from the Parquet file.

        Notes
        -----
        Due to differences in handling List[List[Float32]] across different frameworks,
        it is recommended to read the file using PyArrow and try to convert types where necessary.
        """
        arrow_table = pq.read_table(path)
        return cls.init_from_arrow(arrow_table)

    def __len__(self) -> int:
        """
        Get the number of rows in the dataset.

        Returns
        -------
        int
            The number of rows in the dataset.
        """
        return len(self.table)

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
        return self.__class__(table=self.table.slice(idx, length=1), unsafe=True)

    def __getitems__(
        self, indices: List[Union[int, Iterable[int], Iterable[np.int64], np.ndarray]]
    ) -> "BaseArrowDataset":
        """
        Retrieve multiple rows from the dataset.

        Parameters
        ----------
        indices : List[Union[int, Iterable[int], Iterable[np.int64], np.ndarray]]
            The indices of the rows to retrieve.

        Returns
        -------
        BaseArrowDataset
            A new dataset containing the requested rows.
        """
        if all(isinstance(tmp, (int, np.integer)) for tmp in indices):
            new_indices = indices
        else:
            new_indices = (idx for index_list in indices for idx in index_list)
        rows = (self.table.slice(idx, length=1) for idx in new_indices)
        table = pa.concat_tables(rows)
        return self.__class__(table=table, unsafe=True)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the dataset to a Parquet file.

        Parameters
        ----------
        path : Union[str, Path]
            The file path to save the dataset.

        Notes
        -----
        If the provided path does not end with '.parquet', it will be appended automatically.
        """
        p = Path(path)
        if p.suffix != ".parquet":
            p = p.with_suffix(".parquet")

        folder = p.parent
        os.makedirs(folder, exist_ok=True)
        pq.write_table(self.table, p)

    def to_polars(self) -> pl.DataFrame:
        return pl.from_arrow(self.table)

    @abstractmethod
    def sub_classes_construction(self, *args, **kwargs) -> BaseSeq:
        """
        这个方法是将table的一行数据转换为单条序列的实例
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def iter_as_seqs(self):
        cols = {
            col_name: self.table.column(col_name) for col_name in self.schema().names
        }
        for idx in range(len(self.table)):
            yield self.sub_classes_construction(
                **{col_name: col[idx].as_py() for col_name, col in cols.items()}
            )
