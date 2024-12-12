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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import lightning as L
import pandas as pd
import polars as pl
import pyarrow as pa
from torch.utils.data import DataLoader, Sampler

from ...tokenizers import AbstractTokenizer
from ...utils import get_num_cpus
from ..arrow import BaseArrowDataset, LocSeqDataset, TrajectoryDataset


@dataclass
class BaseSeqDataModule(L.LightningDataModule, ABC):
    """
    Abstract class for sequence data modules.

    Parameters
    ----------
    tokenizer : Union[str, AbstractTokenizer]
        Path of tokenizer or tokenizer instance.
    train_parquet_path : str, optional
        Path to the training parquet file.
    val_parquet_path : str, optional
        Path to the validation parquet file.
    test_parquet_path : str, optional
        Path to the test parquet file.
    train_table : Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset, None], optional
        DataFrame for training that will be transformed into a pyarrow.Table automatically.
    val_table : Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset, None], optional
        DataFrame for validation that will be transformed into a pyarrow.Table automatically.
    test_table : Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset, None], optional
        DataFrame for testing that will be transformed into a pyarrow.Table automatically.
    train_batch_size : int, optional
        Batch size for training.
    val_batch_size : int, optional
        Batch size for validation.
    train_sampler : Union[Sampler, None], optional
        Custom sampler for training.
    val_sampler : Union[Sampler, None], optional
        Custom sampler for validation.
    num_cpus : int, optional
        Number of CPUs to use.
    """

    tokenizer: Union[str, AbstractTokenizer]
    train_parquet_path: Union[str, None] = None
    val_parquet_path: Union[str, None] = None
    test_parquet_path: Union[str, None] = None
    train_table: Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset, None] = (
        None
    )
    val_table: Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset, None] = (
        None
    )
    test_table: Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset, None] = (
        None
    )
    train_batch_size: int = 2
    val_batch_size: int = 2
    train_sampler: Union[Sampler, None] = None
    val_sampler: Union[Sampler, None] = None
    num_cpus: int = 0

    @abstractmethod
    def __post_init__(self):
        """
        Post-initialization method to set up the number of CPUs.
        """
        super().__init__()
        self.num_cpus = get_num_cpus() if self.num_cpus == -1 else self.num_cpus

    @abstractmethod
    def init_from_parquet(self, path: str) -> BaseArrowDataset:
        """
        Load an Arrow dataset from a parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file.

        Returns
        -------
        BaseArrowDataset
            The loaded Arrow dataset.
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractmethod
    def init_from_table(
        self, table: Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset]
    ) -> BaseArrowDataset:
        """
        Load a dataset from a pyarrow table or pandas/polars DataFrame.

        Parameters
        ----------
        table : Union[pa.Table, pl.DataFrame, pd.DataFrame]
            The input table or DataFrame.

        Returns
        -------
        BaseArrowDataset
            The loaded Arrow dataset.
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def _init_ds(self, type_: str) -> BaseArrowDataset:
        """
        Initialize the dataset based on type.

        Parameters
        ----------
        type_ : str
            Type of dataset to initialize. Should be one of {'train', 'val', 'test'}.

        Returns
        -------
        BaseArrowDataset
            The initialized Arrow dataset.

        Raises
        ------
        ValueError
            If type_ is not one of {'train', 'val', 'test'}.
        """
        if type_ not in {"train", "val", "test"}:
            raise ValueError("type_ can only be {'train', 'val', 'test'}")

        config = (
            getattr(self, f"{type_}_parquet_path"),
            getattr(self, f"{type_}_table"),
        )
        func_list = (self.init_from_parquet, self.init_from_table)
        for idx, c in enumerate(config):
            if c is not None:
                return func_list[idx](c)

    def setup(self, stage: str):
        """
        Set up the data module, loading the tokenizer and initializing datasets.

        Parameters
        ----------
        stage : str
            Stage of operation (e.g. 'fit', 'test').
        """
        if isinstance(self.tokenizer, str):
            self.tokenizer: AbstractTokenizer = AbstractTokenizer.load_pretrained(
                self.tokenizer
            )
        elif not isinstance(self.tokenizer, AbstractTokenizer):
            raise ValueError(
                "tokenizer should be path of tokenizer or a tokenizer instance"
            )
        self.train_ds = self._init_ds("train")
        self.val_ds = self._init_ds("val")
        self.test_ds = self._init_ds("test")

    @abstractmethod
    def collate_function(self, ds: BaseArrowDataset):
        """
        Collate function for the dataset. Different modules may require different implementations.

        Parameters
        ----------
        ds : BaseArrowDataset
            The dataset to collate.

        Returns
        -------
        Any
            The collated data.
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def train_dataloader(self):
        """
        Create the training data loader.

        Returns
        -------
        DataLoader
            The data loader for the training dataset.
        """
        return DataLoader(
            self.train_ds,
            batch_size=1 if self.train_sampler else self.train_batch_size,
            collate_fn=self.collate_function,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            num_workers=self.num_cpus,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Create the validation data loader.

        Returns
        -------
        DataLoader
            The data loader for the validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=1 if self.val_sampler else self.val_batch_size,
            collate_fn=self.collate_function,
            sampler=self.val_sampler,
            num_workers=self.num_cpus,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Create the test data loader.

        Returns
        -------
        DataLoader
            The data loader for the test dataset.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_function,
            num_workers=self.num_cpus,
            pin_memory=True,
        )


@dataclass
class BaseLocSeqDataModule(BaseSeqDataModule):
    """
    Abstract class for location sequence data modules.
    """

    def __post_init__(self):
        """
        Post-initialization method.
        """
        super().__post_init__()

    def init_from_parquet(self, path: str) -> LocSeqDataset:
        """
        Load a location sequence dataset from a parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file.

        Returns
        -------
        LocSeqDataset
            The loaded location sequence dataset.
        """
        return LocSeqDataset.init_from_parquet(path)

    def init_from_table(
        self, table: Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset]
    ) -> LocSeqDataset:
        """
        Load a location sequence dataset from a pyarrow table, a pandas/polars DataFrame or a BaseArrowDataset.

        Parameters
        ----------
        table : Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset]
            The input table or DataFrame.

        Returns
        -------
        LocSeqDataset
            The loaded location sequence dataset.
        """
        if isinstance(table, BaseArrowDataset):
            return table
        return LocSeqDataset.init_from_table(table)


@dataclass
class BaseTrajectoryDataModule(BaseSeqDataModule):
    """
    Abstract class for trajectory sequence data modules.
    """

    def __post_init__(self):
        """
        Post-initialization method.
        """
        super().__post_init__()

    def init_from_parquet(self, path: str) -> TrajectoryDataset:
        """
        Load a trajectory dataset from a parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file.

        Returns
        -------
        TrajectoryDataset
            The loaded trajectory dataset.
        """
        return TrajectoryDataset.init_from_parquet(path)

    def init_from_table(
        self, table: Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset]
    ) -> TrajectoryDataset:
        """
        Load a trajectory dataset from a pyarrow table, a pandas/polars DataFrame or a BaseArrowDataset.

        Parameters
        ----------
        table : Union[pa.Table, pl.DataFrame, pd.DataFrame, BaseArrowDataset]
            The input table or DataFrame.

        Returns
        -------
        TrajectoryDataset
            The loaded trajectory dataset.
        """
        if isinstance(table, BaseArrowDataset):
            return table
        return TrajectoryDataset.init_from_table(table)
