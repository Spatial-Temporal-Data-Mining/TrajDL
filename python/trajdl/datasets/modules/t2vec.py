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

"""T2VEC DataModule using standard LocSeqDataset format.

Loads paired loc seq datasets (src = degraded, trg = original) stored as
standard ``LocSeqDataset`` directories. Grid discretization and kd-tree
vocabulary remapping happen once during preprocessing; the collate function
only maps stored token strings to ids and pads.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Tuple, Union

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from ...common.samples import T2VECSample
from ...tokenizers import AbstractTokenizer
from ...utils import get_num_cpus, load_tokenizer
from ..arrow import LocSeqDataset
from ..arrow.convert import varlen_to_padded


@dataclass
class T2VECDataModule(L.LightningDataModule):
    """T2VEC DataModule using standard LocSeqDataset format.

    Loads paired loc seq datasets (src = degraded, trg = original) stored
    as standard ``LocSeqDataset`` directories (tokenized at preprocessing).
    The collate function maps stored token strings to ids and pads.

    Training batches are produced by ``T2VECSampler``, which replicates the
    original t2vec bucket sampling (bucket chosen proportionally to its
    size, members drawn with replacement). One epoch is
    ``num_train_batches`` batches, not a pass over the dataset.

    Parameters
    ----------
    tokenizer : Union[str, Path, AbstractTokenizer]
        Path to tokenizer or tokenizer instance.
    train_src_path : str
        Path to training src (degraded) LocSeqDataset directory.
    train_trg_path : str
        Path to training trg (original) LocSeqDataset directory.
    val_src_path : str
        Path to validation src (degraded) LocSeqDataset directory.
    val_trg_path : str
        Path to validation trg (original) LocSeqDataset directory.
    train_batch_size : int
        Batch size for training.
    val_batch_size : int
        Batch size for validation.
    num_train_batches : int
        Number of batches per training epoch.
    buckets_boundaries : List[Tuple[int, int]]
        Bucket boundaries for the training sampler. The default
        ``[(20, 30), (30, 50), (50, 70), (70, 100)]`` encodes the original
        t2vec 8-bucket ladder; see ``T2VECSampler`` for the equivalence.
    num_cpus : int
        Number of DataLoader workers. Default: -1 (use all CPUs).
    """

    tokenizer: Union[str, Path, AbstractTokenizer] = None
    train_src_path: str = ""
    train_trg_path: str = ""
    val_src_path: str = ""
    val_trg_path: str = ""
    path_params: ClassVar[tuple[str, ...]] = (
        "tokenizer",
        "train_src_path",
        "train_trg_path",
        "val_src_path",
        "val_trg_path",
    )
    train_batch_size: int = 128
    val_batch_size: int = 128
    num_train_batches: int = 1000
    buckets_boundaries: List[Tuple[int, int]] = field(
        default_factory=lambda: [(20, 30), (30, 50), (50, 70), (70, 100)]
    )
    num_cpus: int = -1

    def __post_init__(self):
        super().__init__()
        self.num_workers = get_num_cpus() if self.num_cpus == -1 else self.num_cpus

    def setup(self, stage: str):
        from ..arrow.ext.t2vec import T2VECPairDataset
        from ..sampler.t2vec import T2VECSampler

        if isinstance(self.tokenizer, str):
            self.tokenizer = load_tokenizer(self.tokenizer)

        train_src_ds = LocSeqDataset.load(self.train_src_path)
        train_trg_ds = LocSeqDataset.load(self.train_trg_path)
        self.train_ds = T2VECPairDataset(train_src_ds, train_trg_ds)

        val_src_ds = LocSeqDataset.load(self.val_src_path)
        val_trg_ds = LocSeqDataset.load(self.val_trg_path)
        self.val_ds = T2VECPairDataset(val_src_ds, val_trg_ds)

        self.train_sampler = T2VECSampler(
            self.train_ds,
            self.buckets_boundaries,
            self.num_train_batches,
            self.train_batch_size,
        )

    def collate_fn(self, batch: List[Tuple]) -> T2VECSample:
        """Collate function that maps stored token sequences to id tensors.

        Each item in batch is a (src_view, trg_view) tuple of LocSeqView
        objects; their token strings were produced by the tokenizer at
        preprocessing time (kd-tree remapping included), so this only maps
        vocabulary tokens to ids and pads.

        Parameters
        ----------
        batch : List[Tuple[LocSeqView, LocSeqView]]
            List of (degraded_loc_seq, original_loc_seq) pairs.

        Returns
        -------
        T2VECSample
            Padded batch with src tokens, lengths, and target tokens.
        """
        # 批量编码（views 的 .seq 已是 Python list，走 list-of-lists 扁平路径），
        # 替代逐条 encode_locseq + pad_sequence
        src_flat, src_offsets = self.tokenizer.encode_locseq_batch(
            [degraded.seq for degraded, _ in batch]
        )
        trg_flat, trg_offsets = self.tokenizer.encode_locseq_batch(
            [original.seq for _, original in batch], add_bos=True, add_eos=True
        )

        return T2VECSample(
            src=torch.from_numpy(
                varlen_to_padded(src_flat, src_offsets, pad_value=self.tokenizer.pad)
            ),
            lengths=[int(x) for x in np.diff(src_offsets)],
            target=torch.from_numpy(
                varlen_to_padded(trg_flat, trg_offsets, pad_value=self.tokenizer.pad)
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
