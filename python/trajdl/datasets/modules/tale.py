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

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...common.samples import TALEPretrainSample
from ..arrow.convert import flat_values_offsets
from .abstract import BaseArrowDataset, BaseLocSeqDataModule


class TALEPairTable:
    """
    Columnar store of the flat (context, slot) pre-training rows.

    Row semantics match the baseline flat pair list (issue #116): one row per
    time slot of one center visit, and ``train_batch_size`` counts rows.
    Contexts are fixed-width arrays; Huffman paths are zero-padded to the
    table width. Inner node ids start from 1 (0 is the reserved padding id),
    so trailing zeros are pure padding and ``batch`` trims them to the batch
    maximum — batches are numerically identical to the legacy ``zip_longest``
    collate while steady-state memory drops by ~4x (no per-row Python
    objects).
    """

    _CHUNK_ROWS = 1 << 16

    def __init__(self, ctx_width: int, pos_width: int, neg_width: int):
        self.ctx_width = ctx_width
        self.pos_width = pos_width
        self.neg_width = neg_width
        self._ctx = np.zeros((0, ctx_width), dtype=np.int64)
        self._pos_w = np.zeros((0, pos_width), dtype=np.int64)
        self._neg_w = np.zeros((0, neg_width), dtype=np.int64)
        self._pos_s = np.zeros(0, dtype=np.int64)
        self._prob = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._pos_s.shape[0])

    @staticmethod
    def _rows_to_arrays(
        rows: List[tuple], ctx_width: int, pos_width: int, neg_width: int
    ) -> Tuple[np.ndarray, ...]:
        n = len(rows)
        ctx = np.zeros((n, ctx_width), dtype=np.int64)
        pos = np.zeros((n, pos_width), dtype=np.int64)
        neg = np.zeros((n, neg_width), dtype=np.int64)
        pos_s = np.empty(n, dtype=np.int64)
        prob = np.empty(n, dtype=np.float32)
        for i, (context, pos_w, neg_w, s, p) in enumerate(rows):
            ctx[i] = context
            pos[i, : len(pos_w)] = pos_w
            neg[i, : len(neg_w)] = neg_w
            pos_s[i] = s
            prob[i] = p
        return ctx, pos, neg, pos_s, prob

    @classmethod
    def _concat(
        cls,
        parts: List[Tuple[np.ndarray, ...]],
        ctx_width: int,
        pos_width: int,
        neg_width: int,
    ) -> "TALEPairTable":
        table = cls(ctx_width, pos_width, neg_width)
        table._ctx = np.concatenate([p[0] for p in parts], axis=0)
        table._pos_w = np.concatenate([p[1] for p in parts], axis=0)
        table._neg_w = np.concatenate([p[2] for p in parts], axis=0)
        table._pos_s = np.concatenate([p[3] for p in parts], axis=0)
        table._prob = np.concatenate([p[4] for p in parts], axis=0)
        return table

    @classmethod
    def from_path_pairs(
        cls,
        path_pairs: List[list],
        ctx_width: int,
        pos_width: int,
        neg_width: int,
    ) -> "TALEPairTable":
        # Flatten (context, slot) pairs exactly like the baseline training
        # loop (one row per time slot of one center visit), converting to
        # columnar chunks so the list-of-lists peak stays bounded.
        parts: List[Tuple[np.ndarray, ...]] = []
        chunk: List[tuple] = []
        for row in path_pairs:
            context = row[0]
            for pos_w, neg_w, pos_s, prop in zip(*row[1:]):
                chunk.append((context, pos_w, neg_w, pos_s, prop))
            if len(chunk) >= cls._CHUNK_ROWS:
                parts.append(
                    cls._rows_to_arrays(chunk, ctx_width, pos_width, neg_width)
                )
                chunk = []
        if chunk:
            parts.append(cls._rows_to_arrays(chunk, ctx_width, pos_width, neg_width))
        if not parts:
            return cls(ctx_width, pos_width, neg_width)
        return cls._concat(parts, ctx_width, pos_width, neg_width)

    def batch(self, idxs: Any) -> TALEPretrainSample:
        idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)
        if idxs.size == 0:
            return TALEPretrainSample(
                pos_u=torch.empty(0, self.ctx_width, dtype=torch.long),
                pos_w=torch.empty(0, 0, dtype=torch.long),
                neg_w=torch.empty(0, 0, dtype=torch.long),
                pos_s=torch.empty(0, dtype=torch.long),
                prob=torch.empty(0),
            )
        ctx = np.take(self._ctx, idxs, axis=0)
        pos = np.take(self._pos_w, idxs, axis=0)
        neg = np.take(self._neg_w, idxs, axis=0)
        pos_w = int((pos != 0).sum(axis=1).max()) if self.pos_width else 0
        neg_w = int((neg != 0).sum(axis=1).max()) if self.neg_width else 0
        return TALEPretrainSample(
            pos_u=torch.from_numpy(ctx),
            pos_w=torch.from_numpy(np.ascontiguousarray(pos[:, :pos_w])),
            neg_w=torch.from_numpy(np.ascontiguousarray(neg[:, :neg_w])),
            pos_s=torch.from_numpy(np.take(self._pos_s, idxs, axis=0)),
            prob=torch.from_numpy(np.take(self._prob, idxs, axis=0)),
        )


@dataclass
class TaleDataModule(BaseLocSeqDataModule):
    """
    Data module for TALE pre-training.

    Emits (context, slot) rows: every visit generates one row per time slot
    it belongs to, following the baseline training loop.

    Batching is row-level: ``train_batch_size`` counts flat (context, slot)
    rows, matching the baseline ``next_batch(shuffle(train_set), batch_size)``
    loop where ``train_set`` is a flat pair list. Batching sequences and
    flattening inside the collate would instead put every pair of the batched
    sequences into one optimizer step (an effective batch tens of times
    larger and proportionally fewer updates over the fixed 10 epochs).

    Parameters
    ----------
    forest_path : str, optional
        Path to the pickled ``SliceHuffmanForest`` built offline from the
        training corpus with the same tokenizer as ``tokenizer``.
    window_size : int, optional
    indi_context : bool, optional
        If True, each context token forms its own pair (individual context).
    """

    forest_path: Optional[str] = None
    window_size: int = 2
    indi_context: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.forest = None
        if self.forest_path is not None:
            import pickle

            with open(self.forest_path, "rb") as f:
                self.forest = pickle.load(f)
        self._ctx_width = 1 if self.indi_context else 2 * self.window_size
        self._pos_width, self._neg_width = self._path_widths()
        self.train_pairs: TALEPairTable = TALEPairTable(
            self._ctx_width, self._pos_width, self._neg_width
        )
        self.val_pairs: TALEPairTable = TALEPairTable(
            self._ctx_width, self._pos_width, self._neg_width
        )

    def _path_widths(self) -> Tuple[int, int]:
        # 表内 pos_w/neg_w 的全局填充宽度：森林所有树的所有叶子路径的最大长度。
        # 0 = 单叶树（无内节点路径）或未加载森林。
        if self.forest is None:
            return 0, 0
        return self.forest.max_pos_path_width(), self.forest.max_neg_path_width()

    def _gen_path_pairs(
        self, sentences: List[Any], timestamps: List[Any]
    ) -> List[list]:
        # skip_missing：验证集里存在森林未覆盖的 (poi, minute)（baseline 预训练
        # 没有验证集，不会遇到），跳过而不是 KeyError
        return self.forest.gen_path_pairs(
            sentences,
            timestamps,
            self.window_size,
            self.indi_context,
            skip_missing=True,
        )

    def _materialize_pairs(self, ds: Optional[BaseArrowDataset]) -> TALEPairTable:
        if ds is None:
            return TALEPairTable(self._ctx_width, self._pos_width, self._neg_width)
        sentences, timestamps = self._encode_columns(ds)
        return TALEPairTable.from_path_pairs(
            self._gen_path_pairs(sentences, timestamps),
            self._ctx_width,
            self._pos_width,
            self._neg_width,
        )

    def _encode_columns(self, ds: BaseArrowDataset):
        """Encode the whole column in one flattening pass (zero-copy flat+offsets read), replacing per-row ListScalar iteration."""
        loc_flat, loc_offsets = self.tokenizer.encode_locseq_batch(ds.seq)
        ts_values, ts_offsets = flat_values_offsets(ds.ts_seq)
        sentences = [
            loc_flat[s:e].tolist() for s, e in zip(loc_offsets[:-1], loc_offsets[1:])
        ]
        timestamps = [
            ts_values[s:e].tolist() for s, e in zip(ts_offsets[:-1], ts_offsets[1:])
        ]
        return sentences, timestamps

    def setup(self, stage: str):
        super().setup(stage)
        # 与 baseline 一致：一次性物化全语料的扁平 pair 行（列式存储）
        self.train_pairs = self._materialize_pairs(self.train_ds)
        self.val_pairs = self._materialize_pairs(self.val_ds)

    def collate_function(self, ds: BaseArrowDataset) -> TALEPretrainSample:
        table = self._materialize_pairs(ds)
        return table.batch(np.arange(len(table)))

    def train_dataloader(self):
        return DataLoader(
            range(len(self.train_pairs)),
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.train_pairs.batch,
            num_workers=self.num_cpus,
            pin_memory=True,
        )

    def val_dataloader(self):
        if not self.val_pairs:
            return []
        return DataLoader(
            range(len(self.val_pairs)),
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.val_pairs.batch,
            num_workers=self.num_cpus,
            pin_memory=True,
        )
