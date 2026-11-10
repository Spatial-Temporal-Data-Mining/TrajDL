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

from typing import Dict, Generator, List, Tuple

import numpy as np
import pyarrow.compute as pc
from torch.utils.data import Sampler

from ...trajdl_rust import bucketize
from ..arrow.abstract import BaseArrowDataset
from ..arrow.ext.t2vec import T2VECPairDataset


class T2VECSampler(Sampler):
    """Bucket sampler replicating the original t2vec training data loader.

    Batch generation matches the baseline's ``t2vec/data_utils.py::
    DataLoader.getbatch_one`` exactly:

    1. A bucket is selected with probability proportional to its size
       (``np.random.multinomial``, the original ``self.p``).
    2. ``batch_size`` members are drawn from the chosen bucket **with
       replacement** (``np.random.choice``), the original
       ``self.srcdata[bucket][idx]``.

    ``buckets_boundaries=[(20, 30), (30, 50), (50, 70), (70, 100)]`` is a
    compact encoding of the original 8-bucket first-match ladder
    ``[(20, 30), (30, 30), (30, 50), (50, 50), (50, 70), (70, 70),
    (70, 100), (100, 100]]``: the Rust ``bucketize`` assigns
    ``bucket_key = 2 * p + sub`` which equals the original bucket index for
    every sample, and discards exactly those with
    ``max(src_len, trg_len) > 100`` (verified by brute-force enumeration).

    Lengths follow the original convention: ``src`` counts merged tokens
    without BOS/EOS, ``trg`` includes BOS/EOS (``+ 2``). The pair datasets
    store tokenized sequences (``LocSeqDataset``), so lengths are a
    vectorized ``pa.compute.list_value_length`` over the seq column — no
    tokenization happens here.
    """

    def __init__(
        self,
        ds: T2VECPairDataset,
        buckets_boundaries: List[Tuple[int, int]],
        num_batches: int,
        batch_size: int,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size

        src_lengths = self._seq_lengths(ds.src_dataset)
        trg_lengths = self._seq_lengths(ds.trg_dataset) + 2
        src_indices = ds.src_dataset.feature_table[
            T2VECPairDataset.FEATURE_COL_TRG_IDX
        ].to_numpy()

        src_bound, trg_bound = zip(*buckets_boundaries)
        buckets_map: Dict[int, List[int]] = bucketize(
            src_lengths, src_indices, trg_lengths, list(src_bound), list(trg_bound)
        )

        self.buckets: List[List[int]] = [[] for _ in range(len(buckets_boundaries) * 2)]
        for bucket_idx, bucket in buckets_map.items():
            if 0 <= bucket_idx < len(self.buckets):
                self.buckets[bucket_idx] = bucket

        num_in_buckets = sum(len(bucket) for bucket in self.buckets)
        print(
            f"num out: {len(ds) - num_in_buckets} "
            "(samples outside all buckets, never sampled)"
        )

        # normalize over in-bucket samples so the probabilities sum to 1,
        # matching the original `allocation / sum(allocation)`
        self.dist = np.array([len(bucket) for bucket in self.buckets]) / num_in_buckets

    @staticmethod
    def _seq_lengths(dataset: BaseArrowDataset) -> np.ndarray:
        return np.asarray(
            pc.list_value_length(dataset.seq).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        for _ in range(self.num_batches):
            sample = np.random.multinomial(1, self.dist)
            bucket_idx = np.nonzero(sample)[0][0]
            yield np.random.choice(self.buckets[bucket_idx], self.batch_size)
