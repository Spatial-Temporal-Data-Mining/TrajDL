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
import pyarrow as pa
from torch.utils.data import Sampler

from ... import trajdl_cpp
from ..arrow.ext.t2vec import T2VECDataset


class T2VECSampler(Sampler):
    def __init__(
        self,
        ds: T2VECDataset,
        buckets_boundaries: List[Tuple[int, int]],
        num_batches: int,
        batch_size: int,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size

        src_lengths = pa.compute.list_value_length(
            ds.src_table.column("src")
        ).to_numpy()
        src_indices = ds.src_table.column("label_idx").to_numpy()
        trg_lengths = pa.compute.list_value_length(
            ds.trg_table.column("trg")
        ).to_numpy()

        src_bound, trg_bound = zip(*buckets_boundaries)
        buckets_map: Dict[int, List[int]] = trajdl_cpp.bucketize(
            src_lengths, src_indices, trg_lengths, src_bound, trg_bound
        )
        num_out = 0
        self.buckets = [None] * len(buckets_boundaries) * 2
        for bucket_idx, bucket in buckets_map.items():
            if 0 <= bucket_idx < len(self.buckets):
                self.buckets[bucket_idx] = bucket
            else:
                num_out += len(bucket)
        print(f"num out: {num_out}")

        self.dist = np.array([len(b) for b in self.buckets]) / len(ds)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        for _ in range(self.num_batches):
            sample = np.random.multinomial(1, self.dist)
            bucket_idx = np.nonzero(sample)[0][0]
            yield np.random.choice(len(self.buckets[bucket_idx]), self.batch_size)
