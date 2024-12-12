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

import math
from dataclasses import dataclass
from typing import Generator, List

import numpy as np
from torch.utils.data import Sampler

from ..arrow.abstract import BaseArrowDataset


@dataclass
class SeqInfo:
    sample_idx: int
    size: int


def mean_size(bucket: List[SeqInfo]) -> float:
    return np.mean([seq_info.size for seq_info in bucket]).item()


def generate_sub_buckets(
    stats: List[SeqInfo], num_buckets: int = 10
) -> List[List[SeqInfo]]:
    """Generate sub-buckets from statistics.

    Parameters
    ----------
    stats : List[SeqInfo]
        A list of SeqInfo where each instance contains a sample index and length.
    num_buckets : int, optional
        The desired number of buckets (default is 10).

    Returns
    -------
    List[List[Tuple[int, int]]]
        A list of sub-buckets containing the statistics.

    Raises
    ------
    ValueError
        If num_buckets is less than 1.
    """
    if num_buckets < 1:
        raise ValueError("`num_buckets` must be at least 1.")

    num_samples = len(stats)
    if num_samples < num_buckets:
        num_buckets = num_samples

    sorted_array = sorted(stats, key=lambda x: (x.size, x.sample_idx))
    num_samples_per_bucket = math.floor(num_samples / num_buckets)

    buckets = [
        sorted_array[i * num_samples_per_bucket : (i + 1) * num_samples_per_bucket]
        for i in range(num_buckets)
    ]
    return buckets


def generate_buckets_by_stats(
    stats: List[SeqInfo], num_buckets: int
) -> List[np.ndarray]:
    all_buckets = [
        np.array([seq_info.sample_idx for seq_info in b], dtype=int)
        for b in generate_sub_buckets(stats, num_buckets=num_buckets)
    ]
    return all_buckets


def generate_buckets(
    seqs: BaseArrowDataset,
    num_buckets: int,
) -> List[np.ndarray]:
    """Generate buckets from sequences.

    Parameters
    ----------
    seqs : BaseArrowDataset
        A LocSeqDataset or a TrajectoryDataset.
    num_buckets : int
        The desired number of buckets.

    Returns
    -------
    List[np.ndarray]
        A list of NumPy arrays, where each array contains sample indices for a bucket.
    """
    seqs = seqs.seq

    stats = [
        SeqInfo(sample_idx=sample_idx, size=len(seq))
        for sample_idx, seq in enumerate(seqs)
    ]

    return generate_buckets_by_stats(stats=stats, num_buckets=num_buckets)


class BucketSampler(Sampler):
    """Sampler that produces batches from randomly selected buckets.

    Attributes
    ----------
    ds: BaseArrowDataset
        Dataset.
    num_buckets: int
        The number of buckets.
    num_batches : int
        The number of batches to yield.
    batch_size : int
        The size of each batch.
    """

    def __init__(
        self,
        ds: BaseArrowDataset,
        num_buckets: int,
        num_batches: int,
        batch_size: int,
        seed=None,
    ):
        """Initialize the BucketSampler.

        Parameters
        ----------
        ds: BaseArrowDataset
            Dataset.
        num_buckets: int
            The number of buckets.
        num_batches : int
            The number of batches to yield from the sampler.
        batch_size : int
            The size of each batch.
        seed : int, optional
            Random seed for reproducibility (default is None).
        """
        self.buckets = generate_buckets(seqs=ds, num_buckets=num_buckets)
        self.num_buckets = len(self.buckets)
        self.batch_size = batch_size
        self.num_batches = num_batches

        if seed is not None:
            np.random.seed(seed)

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.num_batches

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Yield batches of sample indices."""
        for _ in range(self.num_batches):
            random_bucket = self.buckets[np.random.choice(self.num_buckets)]
            yield np.random.choice(random_bucket, self.batch_size)
