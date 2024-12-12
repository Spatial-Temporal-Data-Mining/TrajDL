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

import random
import re

import numpy as np
import pytest

from trajdl.datasets import LocSeqDataset, TrajectoryDataset
from trajdl.datasets.sampler.bucket import (
    BucketSampler,
    SeqInfo,
    generate_buckets,
    generate_buckets_by_stats,
    generate_sub_buckets,
    mean_size,
)


def test_seqinfo():
    seq_info = SeqInfo(sample_idx=0, size=-1)
    assert seq_info.sample_idx == 0
    assert seq_info.size == -1


def test_generate_sub_buckets():
    stats = [
        SeqInfo(sample_idx=sample_idx, size=sample_idx) for sample_idx in range(10)
    ]
    sub_buckets = generate_sub_buckets(stats=stats, num_buckets=15)
    assert len(sub_buckets) == len(stats)

    sub_buckets = generate_sub_buckets(stats=stats, num_buckets=5)
    for sub_bucket in sub_buckets:
        assert len(sub_bucket) == 2

    with pytest.raises(
        ValueError, match=re.escape("`num_buckets` must be at least 1.")
    ):
        generate_sub_buckets(stats=stats, num_buckets=0)

    with pytest.raises(
        ValueError, match=re.escape("`num_buckets` must be at least 1.")
    ):
        generate_sub_buckets(stats=stats, num_buckets=-1)

    for _ in range(10):
        stats = [
            SeqInfo(sample_idx=sample_idx, size=random.random() * 100)
            for sample_idx in range(random.randint(1, 100))
        ]
        sub_buckets = generate_sub_buckets(
            stats=stats, num_buckets=random.randint(1, 100)
        )
        for idx in range(len(sub_buckets) - 1):
            assert len(sub_buckets[idx]) >= len(sub_buckets[idx + 1])
            assert mean_size(sub_buckets[idx]) <= mean_size(sub_buckets[idx + 1])


def test_generate_buckets_by_stats():
    for _ in range(10):
        num_samples = random.randint(1, 100)
        num_buckets = random.randint(1, 100)
        stats = [
            SeqInfo(sample_idx=sample_idx, size=random.random() * 100)
            for sample_idx in range(num_samples)
        ]
        buckets = generate_buckets_by_stats(stats=stats, num_buckets=num_buckets)
        assert len(buckets) == (
            num_buckets if num_samples >= num_buckets else num_samples
        )


def test_generate_buckets(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    buckets = generate_buckets(seqs=test_locseq_dataset, num_buckets=5)
    assert np.allclose([b.sum() for b in buckets], [1742, 1830, 1977, 2087, 1695])
    assert np.allclose(
        [b.mean() for b in buckets],
        [64.5185185, 67.7777777, 73.2222222, 77.2962962, 62.7777777],
    )

    buckets = generate_buckets(seqs=test_trajectory_dataset, num_buckets=5)
    assert np.allclose(
        [b.sum() for b in buckets], [9915392, 9931568, 10198402, 9932410, 10017228]
    )
    assert np.allclose(
        [b.mean() for b in buckets], [4957.696, 4965.784, 5099.201, 4966.205, 5008.614]
    )


def test_bucket_sampler(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    for ds in [test_locseq_dataset, test_trajectory_dataset]:
        for _ in range(10):
            num_buckets, num_batches, batch_size = (
                random.randint(1, 10),
                random.randint(1, 10),
                random.randint(1, 1024),
            )
            sampler = BucketSampler(
                ds=ds,
                num_buckets=num_buckets,
                num_batches=num_batches,
                batch_size=batch_size,
            )
            assert len(sampler) == num_batches
            for indices in sampler:
                assert isinstance(indices, np.ndarray)
                assert indices.shape == (batch_size,)

        # test seed
        seed = random.randint(0, 10000)
        sampler1 = BucketSampler(
            ds=ds,
            num_buckets=num_buckets,
            num_batches=num_batches,
            batch_size=batch_size,
            seed=seed,
        )
        indices1 = next(iter(sampler1))

        sampler2 = BucketSampler(
            ds=ds,
            num_buckets=num_buckets,
            num_batches=num_batches,
            batch_size=batch_size,
            seed=seed,
        )
        indices2 = next(iter(sampler2))

        assert len(sampler1) == len(sampler2)
        assert (indices1 == indices2).all()
