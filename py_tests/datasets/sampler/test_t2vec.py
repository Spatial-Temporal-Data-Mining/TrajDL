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

"""Tests for T2VECSampler (replication of the original t2vec sampler)."""

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pytest
from trajdl import (
    CoordinateSystem,
    LocSeq,
    RectangleBoundary,
    SimpleGridSystem,
    Trajectory,
    wgs84,
)
from trajdl.datasets.arrow.ext.t2vec import T2VECPairDataset
from trajdl.datasets.arrow.locseq import LocSeqDataset
from trajdl.datasets.sampler.t2vec import T2VECSampler
from trajdl.tokenizers.t2vec import T2VECTokenizer
from trajdl.trajdl_rust import bucketize

# original t2vec first-match ladder (anchor repo t2vec/t2vec.py default)
ORIG_BUCKETS = [
    (20, 30),
    (30, 30),
    (30, 50),
    (50, 50),
    (50, 70),
    (70, 70),
    (70, 100),
    (100, 100),
]

BOUNDS = [(20, 30), (30, 50), (50, 70), (70, 100)]


def original_insert(src_len: int, trg_len: int) -> int:
    """Verbatim port of DataLoader.insert's first-match bucket search."""
    for i, (src_bound, trg_bound) in enumerate(ORIG_BUCKETS):
        if src_len <= src_bound and trg_len <= trg_bound:
            return i
    return -1


class TestBucketizeEquivalence:
    def test_bucketize_matches_original_ladder(self):
        """bucketize with BOUNDS must equal the original 8-bucket ladder."""
        src_bound, trg_bound = zip(*BOUNDS)

        pairs = [(s, t) for s in range(0, 115, 3) for t in range(0, 115, 3)]
        rng = np.random.default_rng(0)
        pairs += [
            (int(rng.integers(1, 130)), int(rng.integers(1, 130))) for _ in range(5000)
        ]

        src_lengths = np.array([s for s, _ in pairs], dtype=np.int64)
        trg_lengths = np.array([t for _, t in pairs], dtype=np.int64)
        src_indices = np.arange(len(pairs), dtype=np.int64)

        buckets_map = bucketize(
            src_lengths, src_indices, trg_lengths, list(src_bound), list(trg_bound)
        )

        rust_key = {}
        for key, idx_list in buckets_map.items():
            for idx in idx_list:
                rust_key[idx] = key

        for i, (s, t) in enumerate(pairs):
            assert rust_key.get(i, -1) == original_insert(s, t), (i, s, t)


@pytest.fixture(scope="module")
def porto_boundary() -> RectangleBoundary:
    return RectangleBoundary(wgs84(-8.735152, 40.953673), wgs84(-8.156309, 41.307945))


@pytest.fixture(scope="module")
def porto_grid(porto_boundary: RectangleBoundary) -> SimpleGridSystem:
    return SimpleGridSystem(
        boundary=porto_boundary.to_web_mercator(), step_x=100.0, step_y=100.0
    )


@pytest.fixture(scope="module")
def sample_trajs() -> List[Trajectory]:
    trajs = []
    for i in range(20):
        coords = [[-8.6 + j * 0.002 + i * 0.001, 41.15 + j * 0.001] for j in range(30)]
        trajs.append(
            Trajectory.from_list(coords, CoordinateSystem.Wgs84, entity_id=str(i))
        )
    return trajs


@pytest.fixture(scope="module")
def tokenizer(
    porto_boundary: RectangleBoundary,
    porto_grid: SimpleGridSystem,
    sample_trajs: List[Trajectory],
) -> T2VECTokenizer:
    return T2VECTokenizer.fit(
        web_grid=porto_grid,
        gps_boundary=porto_boundary,
        splits=[sample_trajs],
        max_vocab_size=10000,
        min_freq=1,
        with_kd_tree=True,
    )


def _build_src_trajs(
    sample_trajs: List[Trajectory],
) -> Tuple[List[Trajectory], List[int]]:
    src_trajs: List[Trajectory] = []
    trg_idx_list: List[int] = []
    for trg_idx, traj in enumerate(sample_trajs):
        coords = traj.seq
        for drop in [5, 10]:
            src_trajs.append(
                Trajectory.from_list(
                    coords[: len(coords) - drop], CoordinateSystem.Wgs84
                )
            )
            trg_idx_list.append(trg_idx)
    return src_trajs, trg_idx_list


@pytest.fixture(scope="module")
def pair_dataset(
    sample_trajs: List[Trajectory],
    tokenizer: T2VECTokenizer,
) -> T2VECPairDataset:
    tmpdir = tempfile.mkdtemp()
    tmppath = Path(tmpdir)

    trg_seqs = [
        LocSeq.from_list(
            tokenizer.traj_to_loc_seq(traj, add_bos=False, add_eos=False),
            entity_id=traj.entity_id,
        )
        for traj in sample_trajs
    ]
    trg_ds = LocSeqDataset.init_from_seqs(trg_seqs)
    trg_ds.save(tmppath / "trg_ds.parquet")

    src_trajs, trg_idx_list = _build_src_trajs(sample_trajs)
    src_seqs = [
        LocSeq.from_list(
            tokenizer.traj_to_loc_seq(traj, add_bos=False, add_eos=False),
            entity_id=traj.entity_id,
        )
        for traj in src_trajs
    ]

    src_ds = LocSeqDataset.init_from_seqs(src_seqs)
    src_ds.set_feature_table(
        pa.table({"trg_idx": pa.array(trg_idx_list, type=pa.int64())})
    )
    src_ds.save(tmppath / "src_ds.parquet")

    return T2VECPairDataset(
        LocSeqDataset.load(tmppath / "src_ds.parquet"),
        LocSeqDataset.load(tmppath / "trg_ds.parquet"),
    )


class TestT2VECSampler:
    def test_dist_is_normalized(self, pair_dataset: T2VECPairDataset):
        sampler = T2VECSampler(pair_dataset, BOUNDS, 10, 8)
        assert np.isclose(sampler.dist.sum(), 1.0)
        assert len(sampler) == 10

    def test_sampled_indices_are_bucket_members(self, pair_dataset: T2VECPairDataset):
        sampler = T2VECSampler(pair_dataset, BOUNDS, 20, 8)
        all_members = set().union(*[set(b) for b in sampler.buckets if b])

        np.random.seed(42)
        for batch in sampler:
            assert len(batch) == 8
            assert all(int(i) in all_members for i in batch)

    def test_seq_lengths_match_stored_sequences(self, pair_dataset: T2VECPairDataset):
        """Vectorized lengths must equal the per-row stored token counts."""
        for dataset in (pair_dataset.src_dataset, pair_dataset.trg_dataset):
            lengths = T2VECSampler._seq_lengths(dataset)
            assert len(lengths) == len(dataset)
            for i in range(len(dataset)):
                assert lengths[i] == len(dataset.get_view(i).seq)

    def test_trg_length_includes_bos_eos(
        self, pair_dataset: T2VECPairDataset, tokenizer: T2VECTokenizer
    ):
        """src counts stored tokens; trg adds 2 for BOS/EOS, matching the
        original t2vec length convention used by the bucket ladder."""
        src_lengths = T2VECSampler._seq_lengths(pair_dataset.src_dataset)
        trg_lengths = T2VECSampler._seq_lengths(pair_dataset.trg_dataset)

        for i in range(len(pair_dataset.trg_dataset)):
            expected = len(
                tokenizer.encode_locseq(
                    pair_dataset.trg_dataset.get_view(i).seq,
                    add_bos=True,
                    add_eos=True,
                )
            )
            assert trg_lengths[i] + 2 == expected

        for i in range(len(pair_dataset.src_dataset)):
            assert src_lengths[i] == len(
                tokenizer.encode_locseq(pair_dataset.src_dataset.get_view(i).seq)
            )
