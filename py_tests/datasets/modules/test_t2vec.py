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

"""Tests for T2VECDataModule (loc-seq format, tokenized at preprocessing)."""

import tempfile
from pathlib import Path
from typing import List

import pyarrow as pa
import pytest
import torch
from trajdl import (
    CoordinateSystem,
    LocSeq,
    RectangleBoundary,
    SimpleGridSystem,
    Trajectory,
    wgs84,
)
from trajdl.common.samples import T2VECSample
from trajdl.datasets.arrow.locseq import LocSeqDataset
from trajdl.datasets.modules.t2vec import T2VECDataModule
from trajdl.tokenizers.t2vec import T2VECTokenizer


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
    """Generate 20 sample trajectories within Porto boundary."""
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


@pytest.fixture(scope="module")
def dataset_dir(sample_trajs: List[Trajectory], tokenizer: T2VECTokenizer) -> str:
    """Create a temporary directory with src and trg LocSeqDatasets."""
    tmpdir = tempfile.mkdtemp()
    tmppath = Path(tmpdir)

    # trg dataset: token sequences of the originals
    trg_seqs = [
        LocSeq.from_list(
            tokenizer.traj_to_loc_seq(traj, add_bos=False, add_eos=False),
            entity_id=traj.entity_id,
        )
        for traj in sample_trajs
    ]
    trg_ds = LocSeqDataset.init_from_seqs(trg_seqs)
    trg_ds.save(tmppath / "trg_ds.parquet")

    # src dataset: degraded trajectories (subsample to first 20 points)
    # 2 degradation variants per original = 40 total
    src_seqs: List[LocSeq] = []
    trg_idx_list: List[int] = []

    for trg_idx, traj in enumerate(sample_trajs):
        coords = traj.seq
        for drop in [5, 10]:
            degraded = Trajectory.from_list(
                coords[: len(coords) - drop], CoordinateSystem.Wgs84
            )
            src_seqs.append(
                LocSeq.from_list(
                    tokenizer.traj_to_loc_seq(degraded, add_bos=False, add_eos=False),
                    entity_id=degraded.entity_id,
                )
            )
            trg_idx_list.append(trg_idx)

    src_ds = LocSeqDataset.init_from_seqs(src_seqs)
    feature_table = pa.table({"trg_idx": pa.array(trg_idx_list, type=pa.int64())})
    src_ds.set_feature_table(feature_table)
    src_ds.save(tmppath / "src_ds.parquet")

    return tmpdir


class TestT2VECDataModule:
    def test_setup_creates_datasets(self, dataset_dir: str, tokenizer: T2VECTokenizer):
        """DataModule setup should load both src and trg datasets."""
        tmppath = Path(dataset_dir)
        dm = T2VECDataModule(
            tokenizer=tokenizer,
            train_src_path=str(tmppath / "src_ds.parquet"),
            train_trg_path=str(tmppath / "trg_ds.parquet"),
            val_src_path=str(tmppath / "src_ds.parquet"),
            val_trg_path=str(tmppath / "trg_ds.parquet"),
            train_batch_size=8,
            val_batch_size=8,
            num_cpus=0,
        )
        dm.setup("fit")

        assert dm.train_ds is not None
        assert dm.val_ds is not None
        assert len(dm.train_ds) == 40  # 20 trajs * 2 degradations
        assert len(dm.val_ds) == 40

    def test_collate_fn_produces_t2vec_sample(
        self, dataset_dir: str, tokenizer: T2VECTokenizer
    ):
        """Collate function should produce a valid T2VECSample."""
        tmppath = Path(dataset_dir)
        dm = T2VECDataModule(
            tokenizer=tokenizer,
            train_src_path=str(tmppath / "src_ds.parquet"),
            train_trg_path=str(tmppath / "trg_ds.parquet"),
            val_src_path=str(tmppath / "src_ds.parquet"),
            val_trg_path=str(tmppath / "trg_ds.parquet"),
            train_batch_size=4,
            val_batch_size=4,
            num_cpus=0,
        )
        dm.setup("fit")

        # Get a batch manually
        batch = [dm.train_ds[i] for i in range(4)]
        sample = dm.collate_fn(batch)

        assert isinstance(sample, T2VECSample)
        assert isinstance(sample.src, torch.Tensor)
        assert isinstance(sample.target, torch.Tensor)
        assert sample.src.dtype == torch.long
        assert sample.target.dtype == torch.long
        assert sample.src.shape[0] == 4  # batch_size
        assert sample.target.shape[0] == 4
        assert len(sample.lengths) == 4
        assert all(length > 0 for length in sample.lengths)

    def test_collate_fn_lengths_match_src(
        self, dataset_dir: str, tokenizer: T2VECTokenizer
    ):
        """Lengths should match the actual token sequence lengths before padding."""
        tmppath = Path(dataset_dir)
        dm = T2VECDataModule(
            tokenizer=tokenizer,
            train_src_path=str(tmppath / "src_ds.parquet"),
            train_trg_path=str(tmppath / "trg_ds.parquet"),
            val_src_path=str(tmppath / "src_ds.parquet"),
            val_trg_path=str(tmppath / "trg_ds.parquet"),
            train_batch_size=4,
            val_batch_size=4,
            num_cpus=0,
        )
        dm.setup("fit")

        batch = [dm.train_ds[i] for i in range(4)]
        sample = dm.collate_fn(batch)

        # Max length in batch == padded tensor width
        max_src_len = max(sample.lengths)
        assert sample.src.shape[1] == max_src_len

    def test_train_dataloader(self, dataset_dir: str, tokenizer: T2VECTokenizer):
        """Train dataloader should yield valid batches."""
        tmppath = Path(dataset_dir)
        dm = T2VECDataModule(
            tokenizer=tokenizer,
            train_src_path=str(tmppath / "src_ds.parquet"),
            train_trg_path=str(tmppath / "trg_ds.parquet"),
            val_src_path=str(tmppath / "src_ds.parquet"),
            val_trg_path=str(tmppath / "trg_ds.parquet"),
            train_batch_size=8,
            val_batch_size=8,
            num_cpus=0,
        )
        dm.setup("fit")

        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))

        assert isinstance(batch, T2VECSample)
        assert batch.src.shape[0] == 8

    def test_val_dataloader(self, dataset_dir: str, tokenizer: T2VECTokenizer):
        """Val dataloader should yield valid batches."""
        tmppath = Path(dataset_dir)
        dm = T2VECDataModule(
            tokenizer=tokenizer,
            train_src_path=str(tmppath / "src_ds.parquet"),
            train_trg_path=str(tmppath / "trg_ds.parquet"),
            val_src_path=str(tmppath / "src_ds.parquet"),
            val_trg_path=str(tmppath / "trg_ds.parquet"),
            train_batch_size=8,
            val_batch_size=4,
            num_cpus=0,
        )
        dm.setup("fit")

        dataloader = dm.val_dataloader()
        batch = next(iter(dataloader))

        assert isinstance(batch, T2VECSample)
        assert batch.src.shape[0] == 4
