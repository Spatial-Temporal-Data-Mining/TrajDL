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

"""Tests for TrajCL DataModule and collate function."""

import numpy as np
import pytest
import torch
from trajdl import CoordinateSystem, Trajectory
from trajdl.common.samples import TrajCLSample
from trajdl.datasets import TrajectoryDataset
from trajdl.datasets.modules.trajcl import (
    TrajCLDataModule,
    trajcl_augment,
    trajcl_random_augment,
)
from trajdl.tokenizers import TrajCLTokenizer


class TestTrajCLAugment:
    """Tests for trajectory augmentation functions."""

    def test_subset_augmentation(self):
        """Test subset augmentation reduces trajectory length."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
            CoordinateSystem.Wgs84,
        )
        result = trajcl_augment(traj, "subset", ratio=0.6)
        assert len(result) == 3  # 5 * 0.6 = 3
        assert result[0] == pytest.approx([0.0, 0.0]) or len(result) >= 2

    def test_mask_augmentation(self):
        """Test mask augmentation removes points."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
            CoordinateSystem.Wgs84,
        )
        result = trajcl_augment(traj, "mask", ratio=0.4, min_length=2)
        assert len(result) >= 2  # At least min_length
        assert len(result) <= 5  # Not more than original

    def test_shift_augmentation(self):
        """Test shift augmentation modifies coordinates."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]]),
            CoordinateSystem.Wgs84,
        )
        result = trajcl_augment(
            traj, "shift", mu=0.0, sigma=0.5, factor=100.0, bound=100.0
        )
        assert len(result) == 3
        # Coordinates should be shifted (not identical)
        # But within bound

    def test_invalid_augmentation_type(self):
        """Test invalid augmentation type raises error."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [1.0, 0.0]]), CoordinateSystem.Wgs84
        )
        with pytest.raises(ValueError, match="Unknown augmentation type"):
            trajcl_augment(traj, "invalid_type")

    def test_straight_augmentation_is_identity_on_trajectory(self):
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), CoordinateSystem.Wgs84
        )
        result = trajcl_augment(traj, "straight")
        assert isinstance(result, Trajectory)
        assert result.coord_system == CoordinateSystem.Wgs84
        assert np.allclose(result.seq, traj.seq)

    def test_straight_augmentation_is_identity_on_view(
        self, test_trajectory_dataset: TrajectoryDataset
    ):
        """Regression: collate feeds TrajectoryView (from iter_as_seqs), which
        has no to_container — 'straight' must still return an equal Trajectory."""
        view = next(iter(test_trajectory_dataset.iter_as_seqs()))
        result = trajcl_augment(view, "straight")
        assert isinstance(result, Trajectory)
        assert result.coord_system == view.coord_system
        assert np.allclose(result.seq, view.seq)

    def test_random_augment(self):
        """Test random augmentation selects from valid types."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
            CoordinateSystem.Wgs84,
        )
        aug_types = ["subset", "mask", "shift"]
        result = trajcl_random_augment(traj, aug_types, ratio=0.7, min_length=2)
        assert len(result) >= 2  # At least min_length


class TestTrajCLSample:
    """Tests for TrajCLSample dataclass."""

    def test_sample_creation(self):
        """Test creating a TrajCLSample."""
        sample = TrajCLSample(
            view1_cell_seq=torch.randint(0, 100, (4, 10)),
            view1_spatial_feat=torch.randn(4, 10, 4),
            view1_lengths=[5, 8, 3, 10],
            view2_cell_seq=torch.randint(0, 100, (4, 12)),
            view2_spatial_feat=torch.randn(4, 12, 4),
            view2_lengths=[6, 9, 4, 12],
        )
        assert sample.batch_size == 4

    def test_sample_batch_size_property(self):
        """Test batch_size property returns correct value."""
        sample = TrajCLSample(
            view1_cell_seq=torch.randint(0, 100, (2, 5)),
            view1_spatial_feat=torch.randn(2, 5, 4),
            view1_lengths=[3, 5],
            view2_cell_seq=torch.randint(0, 100, (2, 6)),
            view2_spatial_feat=torch.randn(2, 6, 4),
            view2_lengths=[4, 6],
        )
        assert sample.batch_size == 2


class TestTrajCLDataModule:
    """Tests for TrajCLDataModule."""

    def test_datamodule_initialization(self, trajcl_tokenizer: TrajCLTokenizer):
        """Test DataModule can be initialized."""
        dm = TrajCLDataModule(
            tokenizer=trajcl_tokenizer,
            train_batch_size=4,
            val_batch_size=4,
        )
        assert dm.aug_types == ["subset", "mask", "shift"]
        assert dm.train_batch_size == 4

    def test_datamodule_custom_aug_types(self, trajcl_tokenizer: TrajCLTokenizer):
        """Test DataModule with custom augmentation types."""
        dm = TrajCLDataModule(
            tokenizer=trajcl_tokenizer,
            aug_types=["subset", "mask"],
        )
        assert dm.aug_types == ["subset", "mask"]

    def test_datamodule_setup(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
        test_trajectory_dataset: TrajectoryDataset,
    ):
        """Test DataModule setup creates samplers."""
        dm = TrajCLDataModule(
            tokenizer=trajcl_tokenizer,
            train_table=test_trajectory_dataset,
            val_table=test_trajectory_dataset,
            train_batch_size=4,
            val_batch_size=4,
        )
        dm.setup("fit")

    def test_collate_function(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
        test_trajectory_dataset: TrajectoryDataset,
    ):
        """Test collate_function returns valid TrajCLSample."""
        dm = TrajCLDataModule(
            tokenizer=trajcl_tokenizer,
            train_table=test_trajectory_dataset,
            train_batch_size=4,
        )
        dm.setup("fit")

        # Get a small batch using proper indexing
        small_ds = test_trajectory_dataset.__getitems__([0, 1, 2, 3])
        sample = dm.collate_function(small_ds)

        assert isinstance(sample, TrajCLSample)
        assert sample.batch_size == 4
        assert sample.view1_cell_seq.shape[0] == 4
        assert sample.view2_cell_seq.shape[0] == 4
        assert sample.view1_spatial_feat.shape[0] == 4
        assert sample.view1_spatial_feat.shape[2] == 4  # 4 spatial features

    def test_collate_function_for_inference(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
        test_trajectory_dataset: TrajectoryDataset,
    ):
        """Test collate_function_for_inference returns identical views."""
        dm = TrajCLDataModule(
            tokenizer=trajcl_tokenizer,
            train_table=test_trajectory_dataset,
            train_batch_size=4,
        )
        dm.setup("fit")

        small_ds = test_trajectory_dataset.__getitems__([0, 1, 2, 3])
        sample = dm.collate_function_for_inference(small_ds)

        assert isinstance(sample, TrajCLSample)
        assert sample.view1_lengths == sample.view2_lengths
        assert torch.equal(sample.view1_cell_seq, sample.view2_cell_seq)

    def test_train_dataloader(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
        test_trajectory_dataset: TrajectoryDataset,
    ):
        """Test train_dataloader returns valid batches."""
        dm = TrajCLDataModule(
            tokenizer=trajcl_tokenizer,
            train_table=test_trajectory_dataset,
            train_batch_size=4,
        )
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert isinstance(batch, TrajCLSample)
        assert batch.batch_size <= 4
