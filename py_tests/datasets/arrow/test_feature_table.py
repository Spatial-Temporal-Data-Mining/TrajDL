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

"""Tests for BaseArrowDataset.get_feature and feature table operations."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pytest
from trajdl import CoordinateSystem, Trajectory
from trajdl.datasets.arrow.traj import TrajectoryDataset


@pytest.fixture
def small_trajectory_dataset() -> TrajectoryDataset:
    """Create a small TrajectoryDataset with 5 trajectories for testing."""
    trajs = [
        Trajectory.from_list(
            [[i + j * 0.1, 40.0 + j * 0.01] for j in range(10)],
            CoordinateSystem.Wgs84,
        )
        for i in range(5)
    ]
    return TrajectoryDataset.init_from_seqs(trajs)


@pytest.fixture
def dataset_with_features(
    small_trajectory_dataset: TrajectoryDataset,
) -> TrajectoryDataset:
    """TrajectoryDataset with a feature table attached."""
    feature_table = pa.table(
        {
            "label": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "score": pa.array([0.1, 0.2, 0.3, 0.4, 0.5], type=pa.float64()),
        }
    )
    small_trajectory_dataset.set_feature_table(feature_table)
    return small_trajectory_dataset


class TestGetFeature:
    """Tests for BaseArrowDataset.get_feature."""

    def test_get_feature_int_column(self, dataset_with_features: TrajectoryDataset):
        assert dataset_with_features.get_feature(0, "label") == 10
        assert dataset_with_features.get_feature(4, "label") == 50

    def test_get_feature_float_column(self, dataset_with_features: TrajectoryDataset):
        assert dataset_with_features.get_feature(0, "score") == pytest.approx(0.1)
        assert dataset_with_features.get_feature(2, "score") == pytest.approx(0.3)

    def test_get_feature_all_indices(self, dataset_with_features: TrajectoryDataset):
        expected_labels = [10, 20, 30, 40, 50]
        for idx, expected in enumerate(expected_labels):
            assert dataset_with_features.get_feature(idx, "label") == expected

    def test_get_feature_no_feature_table(
        self, small_trajectory_dataset: TrajectoryDataset
    ):
        with pytest.raises(ValueError, match="feature_table is None"):
            small_trajectory_dataset.get_feature(0, "label")

    def test_get_feature_nonexistent_column(
        self, dataset_with_features: TrajectoryDataset
    ):
        with pytest.raises(ValueError, match="not exists"):
            dataset_with_features.get_feature(0, "nonexistent_col")

    def test_get_feature_consistent_with_iter_feature(
        self, dataset_with_features: TrajectoryDataset
    ):
        """get_feature and iter_feature should return the same values."""
        iter_values = list(dataset_with_features.iter_feature("label"))
        for idx, expected in enumerate(iter_values):
            assert dataset_with_features.get_feature(idx, "label") == expected

    def test_get_feature_survives_save_load(
        self, dataset_with_features: TrajectoryDataset
    ):
        """Feature table should persist through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_ds"
            dataset_with_features.save(save_path)

            loaded = TrajectoryDataset.load(save_path)
            for idx in range(len(loaded)):
                assert loaded.get_feature(
                    idx, "label"
                ) == dataset_with_features.get_feature(idx, "label")
                assert loaded.get_feature(idx, "score") == pytest.approx(
                    dataset_with_features.get_feature(idx, "score")
                )
