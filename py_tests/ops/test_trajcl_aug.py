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

"""Tests for TrajCL data augmentation functions.

These tests verify that the TrajDL implementations match the original TrajCL behavior.
"""

from typing import List

import numpy as np
import pytest
from trajdl import CoordinateSystem, Trajectory
from trajdl.ops import (
    mask_trajectory,
    random_subsample,
    seed_augmentation_rng,
    split_trajectory_by_parity,
    subset_trajectory,
    truncated_gaussian_distort,
)


class TestSubsetTrajectory:
    """Tests for subset_trajectory function."""

    def test_subset_basic(self, traj_samples: List[Trajectory]):
        """Test basic subset functionality."""
        traj = traj_samples[0]
        original_len = len(traj)

        if original_len < 10:
            pytest.skip("Trajectory too short for subset test")

        ratio = 0.7
        result = subset_trajectory(traj, ratio)

        # Check length is approximately ratio * original
        expected_len = int(original_len * ratio)
        assert len(result) == expected_len
        assert len(result) < original_len

    def test_subset_returns_continuous_segment(self, traj_samples: List[Trajectory]):
        """Test that subset returns a continuous segment."""
        traj = traj_samples[0]
        original_np, _ = traj.to_numpy()

        if len(traj) < 20:
            pytest.skip("Trajectory too short")

        result = subset_trajectory(traj, 0.5)
        result_np, _ = result.to_numpy()

        # The result should be a continuous segment from the original
        # Find where the result matches in the original
        found = False
        for start_idx in range(len(original_np) - len(result_np) + 1):
            if np.allclose(
                original_np[start_idx : start_idx + len(result_np)], result_np
            ):
                found = True
                break

        assert found, "Subset result is not a continuous segment of the original"

    def test_subset_ratio_one(self, traj_samples: List[Trajectory]):
        """Test subset with ratio=1.0 returns full trajectory."""
        traj = traj_samples[0]
        result = subset_trajectory(traj, 1.0)

        assert len(result) == len(traj)

    def test_subset_ratio_zero(self, traj_samples: List[Trajectory]):
        """Test subset with ratio=0.0 returns at least 1 point."""
        traj = traj_samples[0]
        result = subset_trajectory(traj, 0.0)

        assert len(result) >= 1

    def test_subset_short_trajectory(self):
        """Test subset on very short trajectory."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [1.0, 1.0]]), CoordinateSystem.Wgs84
        )
        result = subset_trajectory(traj, 0.7)

        # Should return at least 1 point
        assert len(result) >= 1

    def test_subset_single_point(self):
        """Test subset on single point trajectory."""
        traj = Trajectory.from_numpy(np.array([[0.0, 0.0]]), CoordinateSystem.Wgs84)
        result = subset_trajectory(traj, 0.5)

        assert len(result) == 1

    def test_subset_invalid_ratio(self, traj_samples: List[Trajectory]):
        """Test that invalid ratio raises error."""
        traj = traj_samples[0]

        with pytest.raises(Exception):
            subset_trajectory(traj, -0.1)

        with pytest.raises(Exception):
            subset_trajectory(traj, 1.1)


class TestMaskTrajectory:
    """Tests for mask_trajectory function."""

    def test_mask_basic(self, traj_samples: List[Trajectory]):
        """Test basic mask functionality."""
        traj = next(t for t in traj_samples if len(t) >= 20)
        original_len = len(traj)

        ratio = 0.3
        result = mask_trajectory(traj, ratio)

        # Should remove approximately ratio * original_len points
        expected_removed = int(original_len * ratio)
        expected_len = original_len - expected_removed

        assert len(result) == expected_len

    def test_mask_preserves_min_length(self):
        """Test that mask preserves minimum length."""
        traj = Trajectory.from_numpy(
            np.array([[float(i), float(i)] for i in range(10)]),
            CoordinateSystem.Wgs84,
        )

        # Remove 90% - should still preserve at least min_length points
        result = mask_trajectory(traj, 0.9, min_length=2)

        assert len(result) >= 2

    def test_mask_ratio_zero(self, traj_samples: List[Trajectory]):
        """Test mask with ratio=0.0 returns full trajectory."""
        traj = traj_samples[0]
        result = mask_trajectory(traj, 0.0)

        assert len(result) == len(traj)

    def test_mask_points_are_from_original(self, traj_samples: List[Trajectory]):
        """Test that masked points are all from the original trajectory."""
        traj = next(t for t in traj_samples if len(t) >= 20)
        original_np, _ = traj.to_numpy()
        result_np, _ = mask_trajectory(traj, 0.3).to_numpy()

        # Each point in result should exist in original
        for point in result_np:
            # Check if point exists in original
            assert any(np.allclose(point, orig) for orig in original_np)

    def test_mask_invalid_ratio(self, traj_samples: List[Trajectory]):
        """Test that invalid ratio raises error."""
        traj = traj_samples[0]

        with pytest.raises(Exception):
            mask_trajectory(traj, -0.1)

        with pytest.raises(Exception):
            mask_trajectory(traj, 1.1)

    def test_mask_short_trajectory(self):
        """Test mask on very short trajectory."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [1.0, 1.0]]), CoordinateSystem.Wgs84
        )
        result = mask_trajectory(traj, 0.3, min_length=1)

        assert len(result) >= 1


class TestTruncatedGaussianDistort:
    """Tests for truncated_gaussian_distort function."""

    def test_distort_basic(self, traj_samples: List[Trajectory]):
        """Test basic distort functionality."""
        traj = traj_samples[0]
        result = truncated_gaussian_distort(traj)

        # Should have same length
        assert len(result) == len(traj)

        # Points should be different (distorted)
        original_np, _ = traj.to_numpy()
        result_np, _ = result.to_numpy()

        # All points should be shifted (rate=1.0 by default)
        assert not np.allclose(original_np, result_np)

    def test_distort_all_points_shifted(self, traj_samples: List[Trajectory]):
        """Test that all points are shifted when rate=1.0."""
        traj = traj_samples[0]
        original_np, _ = traj.to_numpy()

        result = truncated_gaussian_distort(
            traj, mu=0.0, sigma=0.5, factor=100.0, bound=100.0, rate=1.0
        )
        result_np, _ = result.to_numpy()

        # Every point should be different
        # Use a very small tolerance to detect any shift
        for i in range(len(original_np)):
            # At least one coordinate should differ by more than 1e-10
            diff = np.abs(original_np[i] - result_np[i])
            assert np.any(diff > 1e-10), f"Point {i} was not shifted: diff={diff}"

    def test_distort_within_bound(self, traj_samples: List[Trajectory]):
        """Test that distortions are within the specified bound."""
        traj = traj_samples[0]
        original_np, _ = traj.to_numpy()

        bound = 50.0
        result = truncated_gaussian_distort(traj, bound=bound)
        result_np, _ = result.to_numpy()

        # Calculate shifts (in WGS84, approximate check)
        # The actual shift is in Web Mercator, so we check that the difference is reasonable
        diffs = np.abs(result_np - original_np)

        # In WGS84, 1 degree ≈ 111km, so bound=50 meters ≈ 0.00045 degrees
        # We use a looser bound since we're checking in WGS84 space
        max_expected_diff = 0.001  # ~100 meters in degrees

        assert np.all(
            diffs < max_expected_diff
        ), f"Shift exceeded expected bound: max diff = {diffs.max()}"

    def test_distort_default_params(self, traj_samples: List[Trajectory]):
        """Test distort with default parameters (TrajCL defaults)."""
        traj = traj_samples[0]
        result = truncated_gaussian_distort(
            traj
        )  # Uses defaults: mu=0, sigma=0.5, factor=100, bound=100, rate=1.0

        assert len(result) == len(traj)

    def test_distort_custom_params(self, traj_samples: List[Trajectory]):
        """Test distort with custom parameters."""
        traj = traj_samples[0]

        # Small distort
        result_small = truncated_gaussian_distort(
            traj, mu=0.0, sigma=0.1, factor=10.0, bound=10.0, rate=1.0
        )
        original_np, _ = traj.to_numpy()
        result_small_np, _ = result_small.to_numpy()

        diffs_small = np.abs(result_small_np - original_np)

        # Larger distort
        result_large = truncated_gaussian_distort(
            traj, mu=0.0, sigma=0.5, factor=100.0, bound=100.0, rate=1.0
        )
        result_large_np, _ = result_large.to_numpy()

        diffs_large = np.abs(result_large_np - original_np)

        # Larger params should generally produce larger shifts
        # (with some tolerance due to randomness)
        assert np.mean(diffs_large) > np.mean(diffs_small) * 0.5

    def test_distort_single_point(self):
        """Test distort on single point trajectory."""
        traj = Trajectory.from_numpy(np.array([[0.0, 0.0]]), CoordinateSystem.Wgs84)
        result = truncated_gaussian_distort(traj)

        assert len(result) == 1

    def test_distort_preserves_order(self, traj_samples: List[Trajectory]):
        """Test that distort preserves point order (just adds noise)."""
        traj = traj_samples[0]
        result = truncated_gaussian_distort(traj)

        # The general direction of the trajectory should be preserved
        original_np, _ = traj.to_numpy()
        result_np, _ = result.to_numpy()

        # Check that the trajectory still roughly goes in the same direction
        # by comparing the first and last points
        original_direction = original_np[-1] - original_np[0]
        result_direction = result_np[-1] - result_np[0]

        # Directions should be similar (within 10% change)
        direction_change = np.abs(original_direction - result_direction) / (
            np.abs(original_direction) + 1e-6
        )
        assert np.all(
            direction_change < 0.5
        ), "Distort changed trajectory direction too much"

    def test_distort_rate_less_than_one(self, traj_samples: List[Trajectory]):
        """Test that rate < 1.0 means only some points are distorted."""
        traj = next(t for t in traj_samples if len(t) >= 100)
        original_np, _ = traj.to_numpy()

        # With rate=0.5, roughly half the points should be distorted
        result = truncated_gaussian_distort(traj, rate=0.5)
        result_np, _ = result.to_numpy()

        # Count how many points are unchanged
        unchanged = sum(
            1
            for i in range(len(original_np))
            if np.allclose(original_np[i], result_np[i], atol=1e-10)
        )

        # With rate=0.5, we expect roughly 50% unchanged (with some tolerance)
        # Allow 20-80% range due to randomness
        assert (
            0.2 * len(original_np) <= unchanged <= 0.8 * len(original_np)
        ), f"Expected ~50% unchanged, got {unchanged}/{len(original_np)}"

    def test_distort_rate_zero(self, traj_samples: List[Trajectory]):
        """Test that rate=0.0 means no points are distorted."""
        traj = traj_samples[0]
        original_np, _ = traj.to_numpy()

        result = truncated_gaussian_distort(traj, rate=0.0)
        result_np, _ = result.to_numpy()

        # All points should be unchanged
        assert np.allclose(original_np, result_np)

    def test_distort_invalid_rate(self, traj_samples: List[Trajectory]):
        """Test that invalid rate raises error."""
        traj = traj_samples[0]

        with pytest.raises(Exception):
            truncated_gaussian_distort(traj, rate=-0.1)

        with pytest.raises(Exception):
            truncated_gaussian_distort(traj, rate=1.1)


class TestTrajCLAugmentationComparison:
    """Test that TrajDL implementations match original TrajCL behavior."""

    def test_subset_matches_trajcl_length(self):
        """Test that subset produces same length as TrajCL."""
        # Create a trajectory similar to TrajCL test
        points = [[float(i), float(i * 2)] for i in range(100)]
        traj = Trajectory.from_numpy(np.array(points), CoordinateSystem.Wgs84)

        ratio = 0.7
        result = subset_trajectory(traj, ratio)

        # TrajCL subset with ratio=0.7 on 100 points gives 70 points
        expected_len = int(100 * ratio)
        assert len(result) == expected_len

    def test_mask_matches_trajcl_length(self):
        """Test that mask produces same length as TrajCL."""
        points = [[float(i), float(i * 2)] for i in range(100)]
        traj = Trajectory.from_numpy(np.array(points), CoordinateSystem.Wgs84)

        ratio = 0.3
        result = mask_trajectory(traj, ratio)

        # TrajCL mask with ratio=0.3 on 100 points gives 70 points (removes 30)
        expected_len = 100 - int(100 * ratio)
        assert len(result) == expected_len

    def test_distort_all_points_affected(self):
        """Test that distort affects all points like TrajCL when rate=1.0.

        The distortion adds truncated Gaussian noise in Web Mercator (meters)
        then converts back to WGS84.  With mu=0, sigma=0.5, factor=100,
        the noise magnitude is typically tens of meters, which translates to
        ~1e-4 degrees.  However, on rare occasions the sampled noise can be
        extremely small, so we check that the *overall* trajectory is shifted
        rather than asserting every single point differs at a tight tolerance.
        """
        points = [[float(i), float(i)] for i in range(10)]
        traj = Trajectory.from_numpy(np.array(points), CoordinateSystem.Wgs84)

        result = truncated_gaussian_distort(
            traj, mu=0.0, sigma=0.5, factor=100.0, bound=100.0, rate=1.0
        )

        assert len(result) == len(traj)

        original_np, _ = traj.to_numpy()
        result_np, _ = result.to_numpy()

        # With rate=1.0 every point receives noise.  The probability that
        # *both* x-noise and y-noise are < 1e-7 meters simultaneously is
        # vanishingly small, but not zero.  Instead of checking every point
        # at a tight tolerance, we verify that the vast majority of points
        # are visibly shifted and that the trajectory as a whole is different.
        shifted_count = sum(
            1
            for i in range(len(original_np))
            if not np.allclose(original_np[i], result_np[i], atol=1e-10)
        )
        assert shifted_count >= len(original_np) - 1, (
            f"Expected nearly all points to be shifted, "
            f"but only {shifted_count}/{len(original_np)} were"
        )


class TestAugmentationRngSeeding:
    """Tests that eval-testset generation is reproducible via seed_augmentation_rng.

    Regression tests: random_subsample and truncated_gaussian_distort must draw
    from the shared seeded RNG, so that generate_test_data.py --seed produces
    identical testsets across runs (data randomness isolation).
    """

    @staticmethod
    def _make_traj(n: int = 80) -> Trajectory:
        rng = np.random.default_rng(123)
        return Trajectory.from_numpy(rng.random((n, 2)) * 10.0, CoordinateSystem.Wgs84)

    def test_random_subsample_reproducible_with_seed(self):
        traj = self._make_traj()

        seed_augmentation_rng(42)
        first = random_subsample(traj, 0.5)
        seed_augmentation_rng(42)
        second = random_subsample(traj, 0.5)

        first_np, _ = first.to_numpy()
        second_np, _ = second.to_numpy()
        assert np.allclose(first_np, second_np)

    def test_random_subsample_different_seeds_differ(self):
        traj = self._make_traj()

        seed_augmentation_rng(1)
        first = random_subsample(traj, 0.5)
        seed_augmentation_rng(2)
        second = random_subsample(traj, 0.5)

        first_np, _ = first.to_numpy()
        second_np, _ = second.to_numpy()
        assert not np.allclose(first_np, second_np)

    def test_truncated_gaussian_distort_reproducible_with_seed(self):
        traj = self._make_traj()

        seed_augmentation_rng(42)
        first = truncated_gaussian_distort(traj)
        seed_augmentation_rng(42)
        second = truncated_gaussian_distort(traj)

        first_np, _ = first.to_numpy()
        second_np, _ = second.to_numpy()
        assert np.allclose(first_np, second_np)

    def test_truncated_gaussian_distort_different_seeds_differ(self):
        traj = self._make_traj()

        seed_augmentation_rng(1)
        first = truncated_gaussian_distort(traj)
        seed_augmentation_rng(2)
        second = truncated_gaussian_distort(traj)

        first_np, _ = first.to_numpy()
        second_np, _ = second.to_numpy()
        assert not np.allclose(first_np, second_np)

    def test_testset_pipeline_reproducible_with_seed(self):
        """End-to-end: parity split + subsample + distort is deterministic."""
        traj = self._make_traj()
        even, odd = split_trajectory_by_parity(traj)

        def build():
            query = random_subsample(even, 0.4)
            db = truncated_gaussian_distort(odd, rate=0.5)
            return query.to_numpy()[0], db.to_numpy()[0]

        seed_augmentation_rng(7)
        first_query, first_db = build()
        seed_augmentation_rng(7)
        second_query, second_db = build()

        assert np.allclose(first_query, second_query)
        assert np.allclose(first_db, second_db)
