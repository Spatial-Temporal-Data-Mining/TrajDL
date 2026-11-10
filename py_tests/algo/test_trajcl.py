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

"""Tests for generate_spatial_features function.

The expected results are pre-computed using the original TrajCL implementation
(see the upstream TrajCL repo, utils/traj.py).
"""

import numpy as np
import pytest
from trajdl import CoordinateSystem, Trajectory, generate_spatial_features

# ==============================================================================
# Expected results pre-computed from original TrajCL implementation
# ==============================================================================

# Test case 1: Straight line (0,0) -> (1,0) -> (2,0)
# Input: x_min=0, x_max=10, y_min=0, y_max=10, ref_length=1.0
STRAIGHT_LINE_INPUT = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
STRAIGHT_LINE_PARAMS = (0.0, 10.0, 0.0, 10.0, 1.0)
STRAIGHT_LINE_EXPECTED = [
    [0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 1.0, -1.0],
    [0.2, 0.0, 0.0, 0.0],
]

# Test case 2: 90-degree turn (0,0) -> (1,0) -> (1,1)
# Input: x_min=0, x_max=10, y_min=0, y_max=10, ref_length=1.0
TURN_90_INPUT = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
TURN_90_PARAMS = (0.0, 10.0, 0.0, 10.0, 1.0)
TURN_90_EXPECTED = [
    [0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 1.0, -0.5],
    [0.1, 0.1, 0.0, 0.0],
]

# Test case 3: U-turn (0,0) -> (1,0) -> (0,0)
# Input: x_min=0, x_max=10, y_min=0, y_max=10, ref_length=1.0
U_TURN_INPUT = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
U_TURN_PARAMS = (0.0, 10.0, 0.0, 10.0, 1.0)
U_TURN_EXPECTED = [
    [0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
]

# Test case 4: Random 10-point trajectory (seed=42)
# Input: x_min=0, x_max=1, y_min=0, y_max=1, ref_length=0.1
np.random.seed(42)
RANDOM_10_INPUT = np.random.rand(10, 2).tolist()
RANDOM_10_PARAMS = (0.0, 1.0, 0.0, 1.0, 0.1)
RANDOM_10_EXPECTED = [
    [0.3745401188473625, 0.9507143064099162, 0.0, 0.0],
    [0.7319939418114051, 0.5986584841970366, 6.140712456972725, 0.456155893690261],
    [0.15601864044243652, 0.15599452033620265, 7.216657008665586, 0.33504268450776986],
    [0.05808361216819946, 0.8661761457749352, 6.412408590934069, 0.36619687371248677],
    [0.6011150117432088, 0.7080725777960455, 6.01213236277093, -0.04468852893113673],
    [0.020584494295802447, 0.9699098521619943, 8.736323950697146, -0.10412247580433287],
    [0.8324426408004217, 0.21233911067827616, 8.808391386470808, 0.25314059283594903],
    [0.18182496720710062, 0.18340450985343382, 5.069499313330967, 0.37624720258619593],
    [0.3042422429595377, 0.5247564316322378, 3.144012917420658, -1.7311064916835028],
    [0.43194501864211576, 0.2912291401980419, 0.0, 0.0],
]

# Test case 5: Real Web Mercator coordinates (Porto dataset range)
# Input: x_min=-970000, x_max=-940000, y_min=5050000, y_max=5100000, ref_length=141.4
WEB_MERCATOR_INPUT = [
    [-960000.0, 5070000.0],
    [-959000.0, 5071000.0],
    [-958000.0, 5072000.0],
]
WEB_MERCATOR_PARAMS = (-970000.0, -940000.0, 5050000.0, 5100000.0, 141.4)
WEB_MERCATOR_EXPECTED = [
    [0.3333333333333333, 0.4, 0.0, 0.0],
    [0.36666666666666664, 0.42, 10.001510342101097, -1.0],
    [0.4, 0.44, 0.0, 0.0],
]


class TestGenerateSpatialFeatures:
    """Tests for generate_spatial_features function against original TrajCL."""

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        traj = Trajectory.from_numpy(np.array([]).reshape(0, 2), CoordinateSystem.Wgs84)
        features = generate_spatial_features(traj, 0, 10, 0, 10, 1.0)
        assert features == []

    def test_single_point_trajectory(self):
        """Test with single point trajectory."""
        traj = Trajectory.from_numpy(np.array([[5.0, 5.0]]), CoordinateSystem.Wgs84)
        features = generate_spatial_features(traj, 0, 10, 0, 10, 1.0)
        assert len(features) == 1
        assert features[0] == pytest.approx([0.5, 0.5, 0.0, 0.0])

    def test_two_point_trajectory(self):
        """Test with two point trajectory."""
        traj = Trajectory.from_numpy(
            np.array([[0.0, 0.0], [10.0, 10.0]]), CoordinateSystem.Wgs84
        )
        features = generate_spatial_features(traj, 0, 10, 0, 10, 1.0)
        assert len(features) == 2
        assert features[0] == pytest.approx([0.0, 0.0, 0.0, 0.0])
        assert features[1] == pytest.approx([1.0, 1.0, 0.0, 0.0])

    def test_straight_line_matches_original_trajcl(self):
        """Test straight line against original TrajCL implementation."""
        traj = Trajectory.from_numpy(
            np.array(STRAIGHT_LINE_INPUT), CoordinateSystem.Wgs84
        )
        x_min, x_max, y_min, y_max, ref_length = STRAIGHT_LINE_PARAMS
        features = generate_spatial_features(
            traj, x_min, x_max, y_min, y_max, ref_length
        )

        assert len(features) == len(STRAIGHT_LINE_EXPECTED)
        for i, (actual, expected) in enumerate(zip(features, STRAIGHT_LINE_EXPECTED)):
            assert actual == pytest.approx(expected, rel=1e-6), f"Point {i} mismatch"

    def test_90_degree_turn_matches_original_trajcl(self):
        """Test 90-degree turn against original TrajCL implementation."""
        traj = Trajectory.from_numpy(np.array(TURN_90_INPUT), CoordinateSystem.Wgs84)
        x_min, x_max, y_min, y_max, ref_length = TURN_90_PARAMS
        features = generate_spatial_features(
            traj, x_min, x_max, y_min, y_max, ref_length
        )

        assert len(features) == len(TURN_90_EXPECTED)
        for i, (actual, expected) in enumerate(zip(features, TURN_90_EXPECTED)):
            assert actual == pytest.approx(expected, rel=1e-6), f"Point {i} mismatch"

    def test_u_turn_matches_original_trajcl(self):
        """Test U-turn against original TrajCL implementation."""
        traj = Trajectory.from_numpy(np.array(U_TURN_INPUT), CoordinateSystem.Wgs84)
        x_min, x_max, y_min, y_max, ref_length = U_TURN_PARAMS
        features = generate_spatial_features(
            traj, x_min, x_max, y_min, y_max, ref_length
        )

        assert len(features) == len(U_TURN_EXPECTED)
        for i, (actual, expected) in enumerate(zip(features, U_TURN_EXPECTED)):
            assert actual == pytest.approx(expected, rel=1e-6), f"Point {i} mismatch"

    def test_random_10_points_matches_original_trajcl(self):
        """Test random 10-point trajectory against original TrajCL implementation."""
        traj = Trajectory.from_numpy(np.array(RANDOM_10_INPUT), CoordinateSystem.Wgs84)
        x_min, x_max, y_min, y_max, ref_length = RANDOM_10_PARAMS
        features = generate_spatial_features(
            traj, x_min, x_max, y_min, y_max, ref_length
        )

        assert len(features) == len(RANDOM_10_EXPECTED)
        for i, (actual, expected) in enumerate(zip(features, RANDOM_10_EXPECTED)):
            assert actual == pytest.approx(expected, rel=1e-6), f"Point {i} mismatch"

    def test_web_mercator_matches_original_trajcl(self):
        """Test real Web Mercator coordinates against original TrajCL implementation."""
        traj = Trajectory.from_numpy(
            np.array(WEB_MERCATOR_INPUT), CoordinateSystem.Wgs84
        )
        x_min, x_max, y_min, y_max, ref_length = WEB_MERCATOR_PARAMS
        features = generate_spatial_features(
            traj, x_min, x_max, y_min, y_max, ref_length
        )

        assert len(features) == len(WEB_MERCATOR_EXPECTED)
        for i, (actual, expected) in enumerate(zip(features, WEB_MERCATOR_EXPECTED)):
            assert actual == pytest.approx(expected, rel=1e-6), f"Point {i} mismatch"

    def test_feature_dimensions(self):
        """Test that each feature vector has exactly 4 dimensions."""
        traj = Trajectory.from_numpy(np.array(RANDOM_10_INPUT), CoordinateSystem.Wgs84)
        features = generate_spatial_features(traj, 0, 1, 0, 1, 0.1)
        for i, feat in enumerate(features):
            assert len(feat) == 4, f"Feature {i} has {len(feat)} dimensions, expected 4"

    def test_heading_change_values(self):
        """Test that heading_change values match expected semantics."""
        # Straight line: heading_change = -1
        traj = Trajectory.from_numpy(
            np.array(STRAIGHT_LINE_INPUT), CoordinateSystem.Wgs84
        )
        features = generate_spatial_features(traj, 0, 10, 0, 10, 1.0)
        assert features[1][3] == pytest.approx(-1.0, rel=1e-6)

        # 90-degree turn: heading_change = -0.5
        traj = Trajectory.from_numpy(np.array(TURN_90_INPUT), CoordinateSystem.Wgs84)
        features = generate_spatial_features(traj, 0, 10, 0, 10, 1.0)
        assert features[1][3] == pytest.approx(-0.5, rel=1e-6)

        # U-turn: heading_change = 0
        traj = Trajectory.from_numpy(np.array(U_TURN_INPUT), CoordinateSystem.Wgs84)
        features = generate_spatial_features(traj, 0, 10, 0, 10, 1.0)
        assert features[1][3] == pytest.approx(0.0, abs=1e-6)
