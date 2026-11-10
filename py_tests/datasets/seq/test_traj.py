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

import re

import numpy as np
import pytest
from trajdl import CoordinateSystem, Trajectory


class TestTrajectory:
    rng = np.random.default_rng(seed=42)

    @pytest.mark.parametrize("shape", [(3, 3), (3, 4), (2, 0)])
    def test_from_numpy_with_invalid_shape(self, shape):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"DataConvertionError: shape of array must be (n, 2), instead of {list(shape)}"
            ),
        ):
            Trajectory.from_numpy(self.rng.random(size=shape), CoordinateSystem.Wgs84)

    def test_from_numpy_with_invalid_dtype(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Expected a numpy array with dtype=float64, but got dtype='float32'"
            ),
        ):
            Trajectory.from_numpy(
                self.rng.random(size=(3, 2)).astype(np.float32), CoordinateSystem.Wgs84
            )

    def test_from_list_with_invalid_pylist(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Failed to convert element at index 0 to [f64; 2]. Expected a sequence of two numbers."
            ),
        ):
            Trajectory.from_list([1.0, 2.0, 3.0], CoordinateSystem.Wgs84)

    def test_from_list_with_invalid_tuple(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "argument 'seq': 'tuple' object cannot be converted to 'PyList'"
            ),
        ):
            Trajectory.from_list((1, 2, 3), CoordinateSystem.Wgs84)

    def test_from_numpy_accepts_valid_numpy(self):
        traj = Trajectory.from_numpy(
            self.rng.random(size=(10, 2)), CoordinateSystem.Wgs84, entity_id="a"
        )
        assert len(traj) == 10
        assert isinstance(traj.seq, list)
        arr, _ = traj.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert len(traj[0]) == 2
        assert traj.entity_id == "a"

    def test_from_list_accepts_valid_list(self):
        traj = Trajectory.from_list(
            self.rng.random(size=(10, 2)).tolist(),
            CoordinateSystem.Wgs84,
            entity_id="a",
        )
        assert len(traj) == 10
        assert isinstance(traj.seq, list)
        arr, _ = traj.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert len(traj[0]) == 2
        assert traj.entity_id == "a"

    def test_from_numpy_with_empty_array(self):
        assert len(Trajectory.from_list([], CoordinateSystem.Wgs84).seq) == 0

    def test_from_list_with_empty_list(self):
        arr, _ = Trajectory.from_list([], CoordinateSystem.Wgs84).to_numpy()
        assert arr.shape == (0, 2)

    def test_from_numpy_contiguous_array(self):
        """
        测试快速路径：使用内存连续的 C-style numpy 数组。
        这应该会触发 as_slice() 路径。
        """
        # 1. 准备数据
        data = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], dtype=np.float64)

        # 确认我们正在测试正确的路径
        assert (
            data.flags["C_CONTIGUOUS"] is True
        ), "测试前提失败：数组不是C-contiguous的"

        # 2. 调用待测函数
        traj = Trajectory.from_numpy(
            data, CoordinateSystem.Wgs84, entity_id="contiguous_test"
        )

        # 3. 验证结果
        expected_list = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]
        assert traj.seq == expected_list
        assert traj.entity_id == "contiguous_test"

    def test_from_numpy_non_contiguous_array(self):
        """
        测试备用路径：使用非内存连续的 numpy 数组。
        这应该会触发逐行拷贝的路径。
        """
        # 1. 准备数据：通过步长切片创建一个非连续视图
        base_array = np.arange(12, dtype=np.float64).reshape((6, 2))
        # base_array 是 [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]]

        data = base_array[::2]  # 取第0, 2, 4行
        # data 现在是 [[0,1], [4,5], [8,9]], 但它的元素在内存中是跳跃的

        # 确认我们正在测试正确的路径
        assert data.flags["C_CONTIGUOUS"] is False, "测试前提失败：数组是C-contiguous的"

        # 2. 调用待测函数
        traj = Trajectory.from_numpy(
            data, CoordinateSystem.Wgs84, entity_id="non_contiguous_test"
        )

        # 3. 验证结果
        expected_list = [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]]
        assert traj.seq == expected_list
        assert traj.entity_id == "non_contiguous_test"


class TestTrajectoryCoordinateSystem:
    """测试 Trajectory 的坐标系功能"""

    def test_default_coord_system(self):
        """测试默认坐标系为 Wgs84"""
        traj = Trajectory.from_list(
            [[116.0, 39.0], [116.1, 39.1]], CoordinateSystem.Wgs84
        )
        assert traj.coord_system == CoordinateSystem.Wgs84

    def test_from_list_with_coord_system(self):
        """测试 from_list 指定坐标系"""
        # Wgs84 (显式指定)
        traj_wgs = Trajectory.from_list(
            [[116.0, 39.0], [116.1, 39.1]],
            coord_system=CoordinateSystem.Wgs84,
        )
        assert traj_wgs.coord_system == CoordinateSystem.Wgs84

        # WebMercator
        traj_merc = Trajectory.from_list(
            [[12913060.0, 4721671.0], [12914060.0, 4721771.0]],
            coord_system=CoordinateSystem.WebMercator,
        )
        assert traj_merc.coord_system == CoordinateSystem.WebMercator

    def test_from_numpy_with_coord_system(self):
        """测试 from_numpy 指定坐标系"""
        arr_wgs = np.array([[116.0, 39.0], [116.1, 39.1]], dtype=np.float64)
        traj_wgs = Trajectory.from_numpy(arr_wgs, coord_system=CoordinateSystem.Wgs84)
        assert traj_wgs.coord_system == CoordinateSystem.Wgs84

        arr_merc = np.array(
            [[12913060.0, 4721671.0], [12914060.0, 4721771.0]], dtype=np.float64
        )
        traj_merc = Trajectory.from_numpy(
            arr_merc, coord_system=CoordinateSystem.WebMercator
        )
        assert traj_merc.coord_system == CoordinateSystem.WebMercator

    def test_coord_system_is_readonly(self):
        """测试坐标系是只读的（setter 已移除）"""
        traj = Trajectory.from_list([[116.0, 39.0]], CoordinateSystem.Wgs84)
        assert traj.coord_system == CoordinateSystem.Wgs84

        with pytest.raises(AttributeError):
            traj.coord_system = CoordinateSystem.WebMercator

    def test_to_web_mercator_from_wgs84(self):
        """测试 Wgs84 转 WebMercator"""
        traj_wgs = Trajectory.from_list(
            [[116.0, 39.0], [116.1, 39.1]], CoordinateSystem.Wgs84
        )
        traj_merc = traj_wgs.to_web_mercator()

        assert traj_merc.coord_system == CoordinateSystem.WebMercator
        # 原轨迹不变
        assert traj_wgs.coord_system == CoordinateSystem.Wgs84
        # 坐标已转换
        assert traj_wgs[0] != traj_merc[0]

    def test_to_web_mercator_already_mercator(self):
        """测试已经是 WebMercator 时调用 to_web_mercator"""
        traj_merc = Trajectory.from_list(
            [[12913060.0, 4721671.0], [12914060.0, 4721771.0]],
            coord_system=CoordinateSystem.WebMercator,
        )
        traj_converted = traj_merc.to_web_mercator()

        assert traj_converted.coord_system == CoordinateSystem.WebMercator
        # 坐标不变
        np.testing.assert_array_almost_equal(traj_merc[0], traj_converted[0])

    def test_to_wgs84_from_mercator(self):
        """测试 WebMercator 转 Wgs84"""
        traj_merc = Trajectory.from_list(
            [[12913060.0, 4721671.0], [12914060.0, 4721771.0]],
            coord_system=CoordinateSystem.WebMercator,
        )
        traj_wgs = traj_merc.to_wgs84()

        assert traj_wgs.coord_system == CoordinateSystem.Wgs84
        # 原轨迹不变
        assert traj_merc.coord_system == CoordinateSystem.WebMercator

    def test_to_wgs84_already_wgs84(self):
        """测试已经是 Wgs84 时调用 to_wgs84"""
        traj_wgs = Trajectory.from_list(
            [[116.0, 39.0], [116.1, 39.1]], CoordinateSystem.Wgs84
        )
        traj_converted = traj_wgs.to_wgs84()

        assert traj_converted.coord_system == CoordinateSystem.Wgs84
        np.testing.assert_array_almost_equal(traj_wgs[0], traj_converted[0])

    def test_round_trip_conversion(self):
        """测试往返转换精度"""
        original = Trajectory.from_list(
            [[116.0, 39.0], [116.1, 39.1], [116.2, 39.2]], CoordinateSystem.Wgs84
        )

        # Wgs84 -> WebMercator -> Wgs84
        round_trip = original.to_web_mercator().to_wgs84()

        assert round_trip.coord_system == CoordinateSystem.Wgs84
        for i in range(len(original)):
            np.testing.assert_array_almost_equal(original[i], round_trip[i], decimal=10)

    def test_round_trip_from_mercator(self):
        """测试从 WebMercator 开始的往返转换"""
        original = Trajectory.from_list(
            [[12913060.0, 4721671.0], [12914060.0, 4721771.0]],
            coord_system=CoordinateSystem.WebMercator,
        )

        # WebMercator -> Wgs84 -> WebMercator
        round_trip = original.to_wgs84().to_web_mercator()

        assert round_trip.coord_system == CoordinateSystem.WebMercator
        for i in range(len(original)):
            np.testing.assert_array_almost_equal(original[i], round_trip[i], decimal=6)
