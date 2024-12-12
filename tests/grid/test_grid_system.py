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

import os

import numpy as np
import pytest

from trajdl import trajdl_cpp
from trajdl.grid import HierarchyGridSystem, SimpleGridSystem

from ..conftest import FOLDER


@pytest.fixture(scope="session", autouse=True)
def test_boundary() -> trajdl_cpp.RectangleBoundary:
    min_x = -8.690261
    min_y = 41.140092
    max_x = -8.549155
    max_y = 41.185969

    boundary = trajdl_cpp.RectangleBoundary(min_x, min_y, max_x, max_y)
    assert np.isclose(boundary.min_x, min_x)
    assert np.isclose(boundary.min_y, min_y)
    assert np.isclose(boundary.max_x, max_x)
    assert np.isclose(boundary.max_y, max_y)

    assert boundary.to_tuple() == (min_x, min_y, max_x, max_y)
    assert (
        trajdl_cpp.RectangleBoundary.from_tuple((min_x, min_y, max_x, max_y)).to_tuple()
        == boundary.to_tuple()
    )

    with pytest.raises(ValueError, match="Length of tuple should be 4."):
        trajdl_cpp.RectangleBoundary.from_tuple((1, 2, 3))

    for _ in range(100):
        length = 100
        x_arr = np.random.uniform(low=min_x - 0.1, high=max_x + 0.1, size=(length, 1))
        y_arr = np.random.uniform(low=min_y - 0.1, high=max_y + 0.1, size=(length, 1))
        seq = np.concatenate([x_arr, y_arr], axis=1)

        r2 = boundary.in_boundary_np(seq)

        for x, y, r in zip(
            x_arr.squeeze(axis=1).tolist(), y_arr.squeeze(axis=1).tolist(), r2.tolist()
        ):
            assert boundary.in_boundary(x, y) == r

    return boundary


def test_simple_grid_system(traj_samples, test_boundary):
    step_x = 0.1 / 111.320 / 0.99974
    step_y = 0.1 / 110.574
    grid = SimpleGridSystem(
        test_boundary,
        step_x=step_x,
        step_y=step_y,
    )
    assert len(grid) == 158 * 51
    assert grid._num_x_grids == 158
    assert grid._num_y_grids == 51
    assert grid.locate(-8.690261, 41.140092) == "0"
    assert not grid.in_boundary(-8.549155, 41.185969)
    assert grid.locate(-8.549156, 41.185968) == str(158 * 51 - 1)

    assert grid.locate_by_grid_coordinate(grid_x=157, grid_y=50) == "8057"
    assert grid.in_boundary_by_grid_coordinate(grid_x=157, grid_y=50)
    assert not grid.in_boundary_by_grid_coordinate(grid_x=157, grid_y=51)

    for sub_bound, grid_id in grid:
        assert grid.in_boundary(sub_bound.min_x, sub_bound.min_y)
        assert grid.in_boundary(sub_bound.min_x, sub_bound.max_y - 1e-7)
        assert grid.in_boundary(sub_bound.max_x - 1e-7, sub_bound.max_y - 1e-7)
        assert grid.in_boundary(sub_bound.max_x - 1e-7, sub_bound.min_y)

    with pytest.raises(ValueError):
        grid.to_grid_coordinate(str(158 * 51))

    assert grid.to_grid_coordinate(str(158 * 51 - 1)) == (157, 50)
    assert grid.to_grid_coordinate(str(158 * 50)) == (0, 50)

    with pytest.raises(ValueError):
        grid.to_grid_coordinate("abc")

    traj_np = np.array(
        [[-8.690261, 41.140092], [-8.690260, 41.140093], [-8.690262, 41.140091]]
    )
    assert (grid.in_boundary_np(traj_np) == [True, True, False]).all()

    for traj in traj_samples:
        traj_np = traj.seq
        in_bound_list = [grid.in_boundary(x=lng, y=lat) for lng, lat in traj_np]
        in_bound_np = grid.in_boundary_np(traj_np)
        assert (in_bound_list == in_bound_np).all()

        locate_list = [grid.locate_unsafe(x=lng, y=lat) for lng, lat in traj_np]
        locate_np = grid.locate_unsafe_np(traj_np)
        for c, a, b in zip(traj_np, locate_list, locate_np):
            if a != b:
                print(c, a, b)
        assert locate_list == locate_np


def test_hier_grid_system(test_hierarchy: HierarchyGridSystem):
    path = os.path.join(FOLDER, "hier", "h_grid_system.pkl")
    test_hierarchy.save(path)

    def test_h_grid(h_grid: HierarchyGridSystem):
        assert len(h_grid) == 10089

        with pytest.raises(ValueError):
            h_grid.locate(-8.549155, 41.185969)

        assert h_grid.locate_unsafe(-8.549155, 41.185969) is None

        with pytest.raises(ValueError):
            h_grid.locate(-8.549154, 41.185969)

        assert h_grid.locate_unsafe(-8.549154, 41.185969) is None

        assert h_grid.locate(-8.690261, 41.140092) == "0-0-1-0-2-0"
        assert h_grid.locate_unsafe(-8.690261, 41.140092) == "0-0-1-0-2-0"
        assert h_grid.locate(-8.549461, 41.185691999999996) == "0-11-1-5-2-9"
        assert h_grid.locate_unsafe(-8.549461, 41.185691999999996) == "0-11-1-5-2-9"

    for h_grid in [test_hierarchy, HierarchyGridSystem.load(path)]:
        test_h_grid(h_grid=h_grid)
