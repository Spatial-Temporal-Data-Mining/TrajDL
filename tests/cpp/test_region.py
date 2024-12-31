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

import numpy as np

from trajdl import trajdl_cpp
from trajdl.grid.base import SimpleGridSystem


def test_boundary_and_grid():
    boundary = trajdl_cpp.grid.RectangleBoundary(
        min_x=-8.735152,
        min_y=40.953673,
        max_x=-8.156309,
        max_y=41.307945,
    )
    web_boundary = boundary.to_web_mercator()
    assert np.allclose(web_boundary.min_x, -972392.6726418451)
    assert np.allclose(web_boundary.min_y, 5005510.841485837)
    assert np.allclose(web_boundary.max_x, -907956.1646325944)
    assert np.allclose(web_boundary.max_y, 5057870.140667772)

    grid = SimpleGridSystem(boundary=web_boundary, step_x=100.0, step_y=100.0)
    assert grid.num_x_grids == 645
    assert grid.num_y_grids == 524


def test_gps_2_loc():
    boundary = trajdl_cpp.grid.RectangleBoundary(
        min_x=-8.735152,
        min_y=40.953673,
        max_x=-8.156309,
        max_y=41.307945,
    )
    grid = SimpleGridSystem(
        boundary=boundary.to_web_mercator(), step_x=100.0, step_y=100.0
    )
    samples = [
        (-8.631144, 41.154489, "191035"),
        (-8.630838, 41.154489, "191036"),
        (-8.601057, 41.182182, "217514"),
        (-8.638497, 41.150511, "187157"),
        (-8.620794, 41.155884, "192337"),
    ]
    for lng, lat, gt in samples:
        web_point = trajdl_cpp.grid.convert_gps_to_webmercator(lng, lat)
        assert grid.locate(web_point.x, web_point.y) == gt
