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
import pytest

from trajdl import trajdl_cpp
from trajdl.datasets import Trajectory
from trajdl.grid.base import SimpleGridSystem
from trajdl.utils.traj import trajectory_distortion, trajectory_downsampling

rng = np.random.default_rng(seed=42)


def test_downsampling():
    def test(traj_data: np.ndarray, rate: float):
        traj = Trajectory(traj_data)
        for return_as in ["np", "traj"]:
            output = trajectory_downsampling(traj, rate)
            output_data = output.seq if return_as == "traj" else output

            if len(traj) <= 2:
                assert np.allclose(traj_data, output_data, rtol=1e-6, atol=1e-6)
            else:
                assert np.allclose(traj_data[0], output_data[0], rtol=1e-6, atol=1e-6)
                assert np.allclose(traj_data[-1], output_data[-1], rtol=1e-6, atol=1e-6)
                traj_idx, output_idx = 0, 0
                while traj_idx < len(traj) and output_idx < len(output):
                    if np.allclose(
                        traj_data[traj_idx],
                        output_data[output_idx],
                        rtol=1e-6,
                        atol=1e-6,
                    ):
                        output_idx += 1
                    traj_idx += 1
                assert traj_idx == len(traj) and output_idx == len(output)
                if len(output) - 2 > 1500:
                    assert np.allclose(
                        rate, 1 - (len(output) - 2) / (len(traj) - 2), atol=5e-2
                    )

    with pytest.raises(Exception):
        test(rng.uniform(size=(2, 3)), 0.5)

    with pytest.raises(Exception):
        test(rng.uniform(size=(2, 0)), 0.5)

    for _ in range(100):
        test(
            rng.uniform(size=(rng.integers(0, 3000), 2)),
            rng.random(),
        )


def test_distort():
    a = Trajectory(rng.uniform(size=(10, 2)))
    trajectory_distortion(a, 0.5, 50.0)
    # TODO: finish this test case


def test_count_cells():
    ground_truth = {
        "186547": 5,
        "198782": 3,
        "191700": 3,
        "202651": 3,
        "200072": 3,
        "189774": 4,
        "192340": 4,
        "202652": 2,
        "184611": 2,
        "184610": 2,
        "195558": 2,
        "202004": 1,
        "202007": 1,
        "189128": 1,
        "202650": 1,
        "191702": 1,
        "187192": 1,
        "194270": 1,
        "197492": 1,
        "198137": 1,
        "190414": 1,
        "187838": 1,
        "194913": 1,
        "192342": 1,
        "185901": 1,
        "196202": 1,
        "192982": 1,
        "190417": 1,
        "190419": 1,
        "190411": 1,
        "195557": 1,
        "193627": 1,
        "199427": 1,
        "200717": 1,
    }

    t = np.array(
        [
            [-8.608833, 41.147586],
            [-8.608707, 41.147685],
            [-8.608473, 41.14773],
            [-8.608284, 41.148054],
            [-8.607708, 41.148999],
            [-8.60742, 41.149719],
            [-8.607411, 41.149872],
            [-8.607357, 41.149881],
            [-8.607375, 41.149917],
            [-8.607312, 41.149953],
            [-8.607123, 41.150223],
            [-8.606511, 41.15115],
            [-8.605881, 41.152302],
            [-8.605404, 41.153094],
            [-8.605404, 41.153256],
            [-8.60535, 41.153328],
            [-8.605278, 41.15331],
            [-8.605404, 41.153526],
            [-8.607411, 41.153589],
            [-8.609958, 41.153634],
            [-8.612091, 41.153769],
            [-8.612037, 41.154894],
            [-8.613036, 41.155173],
            [-8.613072, 41.155173],
            [-8.61381, 41.155407],
            [-8.616456, 41.155857],
            [-8.617761, 41.156064],
            [-8.617752, 41.156055],
            [-8.617752, 41.156055],
            [-8.618256, 41.156154],
            [-8.6202, 41.156595],
            [-8.620857, 41.157099],
            [-8.622378, 41.157981],
            [-8.623944, 41.15889],
            [-8.624655, 41.159529],
            [-8.624637, 41.159565],
            [-8.624673, 41.159565],
            [-8.624664, 41.159907],
            [-8.624718, 41.161347],
            [-8.624754, 41.162265],
            [-8.624781, 41.162886],
            [-8.624781, 41.162904],
            [-8.624781, 41.162895],
            [-8.624781, 41.163048],
            [-8.624826, 41.163921],
            [-8.624853, 41.163966],
            [-8.624889, 41.163939],
            [-8.624826, 41.164686],
            [-8.624952, 41.16627],
            [-8.62506, 41.166531],
            [-8.625168, 41.166972],
            [-8.625627, 41.166963],
            [-8.626059, 41.166999],
            [-8.626059, 41.16699],
            [-8.626617, 41.166648],
            [-8.627922, 41.166252],
        ]
    )

    boundary = trajdl_cpp.RectangleBoundary(
        min_x=-8.735152,
        min_y=40.953673,
        max_x=-8.156309,
        max_y=41.307945,
    )
    grid = SimpleGridSystem(
        boundary=boundary.to_web_mercator(), step_x=100.0, step_y=100.0
    )
    web_mercator_boundary = boundary.to_web_mercator()

    assert (
        trajdl_cpp.count_locations(
            [t],
            boundary,
            web_mercator_boundary,
            grid.step_x,
            grid.step_y,
            grid.num_x_grids,
        )
        == ground_truth
    )
