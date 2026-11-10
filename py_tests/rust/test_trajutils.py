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
from trajdl import CoordinateSystem, Trajectory, distort, downsampling


def test_downsampling():
    traj = Trajectory.from_numpy(
        np.random.uniform(32, 56, size=(100, 2)), CoordinateSystem.Wgs84
    )
    rate = 0.5
    downsampling(traj, rate, keep_endpoints=True, allow_empty=False)


def test_distortion():
    radius = 50.0
    rate = 0.5
    traj = Trajectory.from_numpy(
        np.random.uniform(32, 56, size=(100, 2)), CoordinateSystem.Wgs84
    )
    distort(traj, rate, radius)
