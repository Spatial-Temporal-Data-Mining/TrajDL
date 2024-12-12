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
from trajdl.datasets import Trajectory


def test_downsampling():
    seq = np.random.uniform(32, 56, size=(100, 2))
    traj = Trajectory(seq=seq)
    rate = 0.5
    new_seq = trajdl_cpp.downsampling(traj.seq, rate)


def test_distortion():
    radius = 50.0
    rate = 0.5
    seq1 = np.random.uniform(32, 56, size=(100, 2))
    seq2 = seq1.copy()
    traj = Trajectory(seq=seq2)
    trajdl_cpp.distort(traj.seq, rate, radius)
