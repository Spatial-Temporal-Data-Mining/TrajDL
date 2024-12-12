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

from typing import Union

import numpy as np

from trajdl import trajdl_cpp
from trajdl.common.enum import ReturnASEnum
from trajdl.datasets.base import Trajectory


def trajectory_downsampling(
    traj: Trajectory, rate: float, return_as: str = "traj"
) -> Union[Trajectory, np.ndarray]:
    result_np = trajdl_cpp.downsampling(traj.seq, rate)
    if return_as == ReturnASEnum.NP.value:
        return result_np
    elif return_as == ReturnASEnum.TRAJ.value:
        return Trajectory(seq=result_np, entity_id=traj.entity_id)
    else:
        raise ValueError("`return_as` should be {'np', 'traj'}")


def trajectory_distortion(
    traj: Trajectory, rate: float, radius: float, return_as: str = "traj"
) -> Union[Trajectory, np.ndarray]:
    result_np = trajdl_cpp.distort(traj.seq, rate, radius)
    if return_as == ReturnASEnum.NP.value:
        return result_np
    elif return_as == ReturnASEnum.TRAJ.value:
        return Trajectory(seq=result_np, entity_id=traj.entity_id)
    else:
        raise ValueError("`return_as` should be {'np', 'traj'}")
