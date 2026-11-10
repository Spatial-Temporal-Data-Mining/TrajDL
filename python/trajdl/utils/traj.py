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

from typing import Tuple, Union

import numpy as np

from .. import CoordinateSystem, Trajectory, distort, downsampling
from ..common.enum import ReturnASEnum
from ..common.stats import TrajCoordStandardStats


def trajectory_downsampling(
    traj: Trajectory, rate: float, return_as: str = "traj"
) -> Union[Trajectory, Tuple[np.ndarray, CoordinateSystem]]:
    result_seq = downsampling(traj, rate, keep_endpoints=True, allow_empty=False)

    if return_as == ReturnASEnum.NP.value:
        return result_seq.to_numpy()
    elif return_as == ReturnASEnum.TRAJ.value:
        result_seq.entity_id = traj.entity_id
        return result_seq
    else:
        raise ValueError("`return_as` should be {'np', 'traj'}")


def trajectory_distortion(
    traj: Trajectory, rate: float, radius: float, return_as: str = "traj"
) -> Union[Trajectory, Tuple[np.ndarray, CoordinateSystem]]:
    result_seq = distort(traj, rate, radius)

    if return_as == ReturnASEnum.NP.value:
        return result_seq.to_numpy()
    elif return_as == ReturnASEnum.TRAJ.value:
        result_seq.entity_id = traj.entity_id
        return result_seq
    else:
        raise ValueError("`return_as` should be {'np', 'traj'}")


def traj_standardize(
    traj: Trajectory, traj_coord_stats: TrajCoordStandardStats
) -> np.ndarray:
    arr, _ = traj.to_numpy()
    return (arr - traj_coord_stats.get_mean()) / traj_coord_stats.get_std()
