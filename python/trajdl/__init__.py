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

from importlib.metadata import version

__version__ = version("trajdl")

# TODO(WIREUP): this is the partial export list. Each algorithm slice adds its
# symbols (convert_coords_to_locseq, generate_spatial_features, traj_to_image,
# ...), and the final wire-up restores the reference __init__.py verbatim.
from .trajdl_rust import (
    Coordinate,  # noqa: F401
    CoordinateSystem,  # noqa: F401
    GridCoord,  # noqa: F401
    LocSeq,  # noqa: F401
    LocSeqView,  # noqa: F401
    RectangleBoundary,  # noqa: F401
    SeqType,
    SimpleGridSystem,
    Trajectory,  # noqa: F401
    TrajectoryView,  # noqa: F401
    UniformDegreeGridSystem,
    collect_token_symbols_for_trajcl,
    convert_traj_to_locseq_with_coords,
    count_locations,
    count_locations_for_t2vec,
    distort,
    downsampling,
    mask_trajectory,
    random_subsample,
    rdp,
    seed_augmentation_rng,
    subset_trajectory,
    truncated_gaussian_distort,
    wgs84,
)
