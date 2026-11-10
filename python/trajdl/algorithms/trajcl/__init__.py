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

"""TrajCL algorithm implementation.

This module implements the TrajCL model for trajectory contrastive learning,
including:
- PositionalEncoding: Standard transformer positional encoding
- SpatialMSM: Spatial Micro-Spatial-Movement Model for local spatial features
- DualSTB: Dual-stream Spatio-Temporal Backbone
- MoCo: Momentum Contrast framework
- TrajCLLightning: Lightning module for training
"""

from .encoder import DualSTB, PositionalEncoding, SpatialMSM
from .moco import MoCo, Projector
from .model import TrajCL, TrajCLLightning
from .trajsimi import TrajSimiLightning, TrajSimiRegression

__all__ = [
    "PositionalEncoding",
    "SpatialMSM",
    "DualSTB",
    "MoCo",
    "Projector",
    "TrajCL",
    "TrajCLLightning",
    "TrajSimiLightning",
    "TrajSimiRegression",
]
