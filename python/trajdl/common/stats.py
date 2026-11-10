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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class TrajCoordStandardStats:
    lng_mean: float
    lat_mean: float
    lng_std: float
    lat_std: float

    def get_mean(self) -> Tuple[float, float]:
        return self.lng_mean, self.lat_mean

    def get_std(self) -> Tuple[float, float]:
        return self.lng_std, self.lat_std

    @staticmethod
    def load(path: str | Path) -> "TrajCoordStandardStats":
        with open(path, "r") as f:
            configs = json.load(f)
            return TrajCoordStandardStats(
                lng_mean=configs["lng_mean"],
                lat_mean=configs["lat_mean"],
                lng_std=configs["lng_std"],
                lat_std=configs["lat_std"],
            )

    def __repr__(self) -> str:
        return f"TrajCoordStats(lng_mean={self.lng_mean}, lat_mean={self.lat_mean}, lng_std={self.lng_std}, lat_std={self.lat_std})"
