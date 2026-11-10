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
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from .. import Coordinate, RectangleBoundary


class BaseGridSystem(ABC):
    def __init__(self, boundary: RectangleBoundary):
        self._boundary = boundary

    @property
    def boundary(self) -> RectangleBoundary:
        return self._boundary

    @abstractmethod
    def __len__(self) -> int:
        """
        How many grids the system contains in total
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractmethod
    def locate_unsafe(self, x: float, y: float) -> str:
        """
        Determine which grid (x, y) belongs to; for performance, the check of
        whether x and y lie inside the boundary may be skipped
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def in_boundary(self, x: float, y: float) -> bool:
        """
        Whether (x, y) is inside the current boundary
        """
        # TODO: 坐标系需要和boundary对齐
        return self.boundary.in_boundary(Coordinate.Wgs84(x, y))

    def in_boundary_np(self, coords: np.ndarray) -> np.ndarray:
        return self.boundary.in_boundary_np(coords)

    def locate(self, x: float, y: float) -> str:
        if not self.in_boundary(x, y):
            raise ValueError("(x, y) is not in this region.")

        loc = self.locate_unsafe(x=x, y=y)
        if loc is None:
            raise ValueError("(x, y) is not in this region.")  # pragma: no cover
        return loc

    def save(self, path: Union[str, Path]) -> None:
        folder = os.path.split(path)[0]
        os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "BaseGridSystem":
        with open(path, "rb") as f:
            return pickle.load(f)
