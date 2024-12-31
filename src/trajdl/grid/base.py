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

import math
import os
import pickle
from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Tuple

import numpy as np

from .. import trajdl_cpp


class BaseGridSystem(ABC):
    def __init__(self, boundary: trajdl_cpp.grid.RectangleBoundary):
        self._boundary = boundary

    @property
    def boundary(self) -> trajdl_cpp.grid.RectangleBoundary:
        return self._boundary

    @abstractmethod
    def __len__(self) -> int:
        """
        一共有多少个网格
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractmethod
    def locate_unsafe(self, x: float, y: float) -> str:
        """
        判断x, y属于哪个网格，为了性能允许不check x，y是否在边界内
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def in_boundary(self, x: float, y: float) -> bool:
        """
        x, y是否在当前的边界内
        """
        return self.boundary.in_boundary(x, y)

    def in_boundary_np(self, coords: np.ndarray) -> np.ndarray:
        return self.boundary.in_boundary_np(coords)

    def locate(self, x: float, y: float) -> str:
        if not self.in_boundary(x, y):
            raise ValueError("(x, y) is not in this region.")

        loc = self.locate_unsafe(x=x, y=y)
        if loc is None:
            raise ValueError("(x, y) is not in this region.")  # pragma: no cover
        return loc

    def save(self, path: str) -> None:
        folder = os.path.split(path)[0]
        os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseGridSystem":
        with open(path, "rb") as f:
            return pickle.load(f)


class SimpleGridSystem(BaseGridSystem):
    """
    基础网格系统，一般x是经度，y是纬度
    """

    boundary: trajdl_cpp.grid.RectangleBoundary
    step_x: float
    step_y: float

    def __init__(
        self, boundary: trajdl_cpp.grid.RectangleBoundary, step_x: float, step_y: float
    ):
        super().__init__(boundary=boundary)
        self._step_x = step_x
        self._step_y = step_y

        self._min_x = self.boundary.min_x
        self._min_y = self.boundary.min_y
        self._max_x = self.boundary.max_x
        self._max_y = self.boundary.max_y

        self._num_y_grids = math.ceil((self._max_y - self._min_y) / self.step_y)
        self._num_x_grids = math.ceil((self._max_x - self._min_x) / self.step_x)

    @property
    def step_x(self) -> float:
        return self._step_x

    @property
    def step_y(self) -> float:
        return self._step_y

    @property
    def num_x_grids(self) -> int:
        return self._num_x_grids

    @property
    def num_y_grids(self) -> int:
        return self._num_y_grids

    def __len__(self) -> int:
        return self._num_x_grids * self._num_y_grids

    def __repr__(self) -> str:
        return f"SimpleGridSystem(boundary={self.boundary}, step_x={self.step_x}, step_y={self.step_y})"  # pragma: no cover

    def __iter__(
        self,
    ) -> Generator[Tuple[trajdl_cpp.grid.RectangleBoundary, str], None, None]:
        for y_idx in range(self._num_y_grids):
            for x_idx in range(self._num_x_grids):
                min_x = self._min_x + x_idx * self.step_x
                min_y = self._min_y + y_idx * self.step_y
                max_x = min(self._max_x, min_x + self.step_x)
                max_y = min(self._max_y, min_y + self.step_y)
                yield (
                    trajdl_cpp.grid.RectangleBoundary(
                        min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y
                    ),
                    self.locate_unsafe(x=(max_x + min_x) / 2, y=(max_y + min_y) / 2),
                )

    def locate_by_grid_coordinate(self, grid_x: int, grid_y: int) -> str:
        """
        将网格坐标转换为位置id

        """
        return trajdl_cpp.grid.locate_by_grid_coordinate(
            grid_x, grid_y, self._num_x_grids
        )

    def locate_unsafe(self, x: float, y: float) -> str:
        """
        使用向下取整，因此所有网格都是左侧和下侧的边界是包含的，右侧和上侧是非包含
        """

        return trajdl_cpp.grid.locate_in_grid(
            x, y, self.boundary, self.step_x, self.step_y, self._num_x_grids
        )

    def locate_unsafe_np(
        self, coords: np.ndarray, unk_loc: Optional[str] = None
    ) -> List[str]:
        return trajdl_cpp.grid.locate_in_grid_np(
            coords,
            self.boundary,
            self.step_x,
            self.step_y,
            self._num_x_grids,
            unk_loc,
        )

    def in_boundary_by_grid_coordinate(self, grid_x: int, grid_y: int) -> bool:
        return 0 <= grid_x < self._num_x_grids and 0 <= grid_y < self._num_y_grids

    def to_grid_coordinate_unsafe(self, loc: str) -> Tuple[int, int]:
        return trajdl_cpp.grid.reverse_locate_in_grid(loc, self._num_x_grids).to_tuple()

    def to_grid_coordinate(self, loc: str) -> Tuple[int, int]:
        try:
            loc_id = int(loc)
        except Exception:
            raise ValueError(f"The given loc {loc} does not belong this grid.")

        if loc_id < 0 or loc_id >= len(self):
            raise ValueError(f"The given loc {loc} does not belong this grid.")
        return self.to_grid_coordinate_unsafe(loc=loc)

    def get_centroid_of_grid(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        给定网格坐标，获取网格中心点的原始坐标
        """
        p = trajdl_cpp.grid.grid_coord_to_centroid_point(
            trajdl_cpp.grid.GridCoord(
                grid_x=grid_x,
                grid_y=grid_y,
            ),
            self._min_x,
            self._min_y,
            self.step_x,
            self.step_y,
        )
        return p.x, p.y
