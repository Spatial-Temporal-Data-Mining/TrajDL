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

from typing import List, Tuple

import pandas as pd
from rtree import index
from tqdm.contrib import tenumerate

from .. import Coordinate, RectangleBoundary, SimpleGridSystem
from ..tokenizers import SimpleTokenizer
from .base import BaseGridSystem


def iter_a_simple_grid_system(gird: SimpleGridSystem):
    for y_idx in range(gird.num_y_grids):
        for x_idx in range(gird.num_x_grids):
            min_x = gird.min_x + x_idx * gird.step_x
            min_y = gird.min_y + y_idx * gird.step_y
            max_x = min(gird.max_x, min_x + gird.step_x)
            max_y = min(gird.max_y, min_y + gird.step_y)
            yield (
                # TODO: 需要注意这里的坐标系, fix later
                RectangleBoundary(
                    min_point=Coordinate.Wgs84(min_x, min_y),
                    max_point=Coordinate.Wgs84(max_x, max_y),
                ),
                gird.locate_in_grid(
                    Coordinate.Wgs84((max_x + min_x) / 2, (max_y + min_y) / 2)
                ),
            )


class HierarchyGridSystem(BaseGridSystem):
    def __init__(
        self,
        boundary: RectangleBoundary,
        steps: List[Tuple[float, float]],
    ):
        super().__init__(boundary=boundary)
        self.steps = steps
        self.all_grids = self.recursive_split_regions()
        self.rtree = self.build_rtree()

    def __len__(self) -> int:
        return len(self.all_grids)

    def __getstate__(self):
        state = self.__dict__.copy()
        # TODO: rtree使用pkl序列化会有一些问题，按理来说可以使用rtree自身的序列化，后续可以集成这个功能
        # 这里暂时先删除rtree，加载的时候进行重构
        if "rtree" in state:
            del state["rtree"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rtree = self.build_rtree()

    @property
    def grid_id_cols(self) -> List[str]:
        return [f"grid_id_level_{level}" for level in range(len(self.steps))]

    def recursive_split_regions(self) -> pd.DataFrame:
        all_grids = []

        def recursive_split_region(
            boundary: RectangleBoundary,
            parent_grid_ids: List[str],
            steps: List[Tuple[float, float]],
            depth: int = 0,
        ):
            if depth >= len(steps):
                parent_grid_ids.append(boundary.to_list())
                all_grids.append(parent_grid_ids)
                return None

            grid_system = SimpleGridSystem(
                boundary=boundary, step_x=steps[depth][0], step_y=steps[depth][1]
            )
            for sub_boundary, sub_grid_id in iter_a_simple_grid_system(grid_system):
                new_parent_grid_ids = parent_grid_ids.copy()
                new_parent_grid_ids.append(f"{depth}-{sub_grid_id}")
                recursive_split_region(
                    sub_boundary, new_parent_grid_ids, steps, depth + 1
                )

        recursive_split_region(self.boundary, [], self.steps)
        all_grids = pd.DataFrame(
            all_grids,
            columns=self.grid_id_cols + ["boundary"],
        )
        all_grids["grid_id"] = all_grids[self.grid_id_cols].apply(
            lambda x: "-".join(x), axis=1
        )
        return all_grids

    def build_rtree(self) -> index.Index:
        rtree = index.Index()
        for idx, line in tenumerate(
            self.all_grids.itertuples(),
            total=self.all_grids.shape[0],
            desc="construct rtree for hierarchy grid system",
        ):
            rtree.insert(idx, line.boundary)
        return rtree

    def locate_unsafe(self, x: float, y: float) -> str:
        indices = self.rtree.intersection((x, y, x, y))
        df = self.all_grids.loc[indices]

        for line in df.itertuples():
            # TODO: 需要考虑坐标系
            if RectangleBoundary.from_list(line.boundary).in_boundary(
                Coordinate.Wgs84(x, y)
            ):
                return line.grid_id
        return None

    def build_simple_tokenizer(self) -> SimpleTokenizer:
        vocab = dict(zip(self.all_grids["grid_id"], self.all_grids.index))
        return SimpleTokenizer.from_vocab(init_vocab=vocab)
