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

from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from sklearn.neighbors import KDTree

from .. import trajdl_cpp
from ..common.enum import TokenEnum
from ..datasets.base import Trajectory
from ..grid.base import SimpleGridSystem
from .abstract import AbstractTrajTokenizer


class T2VECTokenizer(AbstractTrajTokenizer):
    """
    t2vec的tokenizer，专门处理轨迹序列
    """

    def __init__(
        self,
        grid: SimpleGridSystem,
        gps_boundary: trajdl_cpp.RectangleBoundary,
        vocab: Dict[str, int],
        with_kd_tree: bool = False,
        hot_locations: List[str] = None,
        kdtree: KDTree = None,
    ):
        """
        Parameters
        ----------
        grid: SimpleGridSystem
            网格系统

        gps_boundary: trajdl_cpp.RectangleBoundary
            GPS边界

        vocab: Dict[str, int]
            key是location，value是location idx

        with_kd_tree: bool, optional
            是否要使用KD树，当tokenization得到的location不在hot cells里面的时候，可以通过KD树寻找距离最近的hot cell近似
            如果不使用KD树，会用UNK代替。默认值是False，不使用KD树。

        hot_locations: List[str], optional
            热点location的list，里面每一项都是一个location

        kdtree: KDTree
            如果使用KD树，这个是KD树的实例

        """
        super(T2VECTokenizer, self).__init__(vocab)
        self.grid = grid
        self.boundary = gps_boundary
        self.with_kd_tree = with_kd_tree
        self.hot_locations = hot_locations
        self.kdtree = kdtree
        self.unknown_loc_tag = TokenEnum.UNK_TOKEN.value
        self.nearest_hot_map = {}

    @classmethod
    def construct_vocab(
        cls,
        grid: SimpleGridSystem,
        gps_boundary: trajdl_cpp.RectangleBoundary,
        trajectories: Iterable[Trajectory],
        max_vocab_size: int,
        min_freq: int,
    ) -> Dict[str, int]:
        """
        Parameters
        ----------
        grid: SimpleGridSystem
            这个是基于web mercator坐标系的网格系统

        gps_boundary: trajdl_cpp.RectangleBoundary
            这个是基于WGS84坐标系的boundary

        """
        web_mercator_boundary = gps_boundary.to_web_mercator()

        all_cell_counter = trajdl_cpp.count_locations(
            [traj.seq for traj in trajectories],
            gps_boundary,
            web_mercator_boundary,
            grid.step_x,
            grid.step_y,
            grid.num_x_grids,
        )

        # filter out all hot locations
        max_num_hotlocs = min(max_vocab_size, len(all_cell_counter))

        # 取出前max_num_hotlocs个热点token
        top_loc_count = sorted(
            ((loc, cnt) for loc, cnt in all_cell_counter.items() if cnt >= min_freq),
            key=lambda x: x[1],
            reverse=True,
        )[:max_num_hotlocs]

        locations = (loc for loc, _ in top_loc_count)

        vocab = {loc: idx for idx, loc in enumerate(locations)}
        for token in (
            TokenEnum.BOS_TOKEN.value,
            TokenEnum.EOS_TOKEN.value,
            TokenEnum.UNK_TOKEN.value,
            TokenEnum.MASK_TOKEN.value,
            TokenEnum.PAD_TOKEN.value,
        ):
            vocab[token] = len(vocab)

        return vocab

    @classmethod
    def build(
        cls,
        grid: SimpleGridSystem,
        boundary: trajdl_cpp.RectangleBoundary,
        trajectories: Iterable[Trajectory],
        max_vocab_size: int,
        min_freq: int,
        with_kd_tree: bool = False,
    ) -> "T2VECTokenizer":

        vocab = cls.construct_vocab(
            grid=grid,
            gps_boundary=boundary,
            trajectories=trajectories,
            max_vocab_size=max_vocab_size,
            min_freq=min_freq,
        )

        hot_locs = None
        hotcell_kdtree = None
        if with_kd_tree:
            SPECIAL_TOKENS = TokenEnum.values()
            hot_locs = [loc for loc in vocab if loc not in SPECIAL_TOKENS]
            web_mercator_coords = [
                grid.get_centroid_of_grid(*grid.to_grid_coordinate(loc))
                for loc in hot_locs
            ]

            hotcell_kdtree = KDTree(web_mercator_coords, leaf_size=2)
        return T2VECTokenizer(
            grid=grid,
            gps_boundary=boundary,
            vocab=vocab,
            with_kd_tree=with_kd_tree,
            hot_locations=hot_locs,
            kdtree=hotcell_kdtree,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        if "nearest_hot_map" in state:
            del state["nearest_hot_map"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.nearest_hot_map = {}

    def _nearest_hot_loc(self, loc: str) -> str:
        """
        find nearest hot location
        """
        if loc in self.nearest_hot_map:
            return self.nearest_hot_map[loc]

        web_point = self.grid.get_centroid_of_grid(*self.grid.to_grid_coordinate(loc))
        _, idxs = self.kdtree.query([web_point], 1)
        nearest_hot_loc = self.hot_locations[idxs[0][0]]
        self.nearest_hot_map[loc] = nearest_hot_loc
        return nearest_hot_loc

    def k_nearest_hot_loc(
        self, loc_list: List[str], k: int
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        search k-nearest neighbors for given loc_list
        """
        points = [
            self.grid.get_centroid_of_grid(*self.grid.to_grid_coordinate(loc))
            for loc in loc_list
        ]
        # dists, idxs shape: (len(loc_list), k)
        dists, idxs = self.kdtree.query(points, k)
        return dists, [
            [self.hot_locations[idx] for idx in line_indices] for line_indices in idxs
        ]

    def traj_to_loc_seq(
        self, traj: Union[Trajectory, np.ndarray], add_start_end_token: bool
    ) -> List[str]:
        """
        Transform a trajectory into a location sequence
        """
        traj_np = None
        if isinstance(traj, Trajectory):
            traj_np = traj.seq
        elif isinstance(traj, np.ndarray):
            traj_np = traj
        else:
            raise ValueError("`traj` should be a Trajectory or a numpy.ndarray.")

        loc_seq = trajdl_cpp.convert_points_to_seq(
            traj_np,
            self.boundary,
            self.boundary.to_web_mercator(),
            self.grid.step_x,
            self.grid.step_y,
            self.grid.num_x_grids,
            self.unknown_loc_tag,
            add_start_end_token,
            TokenEnum.BOS_TOKEN.value,
            TokenEnum.EOS_TOKEN.value,
        )
        if self.with_kd_tree:
            return [
                loc if loc in self.vocab else self._nearest_hot_loc(loc)
                for loc in loc_seq
            ]
        else:
            return [
                loc if loc in self.vocab else TokenEnum.UNK_TOKEN.value
                for loc in loc_seq
            ]

    def loc2idx(self, loc: str) -> int:
        return self.vocab[loc]

    def tokenize_traj(
        self,
        traj: Union[Trajectory, np.ndarray],
        add_start_end_token: bool = False,
        return_as: str = "py",
    ) -> List[int]:
        """
        transform trajectory into location sequence
        """

        inputs = None
        if isinstance(traj, Trajectory):
            inputs = traj.seq
        elif isinstance(traj, np.ndarray):
            inputs = Trajectory.check_seq(traj)
        else:
            raise ValueError("`traj` should be a Trajectory or a numpy.ndarray.")

        loc_seq = self.traj_to_loc_seq(inputs, add_start_end_token=add_start_end_token)
        r = self._tokenize_loc_seq_impl(loc_seq=loc_seq)

        if return_as == "py":
            return r
        elif return_as == "np":
            return np.array(r, dtype=np.int64)
        elif return_as == "pt":
            return torch.LongTensor(r)
        else:
            raise ValueError("`return_as` could only be {'py', 'np', 'pt'}")

    def _tokenize_loc_seq_impl(self, loc_seq: Iterable[str]) -> List[int]:
        if not self.with_kd_tree:
            return [self.vocab.get(loc, self.unk) for loc in loc_seq]
        else:
            return [self.loc2idx(loc) for loc in loc_seq]
