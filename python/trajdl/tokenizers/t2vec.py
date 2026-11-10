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

import itertools
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from sklearn.neighbors import KDTree

from .. import (
    RectangleBoundary,
    SimpleGridSystem,
    Trajectory,
    TrajectoryView,
    convert_coords_to_locseq,
    count_locations_for_t2vec,
)
from ..common.enum import TokenEnum
from .abstract import AbstractTrajTokenizer


class T2VECTokenizer(AbstractTrajTokenizer):
    """A tokenizer that implements the T2Vec methodology.

    This tokenizer discretizes continuous GPS trajectories into grid cells,
    builds a vocabulary from the most frequent ("hot") cells, and tokenizes
    trajectories accordingly. It can optionally use a KD-Tree to map
    infrequent (non-hot) cells to their nearest hot-cell neighbor.
    """

    def __init__(
        self,
        web_grid: SimpleGridSystem,
        gps_boundary: RectangleBoundary,
        vocab: Dict[str, int],
        with_kd_tree: bool = False,
        hot_locations: List[str] | None = None,
        kdtree: KDTree | None = None,
    ):
        """Initializes the T2VECTokenizer.

        Parameters
        ----------
        web_grid : SimpleGridSystem
            The grid system used for discretizing coordinates.
        gps_boundary : RectangleBoundary
            The geographical boundary for valid GPS points.
        vocab : Dict[str, int]
            The vocabulary mapping hot location strings to IDs.
        with_kd_tree : bool, optional
            If True, use a KD-Tree to find the nearest hot cell for out-of-vocabulary
            cells. Otherwise, they are mapped to the UNK token. Defaults to False.
        hot_locations : List[str], optional
            A list of hot location strings, required if `with_kd_tree` is True.
        kdtree : KDTree, optional
            A pre-built KD-Tree of hot location coordinates, required if
            `with_kd_tree` is True.
        """
        super().__init__(vocab=vocab)
        self.web_grid = web_grid
        self.boundary = gps_boundary
        self.with_kd_tree = with_kd_tree
        self.hot_locations = hot_locations
        self.kdtree = kdtree
        self.unknown_loc_tag = TokenEnum.UNK_TOKEN.value
        self.nearest_hot_map = {}

    @classmethod
    def construct_vocab(
        cls,
        web_grid: SimpleGridSystem,
        gps_boundary: RectangleBoundary,
        trajectories: Iterable[Trajectory],
        max_vocab_size: int,
        min_freq: int,
    ) -> Dict[str, int]:
        """Constructs the T2Vec vocabulary from raw trajectories.

        This method performs the following steps:
        1. Discretizes all points from the input trajectories into grid cells
           using the provided grid system and boundary.
        2. Counts the frequency of each unique grid cell.
        3. Filters out cells with a frequency below `min_freq`.
        4. Selects the top `max_vocab_size` most frequent cells (the "hot" cells).
           Ties in frequency are broken by sorting cell identifiers alphabetically.
        5. Assigns a unique integer ID to each hot cell, starting from 0.
        6. Appends the standard special tokens (BOS, EOS, UNK, etc.) to the
           vocabulary with subsequent IDs.

        Parameters
        ----------
        web_grid : SimpleGridSystem
            The grid system used to convert GPS coordinates to cell identifiers.
        gps_boundary : RectangleBoundary
            The geographical boundary to filter out-of-area points.
        trajectories : Iterable[Trajectory]
            An iterable of raw trajectory data.
        max_vocab_size : int
            The maximum number of "hot" (i.e., non-special) locations to
            include in the vocabulary.
        min_freq : int
            The minimum number of times a cell must appear in the data to be
            considered for the vocabulary.

        Returns
        -------
        Dict[str, int]
            A dictionary mapping hot location strings and special tokens to their
            unique integer IDs.
        """
        all_cell_counter = count_locations_for_t2vec(
            trajectories,
            web_grid,
            gps_boundary,
        )

        # filter out all hot locations
        max_num_hotlocs = min(max_vocab_size, len(all_cell_counter))

        # 取出前max_num_hotlocs个热点token，这里要增加在频数相同时loc的排序，保证结果可复现
        top_loc_count = sorted(
            ((loc, cnt) for loc, cnt in all_cell_counter.items() if cnt >= min_freq),
            key=lambda x: (-x[1], x[0]),
            reverse=False,
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
    def fit(
        cls,
        web_grid: SimpleGridSystem,
        gps_boundary: RectangleBoundary,
        splits: Iterable[Iterable[Trajectory]],
        max_vocab_size: int,
        min_freq: int,
        with_kd_tree: bool = False,
    ) -> "T2VECTokenizer":
        """Fits a T2VECTokenizer instance from multiple splits of raw trajectory data.

        This factory method orchestrates the entire setup process. It first
        merges all splits into a single trajectory stream, then constructs the
        vocabulary of hot locations by calling `construct_vocab`. If
        `with_kd_tree` is True, it builds a KD-Tree on the coordinates of
        these hot locations for efficient nearest-neighbor search.

        Parameters
        ----------
        web_grid : SimpleGridSystem
            See `construct_vocab` for details.
        gps_boundary : RectangleBoundary
            See `construct_vocab` for details.
        splits : Iterable[Iterable[Trajectory]]
            An iterable of splits, where each split is an iterable of raw
            trajectories. For example, pass ``[train_trajs]`` for a single
            split or ``[train_trajs, val_trajs, test_trajs]`` to build the
            vocabulary from multiple splits.
        max_vocab_size : int
            See `construct_vocab` for details.
        min_freq : int
            See `construct_vocab` for details.
        with_kd_tree : bool, optional
            If True, a KD-Tree is built over the hot locations. This allows the
            tokenizer to map any out-of-vocabulary cell to its nearest hot
            neighbor. If False, out-of-vocabulary cells are mapped to the UNK
            token. Defaults to False.

        Returns
        -------
        T2VECTokenizer
            A new, fully configured instance of the T2VECTokenizer.
        """
        vocab = cls.construct_vocab(
            web_grid=web_grid,
            gps_boundary=gps_boundary,
            trajectories=list(itertools.chain.from_iterable(splits)),
            max_vocab_size=max_vocab_size,
            min_freq=min_freq,
        )

        hot_locs = None
        hotcell_kdtree = None
        if with_kd_tree:
            print("constructing KDTree.")
            SPECIAL_TOKENS = TokenEnum.values()
            hot_locs = [loc for loc in vocab if loc not in SPECIAL_TOKENS]
            web_mercator_coords = [
                [coord.x, coord.y]
                for coord in web_grid.grid_symbols_to_centroid_coords(hot_locs)
            ]
            hotcell_kdtree = KDTree(web_mercator_coords, leaf_size=2)
        return T2VECTokenizer(
            web_grid=web_grid,
            gps_boundary=gps_boundary,
            vocab=vocab,
            with_kd_tree=with_kd_tree,
            hot_locations=hot_locs,
            kdtree=hotcell_kdtree,
        )

    def __getstate__(self):
        """Prepares the tokenizer's state for pickling.

        Excludes the `nearest_hot_map` cache, as it is a transient state
        that can be rebuilt at runtime.
        """
        state = self.__dict__.copy()
        if "nearest_hot_map" in state:
            del state["nearest_hot_map"]
        return state

    def __setstate__(self, state):
        """Restores the tokenizer's state from a pickled state.

        Re-initializes the transient `nearest_hot_map` cache as an empty dict.
        """
        self.__dict__.update(state)
        self.nearest_hot_map = {}

    def _nearest_hot_loc(self, loc: str) -> str:
        """Finds the nearest hot location for a given location string.

        Uses the pre-built KD-Tree for an efficient nearest neighbor search.
        Results are cached to speed up subsequent lookups for the same location.
        """
        if loc in self.nearest_hot_map:
            return self.nearest_hot_map[loc]

        centroid = self.web_grid.grid_symbol_to_centroid_coord(loc)
        web_point = [centroid.x, centroid.y]
        _, idxs = self.kdtree.query([web_point], 1)
        nearest_hot_loc = self.hot_locations[idxs[0][0]]
        self.nearest_hot_map[loc] = nearest_hot_loc
        return nearest_hot_loc

    def k_nearest_hot_loc(
        self, loc_list: List[str], k: int
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """Finds the k-nearest hot locations for a list of given locations.

        This is a utility method for nearest neighbor queries.

        Parameters
        ----------
        loc_list : List[str]
            A list of location strings to query.
        k : int
            The number of nearest neighbors to find.

        Returns
        -------
        Tuple[np.ndarray, List[List[str]]]
            A tuple containing:
            - An array of distances to the k-nearest neighbors.
            - A list of lists, where each inner list contains the k-nearest hot
              location strings.
        """
        points = [
            [coord.x, coord.y]
            for coord in self.web_grid.grid_symbols_to_centroid_coords(loc_list)
        ]
        # dists, idxs shape: (len(loc_list), k)
        dists, idxs = self.kdtree.query(points, k)
        return dists, [
            [self.hot_locations[idx] for idx in line_indices] for line_indices in idxs
        ]

    def traj_to_loc_seq(
        self,
        traj: Union[Trajectory, TrajectoryView],
        add_bos: bool,
        add_eos: bool,
    ) -> List[str]:
        """Transforms a raw trajectory into a sequence of location strings.

        This method performs the core discretization logic:
        1. Converts GPS points to grid cell token symbols (e.g., "G_{x}_{y}").
        2. Handles points outside the defined boundary.
        3. Maps each cell to either a hot location, its nearest neighbor (if
           using KD-Tree), or an unknown tag.
        4. Optionally prepends/appends BOS/EOS tokens.

        Parameters
        ----------
        traj : Union[Trajectory, TrajectoryView]
            The input trajectory.
        add_bos : bool
            Whether to prepend the BOS token string.
        add_eos : bool
            Whether to append the EOS token string.

        Returns
        -------
        List[str]
            The resulting sequence of location strings.
        """
        if isinstance(traj, (Trajectory, TrajectoryView)):
            seq = traj
        else:
            raise TypeError(
                f"`traj` must be a Trajectory or TrajectoryView, not {type(traj)}. "
                "Use Trajectory.from_numpy(arr, coord_system) to create one."
            )

        loc_seq = convert_coords_to_locseq(
            seq,
            self.web_grid,
            self.boundary,
            self.unknown_loc_tag,
            TokenEnum.BOS_TOKEN.value if add_bos else None,
            TokenEnum.EOS_TOKEN.value if add_eos else None,
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

    def encode_traj(
        self,
        traj: Union[Trajectory, TrajectoryView],
        add_bos: bool = False,
        add_eos: bool = False,
        return_as: str = "py",
    ) -> Union[List[int], np.ndarray, torch.LongTensor]:
        """Tokenizes a raw trajectory into a sequence of numerical IDs.

        This method orchestrates the full tokenization pipeline by first
        converting the trajectory to a location sequence via `traj_to_loc_seq`,
        and then using the base class's `encode_locseq` to convert the
        string sequence to integer IDs.

        Parameters
        ----------
        traj : Union[Trajectory, TrajectoryView]
            The trajectory to tokenize.
        add_bos : bool, optional
            Whether to add the BOS token, by default False.
        add_eos : bool, optional
            Whether to add the EOS token, by default False.
        return_as : str, optional
            The desired output format ('py', 'np', 'pt'), by default 'py'.

        Returns
        -------
        Union[List[int], np.ndarray, torch.LongTensor]
            The final sequence of token IDs.
        """

        if isinstance(traj, (Trajectory, TrajectoryView)):
            inputs = traj
        else:
            raise TypeError(
                f"`traj` must be a Trajectory or TrajectoryView, not {type(traj)}. "
                "Use Trajectory.from_numpy(arr, coord_system) to create one."
            )

        loc_seq = self.traj_to_loc_seq(inputs, add_bos=add_bos, add_eos=add_eos)
        return self.encode_locseq(
            loc_seq=loc_seq, add_bos=False, add_eos=False, return_as=return_as
        )
