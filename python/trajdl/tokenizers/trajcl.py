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

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from .. import (
    SimpleGridSystem,
    Trajectory,
    TrajectoryView,
    convert_traj_to_locseq_with_coords,
)
from ..common.enum import TokenEnum
from .abstract import AbstractTrajTokenizer


@dataclass
class TrajEncodingResult:
    """Result of encoding a trajectory with TrajCLTokenizer.

    Attributes
    ----------
    token_ids : Union[List[int], np.ndarray, torch.LongTensor]
        The sequence of token IDs corresponding to the trajectory.
    coords : Trajectory | None
        The coordinates corresponding to each token in the sequence,
        represented as a Trajectory object. Length equals len(token_ids).
        The coordinate system matches the input trajectory (WGS84 or Web Mercator).
    """

    token_ids: Union[List[int], np.ndarray, torch.LongTensor]
    coords: Trajectory | None


class TrajCLTokenizer(AbstractTrajTokenizer):
    """A tokenizer for the TrajCL model.

    Vocab construction policy
    -------------------------
    The vocab is built from the **entire grid** (every cell of ``web_grid``),
    not from the cells actually visited by any trajectory. This is required by
    TrajCL's design: cell embeddings come from Node2Vec on the grid adjacency
    graph (covering all geometric cells), and there is **no** KD-Tree fallback
    for out-of-vocabulary cells. If we only kept the visited cells in vocab,
    inference-time perturbations (e.g. ``distort``) would push points to
    never-visited cells and produce ``[UNK]`` tokens whose embeddings are
    random-init special tokens — destroying the geometric meaning. By making
    vocab the geometric full set, every boundary-inside point always lands on
    a real grid cell with a meaningful Node2Vec embedding.

    Token id alignment
    ------------------
    The ``SimpleGridSystem`` exposes ``enumerate_grid_symbols()`` which yields
    ``(symbol, raw_grid_id)`` pairs in either column-major or row-major order
    depending on how the grid was constructed. For TrajCL specifically, the
    upstream Node2Vec / cell_embeddings.pt convention is **column-major**
    (``cell_id = i_x * num_y_grids + i_y``), so the grid passed in here is
    expected to have ``column_major=True``. The token id of a grid cell then
    naturally equals its raw grid id, which equals the row index of its
    embedding in ``cell_embeddings.pt``. Special tokens are appended at the
    tail.

    OOV behavior
    ------------
    Theoretically, after the upstream pipeline filters trajectories to be
    within ``boundary + buffer`` (the same buffer used to train Node2Vec), no
    point should fall outside the grid. If that contract is broken — for
    example the user feeds in unfiltered trajectories — out-of-grid points
    will be mapped to ``[UNK]`` by the underlying Rust converter, and this
    tokenizer will emit a one-shot ``RuntimeWarning`` from
    :meth:`traj_to_symbols_and_coords` to surface the issue without spamming
    the log.
    """

    def __init__(
        self,
        web_grid: SimpleGridSystem,
        vocab: Dict[str, int],
    ):
        """Initializes the TrajCLTokenizer.

        Parameters
        ----------
        web_grid : SimpleGridSystem
            The grid system (in Web Mercator) used for discretizing coordinates.
            Should be constructed with ``column_major=True`` so that the token
            ids produced by :meth:`fit` align with TrajCL's cell_embeddings.pt
            row order.
        vocab : Dict[str, int]
            The vocabulary mapping token symbols to IDs.
        """
        super().__init__(vocab=vocab)
        self.web_grid = web_grid
        # One-shot warning latch for OOV cells. Pickled tokenizers will reset
        # this through ``__setstate__``-equivalent path (parent has none, so
        # the attribute is restored verbatim from ``__dict__``).
        self._oov_warned: bool = False

    @classmethod
    def fit(cls, web_grid: SimpleGridSystem) -> "TrajCLTokenizer":
        """Fits a TrajCLTokenizer over the entire grid.

        The vocab is constructed by enumerating every cell of ``web_grid`` via
        :meth:`SimpleGridSystem.enumerate_grid_symbols`, so that the resulting
        token id of each grid symbol equals its raw grid id under the grid's
        ``column_major`` setting. The five special tokens
        (``BOS / EOS / UNK / MASK / PAD``) are appended at the tail.

        Parameters
        ----------
        web_grid : SimpleGridSystem
            The grid system used for discretizing coordinates. For TrajCL +
            Node2Vec alignment, construct it with ``column_major=True`` so the
            resulting ``token_id`` equals ``i_x * num_y_grids + i_y``, matching
            the row order of ``cell_embeddings.pt``.

        Returns
        -------
        TrajCLTokenizer
            A new, fully configured instance of the TrajCLTokenizer.
        """
        # 把"枚举所有 grid cell + 给定 token id"这件事下沉到 Rust 侧的
        # SimpleGridSystem.enumerate_grid_symbols。它保证：
        #   1. 长度 == num_x_grids * num_y_grids
        #   2. token_id 严格 0..N 连续无空缺
        #   3. token_id 与 locate_by_grid_coordinate 的输出完全一致
        # 这样上层无需关心行/列主序，也避免在 Python 侧手写双层 for 引入 bug。
        vocab: Dict[str, int] = dict(web_grid.enumerate_grid_symbols())

        for token in (
            TokenEnum.BOS_TOKEN.value,
            TokenEnum.EOS_TOKEN.value,
            TokenEnum.UNK_TOKEN.value,
            TokenEnum.MASK_TOKEN.value,
            TokenEnum.PAD_TOKEN.value,
        ):
            if token not in vocab:
                vocab[token] = len(vocab)

        return cls(web_grid=web_grid, vocab=vocab)

    def encode_traj(
        self,
        traj: Union[Trajectory, TrajectoryView],
        add_start_end_token: bool = False,
        return_as: str = "py",
        return_coords: bool = False,
    ) -> TrajEncodingResult:
        """Tokenizes a raw trajectory into a :class:`TrajEncodingResult`.

        This is the primary interface required by :class:`AbstractTrajTokenizer`.
        It delegates to :meth:`traj_to_symbols_and_coords` for the
        trajectory-to-symbol conversion (with deduplication and OOV handling),
        then maps symbols to integer IDs via :meth:`encode_locseq`.

        Parameters
        ----------
        traj : Union[Trajectory, TrajectoryView]
            The trajectory object containing GPS points (WGS84 or Web Mercator).
        add_start_end_token : bool, optional
            Whether to prepend BOS and append EOS tokens, by default False.
        return_as : str, optional
            The desired output format for ``token_ids`` (``'py'``, ``'np'``,
            ``'pt'``), by default ``'py'``.
        return_coords : bool, optional
            If ``True``, the returned :class:`TrajEncodingResult` includes the
            per-token coordinates as a :class:`Trajectory`. If ``False``
            (default), ``coords`` is ``None``.

        Returns
        -------
        TrajEncodingResult
            A dataclass with ``token_ids`` and optionally ``coords``.
        """
        symbols, coords = self.traj_to_symbols_and_coords(traj)
        token_ids = self.encode_locseq(
            loc_seq=symbols,
            add_bos=add_start_end_token,
            add_eos=add_start_end_token,
            return_as=return_as,
        )
        return TrajEncodingResult(
            token_ids=token_ids,
            coords=coords if return_coords else None,
        )

    def traj_to_symbols_and_coords(
        self,
        traj: Union[Trajectory, TrajectoryView],
    ) -> Tuple[List[str], Trajectory]:
        """Converts a raw trajectory to a deduplicated token symbol sequence and
        corresponding coordinate trajectory.

        Consecutive duplicate grid cells are removed. Points outside the grid
        are mapped to the ``[UNK]`` symbol by the underlying Rust converter.
        When that happens we emit a one-shot ``RuntimeWarning`` to alert the
        caller that the upstream boundary-filtering contract was likely not
        honored — under TrajCL's design the vocab covers the entire grid, so
        an in-grid point should never become OOV.

        Parameters
        ----------
        traj : Union[Trajectory, TrajectoryView]
            The input trajectory (WGS84 or Web Mercator).

        Returns
        -------
        Tuple[List[str], Trajectory]
            - Deduplicated token symbol list.
            - Corresponding coordinates as a Trajectory (same coordinate system
              as input), same length as the symbol list.
        """
        if isinstance(traj, (Trajectory, TrajectoryView)):
            seq = traj
        else:
            raise TypeError(
                f"`traj` must be a Trajectory or TrajectoryView, not {type(traj)}. "
                "Use Trajectory.from_numpy(arr, coord_system) to create one."
            )

        unk_symbol = self.id_to_token(self.unk)

        # 将轨迹转换为token，并做近邻连续token去重，返回每个token第一个经纬度点的坐标（web mercator）
        symbols, coords = convert_traj_to_locseq_with_coords(
            seq,
            self.web_grid,
            unk_symbol,
        )

        # OOV 体检：trajcl 的 vocab 覆盖整个网格，任何在网格内的点都不应该变成 [UNK]。
        # 一旦出现 [UNK]，说明上层没有按 boundary 过滤轨迹（或网格 buffer 太小），
        # 这会让模型在该位置取到无几何意义的 special-token embedding，需要让用户感知。
        if not self._oov_warned and unk_symbol in symbols:
            warnings.warn(
                f"TrajCLTokenizer encountered out-of-grid point(s) and emitted "
                f"'{unk_symbol}'. The vocab covers the whole grid, so this "
                "indicates the input trajectory has points outside the grid "
                "boundary. Filter trajectories against `web_grid.boundary` "
                "upstream to avoid degraded embeddings. (This warning is "
                "emitted only once per tokenizer instance.)",
                RuntimeWarning,
                stacklevel=2,
            )
            self._oov_warned = True

        return symbols, coords
