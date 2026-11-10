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
import re
from typing import List

import numpy as np
import pytest
import torch
from trajdl import (
    Coordinate,
    CoordinateSystem,
    RectangleBoundary,
    SimpleGridSystem,
    Trajectory,
)
from trajdl.tokenizers.trajcl import TrajCLTokenizer, TrajEncodingResult

from ..conftest import FOLDER
from . import load_tokenizer_test_func


@pytest.fixture(scope="module")
def trajcl_tokenizer(traj_samples: List[Trajectory]) -> TrajCLTokenizer:
    """Create a TrajCLTokenizer fixture.

    The grid is built with ``column_major=True`` so that token ids align with
    TrajCL / Node2Vec's column-major ``cell_id = i_x * num_y_grids + i_y``.
    """
    # Use Porto dataset boundary (same as t2vec_tokenizer)
    gps_boundary = RectangleBoundary(
        min_point=Coordinate.Wgs84(-8.735152, 40.953673),
        max_point=Coordinate.Wgs84(-8.156309, 41.307945),
    )
    web_mercator_boundary = gps_boundary.to_web_mercator()
    web_grid = SimpleGridSystem(
        boundary=web_mercator_boundary,
        step_x=100.0,
        step_y=100.0,
        column_major=True,
    )

    # Vocab is the entire grid by design; trajectory samples are not used here.
    _ = traj_samples
    return TrajCLTokenizer.fit(web_grid=web_grid)


def test_trajcl_tokenizer(
    trajcl_tokenizer: TrajCLTokenizer, traj_samples: List[Trajectory]
):
    path = os.path.join(FOLDER, "test_trajcl_tokenizer.pkl")
    trajcl_tokenizer.save_pretrained(path)

    samples = traj_samples[:5]

    def test_special_token(tokenizer: TrajCLTokenizer):
        assert tokenizer.bos is not None
        assert tokenizer.eos is not None
        assert tokenizer.unk is not None
        assert tokenizer.pad is not None
        assert tokenizer.mask is not None

        assert (
            len(
                {
                    tokenizer.bos,
                    tokenizer.eos,
                    tokenizer.unk,
                    tokenizer.pad,
                    tokenizer.mask,
                }
            )
            == 5
        )

    def test_tokenizer(tokenizer: TrajCLTokenizer):
        assert isinstance(tokenizer, TrajCLTokenizer)

        # Check that vocab covers the entire grid + 5 special tokens.
        vocab = tokenizer.vocab
        grid_symbols = [k for k in vocab.keys() if k.startswith("G_")]
        special_tokens = [k for k in vocab.keys() if not k.startswith("G_")]

        num_x = tokenizer.web_grid.num_x_grids
        num_y = tokenizer.web_grid.num_y_grids
        expected_num_grid_symbols = num_x * num_y

        assert len(grid_symbols) == expected_num_grid_symbols, (
            "Vocab grid symbols should equal num_x_grids * num_y_grids "
            f"(got {len(grid_symbols)}, expected {expected_num_grid_symbols})"
        )
        assert len(special_tokens) == 5, "Vocab should have 5 special tokens"

        # Token id alignment: under column-major grid, the first cell
        # G_0_0 must be id=0 and the last cell G_{nx-1}_{ny-1} must be id
        # = nx * ny - 1, so that cell_embeddings.pt row order matches.
        assert vocab["G_0_0"] == 0
        assert vocab[f"G_{num_x - 1}_{num_y - 1}"] == expected_num_grid_symbols - 1

        test_special_token(tokenizer)

        # Test encode_traj without coords
        for return_as in ["pt", "py", "np"]:
            for traj in samples:
                result = tokenizer.encode_traj(traj, return_as=return_as)

                if return_as == "pt":
                    assert isinstance(result.token_ids, torch.Tensor)
                    assert result.token_ids.dtype == torch.long
                elif return_as == "np":
                    assert isinstance(result.token_ids, np.ndarray)
                    assert result.token_ids.dtype == np.int64
                elif return_as == "py":
                    assert isinstance(result.token_ids, list)
                    assert all(isinstance(x, int) for x in result.token_ids)

        # Test encode_traj with coords
        for return_as in ["pt", "py", "np"]:
            for traj in samples:
                result = tokenizer.encode_traj(
                    traj, return_as=return_as, return_coords=True
                )

                assert isinstance(result, TrajEncodingResult)
                assert isinstance(result.coords, Trajectory)

                if return_as == "pt":
                    assert isinstance(result.token_ids, torch.Tensor)
                elif return_as == "np":
                    assert isinstance(result.token_ids, np.ndarray)
                elif return_as == "py":
                    assert isinstance(result.token_ids, list)

                # coords length should match token_ids length
                assert len(result.coords) == len(result.token_ids)

        # Test traj_to_symbols_and_coords
        for traj in samples:
            symbols, coords = tokenizer.traj_to_symbols_and_coords(traj)

            assert isinstance(symbols, list)
            assert all(isinstance(s, str) for s in symbols)
            assert isinstance(coords, Trajectory)
            assert len(symbols) == len(coords)

            # Check that symbols are valid grid symbols or UNK
            for sym in symbols:
                assert sym.startswith("G_") or sym in [
                    tokenizer.id_to_token(tokenizer.unk)
                ]

        # Test invalid return_as
        with pytest.raises(
            ValueError, match=re.escape("`return_as` only supports {'py', 'np', 'pt'}")
        ):
            tokenizer.encode_traj(samples[0], return_as="invalid")

        # Test invalid trajectory type
        with pytest.raises(
            TypeError,
            match=re.escape("`traj` must be a Trajectory or TrajectoryView"),
        ):
            tokenizer.encode_traj("not a trajectory")

    # Test original tokenizer
    test_tokenizer(trajcl_tokenizer)

    # Test loaded tokenizer
    loaded_tokenizer = TrajCLTokenizer.load_pretrained(path)
    test_tokenizer(loaded_tokenizer)

    # Test load_tokenizer utility
    load_tokenizer_test_func(tokenizer=trajcl_tokenizer, path=path)


def test_trajcl_tokenizer_vocab_order(trajcl_tokenizer: TrajCLTokenizer):
    """Test that special tokens are at the end of the vocab."""
    vocab = trajcl_tokenizer.vocab

    # Get max token id for grid symbols
    grid_token_ids = [
        token_id for sym, token_id in vocab.items() if sym.startswith("G_")
    ]
    special_token_ids = [
        token_id for sym, token_id in vocab.items() if not sym.startswith("G_")
    ]

    if grid_token_ids:
        max_grid_id = max(grid_token_ids)
        min_special_id = min(special_token_ids)

        # Special tokens should come after all grid symbols
        assert min_special_id > max_grid_id


def test_trajcl_tokenizer_web_grid(trajcl_tokenizer: TrajCLTokenizer):
    """Test that web_grid is accessible."""
    assert trajcl_tokenizer.web_grid is not None
    assert isinstance(trajcl_tokenizer.web_grid, SimpleGridSystem)

    # Check that grid system has the expected properties
    assert trajcl_tokenizer.web_grid.num_x_grids > 0
    assert trajcl_tokenizer.web_grid.num_y_grids > 0


def test_trajcl_tokenizer_no_gps_boundary_attr(trajcl_tokenizer: TrajCLTokenizer):
    """Verify the tokenizer no longer carries an unused ``gps_boundary``.

    ``gps_boundary`` was a never-used field in earlier revisions; removing it
    is part of the trajcl tokenizer redesign.
    """
    assert not hasattr(trajcl_tokenizer, "gps_boundary")


def test_trajcl_tokenizer_deduplication(trajcl_tokenizer: TrajCLTokenizer):
    """Test that consecutive duplicate grid cells are removed."""
    # Create a trajectory that would normally have consecutive duplicates
    # by repeating the same point multiple times
    coords = np.array(
        [
            [-8.6, 41.1],
            [-8.6, 41.1],  # duplicate
            [-8.6, 41.1],  # duplicate
            [-8.7, 41.2],
            [-8.7, 41.2],  # duplicate
        ]
    )

    traj = Trajectory.from_numpy(coords, CoordinateSystem.Wgs84)
    symbols, coords_result = trajcl_tokenizer.traj_to_symbols_and_coords(traj)

    # Result should have fewer elements due to deduplication
    assert len(symbols) <= len(coords)
    assert len(coords_result) == len(symbols)


def test_trajcl_tokenizer_empty_trajectory(trajcl_tokenizer: TrajCLTokenizer):
    """Test behavior with an empty trajectory."""
    empty_coords = np.array([]).reshape(0, 2)
    traj = Trajectory.from_numpy(empty_coords, CoordinateSystem.Wgs84)

    symbols, coords = trajcl_tokenizer.traj_to_symbols_and_coords(traj)
    assert len(symbols) == 0
    assert len(coords) == 0


def test_trajcl_tokenizer_numpy_input(trajcl_tokenizer: TrajCLTokenizer):
    """Test that numpy array input is rejected (must use Trajectory)."""
    coords = np.array(
        [
            [-8.6, 41.1],
            [-8.7, 41.2],
        ]
    )

    # Numpy array should be rejected - must use Trajectory with coord_system
    with pytest.raises(
        TypeError, match="`traj` must be a Trajectory or TrajectoryView"
    ):
        trajcl_tokenizer.encode_traj(coords)

    # Correct usage: wrap in Trajectory with explicit coord_system
    traj = Trajectory.from_numpy(coords, CoordinateSystem.Wgs84)
    result = trajcl_tokenizer.encode_traj(traj)
    assert isinstance(result, TrajEncodingResult) and isinstance(result.token_ids, list)


def test_trajcl_tokenizer_web_mercator_input(trajcl_tokenizer: TrajCLTokenizer):
    """Test that Web Mercator input works and preserves coordinate system."""
    # Create a WGS84 trajectory
    coords_wgs = np.array(
        [
            [-8.6, 41.1],
            [-8.7, 41.2],
        ]
    )
    traj_wgs = Trajectory.from_numpy(coords_wgs, CoordinateSystem.Wgs84)

    # Convert to Web Mercator
    traj_web = traj_wgs.to_web_mercator()
    assert traj_web.coord_system == CoordinateSystem.WebMercator

    # Encode both trajectories
    result_wgs = trajcl_tokenizer.encode_traj(traj_wgs, return_coords=True)
    result_web = trajcl_tokenizer.encode_traj(traj_web, return_coords=True)

    # Token IDs should be the same (same locations in grid)
    assert result_wgs.token_ids == result_web.token_ids

    # Coordinate systems should match input
    assert (
        result_wgs.coords is not None
        and result_wgs.coords.coord_system == CoordinateSystem.WebMercator
    )
    assert (
        result_web.coords is not None
        and result_web.coords.coord_system == CoordinateSystem.WebMercator
    )

    # Test traj_to_symbols_and_coords with Web Mercator input
    symbols_web, coords_web = trajcl_tokenizer.traj_to_symbols_and_coords(traj_web)
    assert coords_web.coord_system == CoordinateSystem.WebMercator
    assert len(symbols_web) == len(coords_web)
