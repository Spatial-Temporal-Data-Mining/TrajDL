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
import tempfile
from pathlib import Path
from typing import List

import polars as pl
import pytest
from trajdl import (
    Coordinate,
    CoordinateSystem,
    LocSeq,
    RectangleBoundary,
    SimpleGridSystem,
    Trajectory,
)
from trajdl.datasets import LocSeqDataset, TrajectoryDataset
from trajdl.datasets.open_source.conf import GowallaDataset, PortoDataset
from trajdl.grid.hierarchy import HierarchyGridSystem
from trajdl.tokenizers import LocSeqTokenizer, T2VECTokenizer, TrajCLTokenizer

FOLDER = Path(tempfile.gettempdir()) / "trajdl" / "output" / "tests"
os.makedirs(FOLDER, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def test_locseqs() -> List[LocSeq]:
    agg_df_path = os.path.join(FOLDER, "gowalla_agg_df.parquet")
    if os.path.exists(agg_df_path):
        agg_df = pl.read_parquet(agg_df_path)
    else:
        df = (
            GowallaDataset()
            .load(return_as="pl", unsafe=True)
            .sort(["user_id", "check_in_time"])
            .limit(10000)
        )
        agg_df = (
            df.group_by("user_id")
            .agg(pl.col("loc_id").sort_by(pl.col("check_in_time")))
            .sort(["user_id"])
        )
        agg_df.write_parquet(agg_df_path)

    return [
        LocSeq.from_list(seq=loc_list, entity_id=user_id)
        for user_id, loc_list in agg_df.iter_rows()
    ]


@pytest.fixture(scope="session", autouse=True)
def test_locseq_dataset(test_locseqs: List[LocSeq]) -> LocSeqDataset:
    return LocSeqDataset.init_from_seqs(test_locseqs)


@pytest.fixture(scope="session", autouse=True)
def locseq_tokenizer(test_locseqs: List[LocSeq]) -> LocSeqTokenizer:
    """
    构建一个简单的LocSeqTokenizer
    """
    return LocSeqTokenizer.fit(splits=[test_locseqs], count_start_end_token=True)


@pytest.fixture(scope="session", autouse=True)
def traj_samples() -> List[Trajectory]:
    traj_df_path = os.path.join(FOLDER, "traj_sample_df.parquet")
    if os.path.exists(traj_df_path):
        trajs = pl.read_parquet(traj_df_path)["POLYLINE"]
    else:
        traj_df = (
            PortoDataset()
            .load(return_as="pl", unsafe=True)
            .filter(~pl.col("MISSING_DATA"))
            .select("POLYLINE")
            .limit(10000)
        )
        traj_df.write_parquet(traj_df_path)
        trajs = traj_df["POLYLINE"]

    return [
        Trajectory.from_numpy(
            traj.to_numpy(),
            CoordinateSystem.Wgs84,
            entity_id=str(sample_idx),
        )
        for sample_idx, traj in enumerate(trajs)
    ]


@pytest.fixture(scope="session", autouse=True)
def test_trajectory_dataset(traj_samples: List[Trajectory]) -> TrajectoryDataset:
    return TrajectoryDataset.init_from_seqs(traj_samples)


@pytest.fixture(scope="session", autouse=True)
def t2vec_tokenizer(traj_samples: List[Trajectory]) -> T2VECTokenizer:
    gps_boundary = RectangleBoundary(
        min_point=Coordinate.Wgs84(-8.735152, 40.953673),
        max_point=Coordinate.Wgs84(-8.156309, 41.307945),
    )
    web_mercator_boundary = gps_boundary.to_web_mercator()
    web_grid = SimpleGridSystem(
        boundary=web_mercator_boundary, step_x=100.0, step_y=100.0
    )

    return T2VECTokenizer.fit(
        web_grid=web_grid,
        gps_boundary=gps_boundary,
        splits=[traj_samples],
        max_vocab_size=40000,
        min_freq=100,
        with_kd_tree=True,
    )


@pytest.fixture(scope="session", autouse=True)
def test_hierarchy() -> HierarchyGridSystem:
    boundary = RectangleBoundary(
        min_point=Coordinate.Wgs84(-8.690261, 41.140092),
        max_point=Coordinate.Wgs84(-8.549155, 41.185969),
    )
    steps = [(0.04, 0.016), (0.02, 0.005), (0.0008, 0.0009)]
    return HierarchyGridSystem(boundary=boundary, steps=steps)


@pytest.fixture(scope="session", autouse=True)
def trajcl_tokenizer(traj_samples: List[Trajectory]) -> TrajCLTokenizer:
    """Create a TrajCLTokenizer for testing.

    TrajCL relies on Node2Vec / cell_embeddings.pt whose row order is
    column-major (``cell_id = i_x * num_y_grids + i_y``), so the grid is built
    with ``column_major=True`` to make tokenizer ids align with that row order.
    """
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

    # Note: ``traj_samples`` is intentionally unused — TrajCL's vocab is the
    # entire grid (see TrajCLTokenizer docstring), so fit only needs web_grid.
    _ = traj_samples
    return TrajCLTokenizer.fit(web_grid=web_grid)
