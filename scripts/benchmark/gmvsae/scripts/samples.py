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

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.contrib import tenumerate

from trajdl import trajdl_cpp
from trajdl.datasets import LocSeq, LocSeqDataset
from trajdl.datasets.open_source.conf import PortoDataset
from trajdl.grid import SimpleGridSystem
from trajdl.tokenizers.locseq import LocSeqTokenizer

seed = 42


def height2lat(height: float) -> float:
    return height / 110.574


def width2lng(width: float) -> float:
    return width / 111.320 / 0.99974


def get_loc_seqs(
    grid: SimpleGridSystem,
    shortest: int,
    longest: int,
) -> Generator[LocSeq, None, None]:
    trajectories = (
        PortoDataset()
        .load(return_as="pl")
        .select("POLYLINE")
        .filter(
            (pl.col("POLYLINE").list.len() >= shortest)
            & (pl.col("POLYLINE").list.len() <= longest)
        )["POLYLINE"]
    )

    def transform_traj_into_loc_seq(traj_np: np.ndarray, idx: int) -> Optional[LocSeq]:
        if grid.in_boundary_np(traj_np).all():
            return LocSeq(grid.locate_unsafe_np(traj_np), entity_id=str(idx))
        return None

    return Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(transform_traj_into_loc_seq)(traj_pl.to_numpy(), idx)
        for idx, traj_pl in tenumerate(
            trajectories, desc="transform trajectories into location sequences"
        )
    )


def _perturb_point(grid: SimpleGridSystem, loc: str, level: int, offset=None) -> str:
    """
    这个函数是用来扰动location用的
    先将位置还原成网格里面的横纵坐标，然后按九宫格里面外圈的8个方向移动这个坐标，获得新的location
    level用来表示向外挪动的距离，1就表示九宫格的外圈，2就表示再往外走的一圈，但实际上2的时候只会走8个方向的二阶距离，是米字

    Parameters
    ----------
    loc: str, 位置id
    level: int，级别
    """
    grid_x, grid_y = grid.to_grid_coordinate(loc)
    if offset is None:
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]
    else:
        x_offset, y_offset = offset
    if grid.in_boundary_by_grid_coordinate(
        grid_x + x_offset * level, grid_y + y_offset * level
    ):
        grid_x += x_offset * level
        grid_y += y_offset * level
    return grid.locate_by_grid_coordinate(grid_x, grid_y)


def perturb_locseq(locseq: LocSeq, level: int, prob: float) -> LocSeq:
    loc_list = [locseq.o]
    for idx in range(1, len(locseq) - 1):
        loc = locseq[idx]
        loc_list.append(
            _perturb_point(grid, loc, level) if np.random.random() < prob else loc
        )
    loc_list.append(locseq.d)
    return LocSeq(seq=loc_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=False,
        default=Path("output") / "porto",
        help="用于管理数据的目录",
    )
    args = parser.parse_args()

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    grid_height, grid_width = 0.1, 0.1
    lat_size, lng_size = height2lat(grid_height), width2lng(grid_width)
    grid = SimpleGridSystem(
        trajdl_cpp.grid.RectangleBoundary(
            min_x=-8.690261,
            min_y=41.140092,
            max_x=-8.549155,
            max_y=41.185969,
        ),
        step_x=lng_size,
        step_y=lat_size,
    )
    print("Grid size:", len(grid))

    shortest, longest = 20, 1200
    min_od_traj_num = 25
    test_traj_num = 5
    assert min_od_traj_num > test_traj_num

    od_agg = defaultdict(list)
    for loc_seq in get_loc_seqs(grid, shortest, longest):
        if loc_seq:
            od_agg[(loc_seq.o, loc_seq.d)].append(loc_seq)

    train_path = os.path.join(output_folder, "train_ds.parquet")
    val_path = os.path.join(output_folder, "val_ds.parquet")

    # 划分训练和验证集
    train_loc_seqs, val_loc_seqs, valid_ods = [], [], set()
    for od, loc_seqs in tqdm(od_agg.items(), desc="generating dataset"):
        num_loc_seqs = len(loc_seqs)
        if num_loc_seqs >= min_od_traj_num:
            for idx in range(num_loc_seqs - test_traj_num):
                train_loc_seqs.append(loc_seqs[idx])
            for idx in range(num_loc_seqs - test_traj_num, num_loc_seqs):
                val_loc_seqs.append(loc_seqs[idx])
            valid_ods.add(od)

    # 存储训练和验证集
    for loc_seqs, path in [(train_loc_seqs, train_path), (val_loc_seqs, val_path)]:
        LocSeqDataset.init_from_loc_seqs(loc_seqs).save(path)

    # 构建tokenizer
    tokenizer: LocSeqTokenizer = LocSeqTokenizer.build(
        loc_seqs=train_loc_seqs, count_start_end_token=False
    )
    tokenizer.save_pretrained(os.path.join(output_folder, "tokenizer.pkl"))

    # 对数据集进行扰动
    rng = np.random.default_rng(seed=seed)
    ratio = 0.05
    level = 2
    point_prob = 0.3

    num_train_trajs = len(train_loc_seqs)
    print(f"num_train_trajs: {num_train_trajs}")

    train_outlier_idx = rng.choice(
        num_train_trajs, int(num_train_trajs * ratio), replace=False
    )
    print(f"num outliers in training set: {train_outlier_idx.shape[0]}")

    if train_outlier_idx.shape[0] > 0:
        for outlier_idx in tqdm(train_outlier_idx):
            train_loc_seqs[outlier_idx] = perturb_locseq(
                train_loc_seqs[outlier_idx], level=level, prob=point_prob
            )

        LocSeqDataset.init_from_loc_seqs(train_loc_seqs).save(
            os.path.join(output_folder, "train_outliers_perturb.parquet")
        )

        np.save(os.path.join(output_folder, "train_outlier_idx.npy"), train_outlier_idx)
