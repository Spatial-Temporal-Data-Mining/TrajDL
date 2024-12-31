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
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

from trajdl import trajdl_cpp
from trajdl.common.enum import TokenEnum
from trajdl.datasets import Trajectory, TrajectoryDataset
from trajdl.datasets.open_source.conf import PortoDataset
from trajdl.grid.base import SimpleGridSystem
from trajdl.tokenizers.t2vec import T2VECTokenizer

boundary = trajdl_cpp.grid.RectangleBoundary(-8.735152, 40.953673, -8.156309, 41.307945)
grid = SimpleGridSystem(boundary=boundary.to_web_mercator(), step_x=100.0, step_y=100.0)

# 训练样本范围、验证样本范围、测试样本起始下标，测试样本数
NUM_TRAIN, NUM_VAL, TEST_START_IDX, NUM_TEST = 1000000, 10000, 1020000, 101000

# 训练和验证样本的序列长度范围
MIN_LENGTH, MAX_LENGTH = 20, 100

# 测试样本的序列长度范围
MIN_LENGTH_TEST, MAX_LENGTH_TEST = 60, 200

# KNN的size
K = 10


def get_all_trajs() -> List[Trajectory]:
    # 获取所有轨迹数据
    trajs = (
        PortoDataset()
        .load(return_as="pl")
        .filter(pl.col("MISSING_DATA") == False)
        .sort("TIMESTAMP")["POLYLINE"]
    )
    print(f"num trajectories: {len(trajs)}")

    return [
        Trajectory(traj_pl.to_numpy(), entity_id=str(idx))
        for idx, traj_pl in tenumerate(trajs, desc="transform trajectories")
    ]


def construct_dataset(output_folder: Path):
    # 读取轨迹数据
    all_trajs: List[Trajectory] = get_all_trajs()

    # 构建tokenizer
    tokenizer = T2VECTokenizer.build(
        grid=grid,
        boundary=boundary,
        trajectories=all_trajs,
        max_vocab_size=40000,
        min_freq=100,
        with_kd_tree=True,
    )
    print(f"num vocab: {len(tokenizer)}")
    tokenizer.save_pretrained(output_folder / "tokenizer.pkl")

    train_traj, val_traj = [], []
    for idx in trange(NUM_TRAIN + NUM_VAL, desc="construct train and val set"):
        traj = all_trajs[idx]
        if MIN_LENGTH <= len(traj) <= MAX_LENGTH:
            if idx <= NUM_TRAIN:
                train_traj.append(traj)
            else:
                val_traj.append(traj)

    test_traj = []
    for idx in trange(TEST_START_IDX, len(all_trajs), desc="construct test set"):
        traj = all_trajs[idx]
        if len(test_traj) >= NUM_TEST:
            break
        if MIN_LENGTH_TEST <= len(traj) <= MAX_LENGTH_TEST:
            test_traj.append(traj)

    print("saving test dataset...")
    TrajectoryDataset.init_from_trajectories(test_traj).save(
        output_folder / "test_ds.parquet"
    )
    return tokenizer, train_traj, val_traj


def downsampling_distort(traj_np: np.ndarray) -> List[np.ndarray]:
    return trajdl_cpp.batch_downsample_and_distort(
        traj_np, [0, 0.2, 0.4, 0.5, 0.6], [0, 0.2, 0.4, 0.6], 50.0
    )


def construct_pairs(traj_np):
    trg = tokenizer.tokenize_traj(traj_np, add_start_end_token=True)
    # 因为tokenization的时候会合并近邻的token，所以src是有可能比trg长的，因为有扰动
    return (
        [
            tokenizer.tokenize_traj(src_traj_np)
            for src_traj_np in downsampling_distort(traj_np)
        ],
        trg,
    )


def construct_pair_dataset(
    output_folder: Path, prefix: str, traj_list: List[Trajectory]
):
    schema_for_src = pa.schema(
        [
            pa.field("src", pa.large_list(pa.int64())),
            pa.field("label_idx", pa.int64()),
        ]
    )
    schema_for_trg = pa.schema(
        [
            pa.field("trg", pa.large_list(pa.int64())),
        ]
    )

    def tmp_func(traj_np: np.ndarray, idx: int):
        src_list, trg_sample = construct_pairs(traj_np)

        src_batch = pa.RecordBatch.from_arrays(
            [
                pa.array(src_list, type=pa.large_list(pa.int64())),
                pa.array([idx for _ in range(len(src_list))], type=pa.int64()),
            ],
            names=["src", "label_idx"],
        )

        trg_batch = pa.RecordBatch.from_arrays(
            [
                pa.array([trg_sample], type=pa.large_list(pa.int64())),
            ],
            names=["trg"],
        )

        return src_batch.serialize(), trg_batch.serialize()

    arrow_batches = Parallel(n_jobs=8)(
        delayed(tmp_func)(traj.seq, idx)
        for idx, traj in tenumerate(traj_list, desc=f"construct {prefix} dataset")
    )

    src_batches, trg_batches = zip(*arrow_batches)

    src_table = pa.Table.from_batches(
        pa.ipc.read_record_batch(batch, schema_for_src) for batch in tqdm(src_batches)
    )
    trg_table = pa.Table.from_batches(
        pa.ipc.read_record_batch(batch, schema_for_trg) for batch in tqdm(trg_batches)
    )

    pq.write_table(src_table, output_folder / f"full_{prefix}_src.parquet")
    pq.write_table(trg_table, output_folder / f"full_{prefix}_trg.parquet")


def train_knn(output_folder: Path, tokenizer: T2VECTokenizer):
    vocab_list = tokenizer.vocab.keys()
    SPECIAL_TOKENS = TokenEnum.values()
    loc_list, idx_list = zip(
        *(
            (loc, tokenizer.loc2idx(loc))
            for loc in vocab_list
            if loc not in SPECIAL_TOKENS
        )
    )
    dists, locations = tokenizer.k_nearest_hot_loc(loc_list, k=K)

    # (num_locations, K)
    V = np.zeros(shape=(len(vocab_list), K), dtype=np.int64)

    # (num_locations, K)
    D = np.zeros_like(V, dtype=np.float32)
    D[idx_list, :] = dists

    for token in SPECIAL_TOKENS:
        idx = tokenizer.loc2idx(token)
        V[idx] = idx

    for line_idx, loc_list in zip(idx_list, locations):
        V[line_idx] = [tokenizer.loc2idx(loc) for loc in loc_list]

    np.save(output_folder / "knn_indices.npy", V)
    np.save(output_folder / "knn_distances.npy", D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=False,
        default=Path("output") / "porto",
        help="用于管理数据的目录",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=-1,
        help="并行的进程数，默认是-1，使用所有CPU核心",
    )
    args = parser.parse_args()

    output_folder = args.output_folder
    tokenizer, train_traj_list, val_traj_list = construct_dataset(output_folder)
    construct_pair_dataset(output_folder, "train", train_traj_list)
    construct_pair_dataset(output_folder, "val", val_traj_list)
    train_knn(output_folder, tokenizer)
