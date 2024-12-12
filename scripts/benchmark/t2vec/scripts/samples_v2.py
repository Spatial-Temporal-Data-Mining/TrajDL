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
from typing import Generator, Iterable, List, Optional

import numpy as np
import polars as pl
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

from trajdl import trajdl_cpp
from trajdl.common.enum import TokenEnum
from trajdl.datasets import LocSeq, LocSeqDataset, Trajectory, TrajectoryDataset
from trajdl.datasets.open_source.conf import PortoDataset
from trajdl.grid.base import SimpleGridSystem
from trajdl.tokenizers.t2vec import T2VECTokenizer


def get_all_trajs() -> List[Trajectory]:
    """
    read porto csv file and build a trajectory generator
    """
    print("loading porto dataset...")
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


def get_next(
    iterator, min_length: int, max_length: int
) -> Optional[Generator[Trajectory, None, None]]:
    while True:
        try:
            traj = next(iterator)
            if min_length <= len(traj) <= max_length:
                return traj
        except StopIteration:
            return None


def create_dataset(
    all_trajs: Iterable[Trajectory],
    folder: str,
    num_train_trajs: int,
    num_val_trajs: int,
    num_test_trajs: int,
    min_length: int,
    max_length: int,
):

    iterator = iter(all_trajs)
    configs = [
        ("train", num_train_trajs),
        ("val", num_val_trajs),
        ("test", num_test_trajs),
    ]

    for prefix, num_trajs in configs:
        TrajectoryDataset.init_from_trajectories(
            get_next(iterator, min_length, max_length)
            for _ in trange(num_trajs, desc=f"generating {prefix} dataset")
        ).save(os.path.join(folder, f"{prefix}_ds.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="output/t2vec")
    args = parser.parse_args()

    output_folder = args.output_folder

    boundary = trajdl_cpp.RectangleBoundary(
        min_x=-8.735152,
        min_y=40.953673,
        max_x=-8.156309,
        max_y=41.307945,
    )
    grid = SimpleGridSystem(
        boundary=boundary.to_web_mercator(), step_x=100.0, step_y=100.0
    )

    all_trajs: List[Trajectory] = get_all_trajs()

    tokenizer = T2VECTokenizer.build(
        grid=grid,
        boundary=boundary,
        trajectories=all_trajs,
        max_vocab_size=40000,
        min_freq=100,
        with_kd_tree=True,
    )
    print(f"num vocab: {len(tokenizer)}")
    tokenizer.save_pretrained(os.path.join(output_folder, "tokenizer.pkl"))

    k = 10
    vocab_list = tokenizer.vocab.keys()
    SPECIAL_TOKENS = TokenEnum.values()
    loc_list, idx_list = zip(
        *(
            (loc, tokenizer.loc2idx(loc))
            for loc in vocab_list
            if loc not in SPECIAL_TOKENS
        )
    )
    dists, locations = tokenizer.k_nearest_hot_loc(loc_list, k=k)

    # (num_locations, k)
    V = np.zeros(shape=(len(vocab_list), k), dtype=np.int64)

    # (num_locations, k)
    D = np.zeros_like(V, dtype=np.float32)
    D[idx_list, :] = dists

    for token in SPECIAL_TOKENS:
        idx = tokenizer.loc2idx(token)
        V[idx] = idx

    for line_idx, loc_list in zip(idx_list, locations):
        V[line_idx] = [tokenizer.loc2idx(loc) for loc in loc_list]

    np.save(os.path.join(output_folder, "knn_indices.npy"), V)
    np.save(os.path.join(output_folder, "knn_distances.npy"), D)

    create_dataset(
        all_trajs=all_trajs,
        folder=output_folder,
        num_train_trajs=1000000,
        num_val_trajs=10000,
        num_test_trajs=101000,
        min_length=20,
        max_length=100,
    )

    train_ds = TrajectoryDataset.init_from_parquet(
        os.path.join(output_folder, "train_ds.parquet")
    )
    train_locseqs = [
        LocSeq(
            seq=tokenizer.traj_to_loc_seq(traj, add_start_end_token=True),
            entity_id=traj.entity_id,
        )
        for traj in tqdm(
            train_ds.iter_as_seqs(),
            total=len(train_ds),
            desc="transform trajectories to location sequences for word2vec",
        )
    ]
    train_locseq_ds = LocSeqDataset.init_from_loc_seqs(train_locseqs)
    train_locseq_ds.save(os.path.join(output_folder, "train_locseq_ds.parquet"))
