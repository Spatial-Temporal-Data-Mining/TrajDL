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
from typing import Callable, Optional, Tuple, Union

import lightning as L
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm

from trajdl import trajdl_cpp
from trajdl.algorithms.t2vec import T2VEC
from trajdl.common.samples import T2VECSample
from trajdl.datasets import Trajectory, TrajectoryDataset
from trajdl.tokenizers.t2vec import T2VECTokenizer
from trajdl.utils import get_num_cpus

L.seed_everything(42, workers=True)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--exp", choices=["exp1", "exp2", "exp3"])
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--rate", nargs="*", type=float)
args = parser.parse_args()


def load_a_trained_model(checkpoint_path: str):
    return T2VEC.load_from_checkpoint(checkpoint_path).eval()


def uniformsplit(
    traj: Trajectory, func: Union[Callable, None] = None
) -> Tuple[Trajectory, Trajectory]:
    entity_id = traj.entity_id

    r1, r2 = func(traj.seq)
    return Trajectory(seq=r1, entity_id=entity_id), Trajectory(
        seq=r2, entity_id=entity_id
    )


def collate_fn(batch, tokenizer: T2VECTokenizer) -> T2VECSample:
    samples = []
    lengths = []
    for traj in batch:
        samples.append(tokenizer.tokenize_traj(traj, return_as="pt"))
        lengths.append(len(samples[-1]))
    # (B, T), List[int]
    return T2VECSample(
        src=pad_sequence(samples, batch_first=True, padding_value=tokenizer.pad),
        lengths=lengths,
    )


if __name__ == "__main__":
    folder = args.folder
    output_folder = os.path.join(folder, "eval")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    query_size = 1000
    ckpt_path = args.ckpt_path
    output_prefix = args.exp
    rate_list = args.rate
    if rate_list is None:
        rate_list = [None]

    def get_split_func(exp_name: str, rate: Optional[float]):
        if exp_name == "exp1":
            return lambda x: trajdl_cpp.split_traj(x)
        elif exp_name == "exp2":
            return lambda x: trajdl_cpp.downsample_split_traj(x, rate)
        elif exp_name == "exp3":
            return lambda x: trajdl_cpp.distort_split_traj(x, rate, 50.0)
        else:
            raise ValueError("exp should be {'exp1', 'exp2', 'exp3'}")

    tokenizer_path = os.path.join(folder, "tokenizer.pkl")
    tokenizer: T2VECTokenizer = T2VECTokenizer.load_pretrained(tokenizer_path)
    model = load_a_trained_model(ckpt_path)

    trajs = list(
        TrajectoryDataset.init_from_parquet(
            os.path.join(folder, "test_ds.parquet")
        ).iter_as_seqs()
    )
    trainer = L.Trainer(logger=False, enable_checkpointing=False)

    for rate in rate_list:

        def tmp_func(traj: Trajectory) -> Tuple[Trajectory, Trajectory]:
            return uniformsplit(traj, func=get_split_func(output_prefix, rate))

        traj_list = (
            tmp_func(traj)
            for traj in tqdm(
                trajs, desc=f"construct dataset for {output_prefix}, rate: {rate}"
            )
        )

        query_list = []
        db_list = []
        for idx, (traj1, traj2) in enumerate(traj_list):
            if idx < query_size:
                query_list.append(traj1)
            db_list.append(traj2)
        print(f"size of query set: {len(query_list)}, size of db: {len(db_list)}")

        for dataset_name, dataset in (("query", query_list), ("db", db_list)):
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=lambda x: collate_fn(x, tokenizer),
                pin_memory=True,
                num_workers=get_num_cpus(),
            )

            output = []
            for emb in trainer.predict(model, dataloaders=[dataloader]):
                output.extend(i.detach().cpu().numpy() for i in emb)

            np.save(
                os.path.join(
                    output_folder, f"{output_prefix}-{dataset_name}-{rate}-vec.npy"
                ),
                output,
            )
            np.save(
                os.path.join(
                    output_folder, f"{output_prefix}-{dataset_name}-{rate}-label.npy"
                ),
                [int(traj.entity_id) for traj in dataset],
            )
