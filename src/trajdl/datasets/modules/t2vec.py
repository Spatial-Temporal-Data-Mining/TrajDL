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

from dataclasses import dataclass
from typing import List, Tuple, Union

import lightning as L
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ...common.samples import T2VECSample
from ...tokenizers import AbstractTokenizer
from ...tokenizers.t2vec import T2VECTokenizer
from ...utils import get_num_cpus, load_tokenizer
from ...utils.traj import trajectory_distortion, trajectory_downsampling
from ..arrow import TrajectoryDataset
from ..arrow.ext.t2vec import T2VECDataset
from ..base import Trajectory
from ..sampler.bucket import BucketSampler
from ..sampler.t2vec import T2VECSampler
from .abstract import BaseTrajectoryDataModule


def downsampling_distort(traj: Trajectory) -> Trajectory:
    """
    给定一条轨迹数据，随机进行下采样和扰动，返回一条新的轨迹

    Parameters
    ----------
    traj: Trajectory
        轨迹

    Returns
    ----------
    traj: Trajectory
        经过下采样和扰动后的轨迹

    """
    dropping_rate = np.random.choice([0, 0.2, 0.4, 0.5, 0.6])
    distorting_rate = np.random.choice([0, 0.2, 0.4, 0.6])
    result = trajectory_downsampling(traj, dropping_rate)
    return trajectory_distortion(result, distorting_rate, 50.0)


def generate_samples(
    tokenizer: T2VECTokenizer, traj: Trajectory
) -> Tuple[torch.LongTensor, int, torch.LongTensor]:
    src = tokenizer.tokenize_traj(downsampling_distort(traj), return_as="pt")
    trg = tokenizer.tokenize_traj(traj, add_start_end_token=True, return_as="pt")
    return src, src.shape[0], trg


class T2VECDataModule(L.LightningDataModule):
    """
    T2VEC的DataModule，训练和验证用
    """

    def __init__(
        self,
        tokenizer: Union[str, AbstractTokenizer],
        train_src_path: str,
        train_trg_path: str,
        val_src_path: str,
        val_trg_path: str,
        train_batch_size: int,
        val_batch_size: int,
        num_train_batches: int,
        buckets_boundaries: List[Tuple[int, int]],
        num_cpus: int = -1,
    ):
        """
        Parameters
        ----------
        tokenizer: Union[str, AbstractTokenizer]
            tokenizer的path或者tokenizer实例

        train_src_path: str
            训练集src序列的数据集路径

        train_trg_path: str
            训练集trg序列的数据集路径

        val_src_path: str
            验证集src序列的数据集路径

        val_trg_path: str
            验证集trg序列的数据集路径

        train_batch_size: int
            训练集的batch size

        val_batch_size: int
            验证集的batch size

        num_train_batches: int
            训练集的batch数

        buckets_boundaries: List[Tuple[int, int]]
            训练集根据src和trg序列长度进行分桶时，桶的边界。论文代码使用的是[(20, 30), (30, 50), (50, 70), (70, 100)]
            会使用这些边界对训练数据进行分桶，提升训练效率，具体分桶逻辑需要看Sampler的实现

        num_cpus: int, optional
            dataloader的进程数，默认使用-1，即使用CPU的核心数

        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_src_path = train_src_path
        self.train_trg_path = train_trg_path
        self.val_src_path = val_src_path
        self.val_trg_path = val_trg_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_train_batches = num_train_batches
        self.buckets_boundaries = buckets_boundaries
        self.num_workers = get_num_cpus() if num_cpus < 0 else num_cpus

    def setup(self, stage: str):
        if hasattr(self, "train_ds"):
            del self.train_ds
        if hasattr(self, "val_ds"):
            del self.val_ds

        self.train_ds = T2VECDataset(
            pq.read_table(self.train_src_path), pq.read_table(self.train_trg_path)
        )
        self.val_ds = T2VECDataset(
            pq.read_table(self.val_src_path), pq.read_table(self.val_trg_path)
        )
        self.tokenizer = load_tokenizer(self.tokenizer)
        self.train_sampler = T2VECSampler(
            self.train_ds,
            self.buckets_boundaries,
            self.num_train_batches,
            self.train_batch_size,
        )

    def collate_fn_train(
        self, batch: List[Tuple[pa.ListScalar, pa.ListScalar]]
    ) -> T2VECSample:
        """
        batch: List[Tuple[pyarrow.ListScalar, pyarrow.ListScalar]]
        """
        src_list, lengths, target = [], [], []
        for src, trg in batch:
            src_list.append(torch.LongTensor(src.as_py()))
            lengths.append(len(src))
            target.append(torch.LongTensor(trg.as_py()))
        return T2VECSample(
            src=pad_sequence(
                src_list, batch_first=True, padding_value=self.tokenizer.pad
            ),
            lengths=lengths,
            target=pad_sequence(
                target, batch_first=True, padding_value=self.tokenizer.pad
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_train,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_train,
        )


@dataclass
class T2VECDataModuleV2(BaseTrajectoryDataModule):
    """
    k是用来控制负样本倍数的，当k等于1的时候，一个正样本对应一个负样本，k等于2的时候，一个正样本对应两个负样本
    """

    num_train_batches: int = 10
    num_val_batches: int = 10

    num_train_buckets: int = 10
    num_val_buckets: int = 10

    k: int = 1

    def __post_init__(self):
        super().__post_init__()

    def setup(self, stage: str):
        super().setup(stage=stage)

        self.train_sampler = BucketSampler(
            ds=self.train_ds,
            num_buckets=self.num_train_buckets,
            num_batches=self.num_train_batches,
            batch_size=self.train_batch_size,
        )
        self.val_sampler = BucketSampler(
            ds=self.val_ds,
            num_buckets=self.num_val_buckets,
            num_batches=self.num_val_batches,
            batch_size=self.val_batch_size,
        )

    def collate_function(self, ds: TrajectoryDataset) -> T2VECSample:
        batch_size = len(ds)

        if self.k > batch_size:
            raise RuntimeError("k must be less than batch size")

        rand_idx = np.random.choice(batch_size, batch_size // self.k)

        trajectories = ds.seq

        samples = []
        for idx in rand_idx:
            # TODO: this transformation should be optimized from
            # arrow -> py -> numpy -> cpp into arrow -> cpp
            traj = Trajectory(np.array(trajectories[idx].as_py()))
            for _ in range(self.k):
                samples.append(generate_samples(self.tokenizer, traj))

        src_samples, src_lens, trg_samples = zip(*samples)
        return T2VECSample(
            src=pad_sequence(
                src_samples, batch_first=True, padding_value=self.tokenizer.pad
            ),
            lengths=src_lens,
            target=pad_sequence(
                trg_samples, batch_first=True, padding_value=self.tokenizer.pad
            ),
        )
