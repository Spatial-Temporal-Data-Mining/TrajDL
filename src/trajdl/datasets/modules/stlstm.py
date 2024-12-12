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
from typing import Any, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from ...common.samples import STLSTMSample
from ...utils import load_bucketizer, valid_lengths_to_mask
from ..arrow import LocSeqDataset
from ..sampler.bucket import BucketSampler
from ..sampler.session import SessionSampler
from .abstract import BaseLocSeqDataModule


@dataclass
class STLSTMDataModule(BaseLocSeqDataModule):
    num_train_batches: int = 10
    num_val_batches: int = 10

    num_train_buckets: int = 10
    num_val_buckets: int = 10

    ts_bucketizer: Optional[Any] = None
    loc_bucketizer: Optional[Any] = None

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
        self.ts_bucketizer = load_bucketizer(self.ts_bucketizer)
        self.loc_bucketizer = load_bucketizer(self.loc_bucketizer)

    def collate_function(self, batch_ds: LocSeqDataset):
        loc_seqs = []
        ts_upper = []
        ts_lower = []
        sd_upper = []
        sd_lower = []
        valid_lengths = []
        labels = []
        for seq, tsd, disd in zip(batch_ds.seq, batch_ds.ts_delta, batch_ds.dis_delta):
            seq = self.tokenizer.tokenize_loc_seq(seq, return_as="pt")
            loc_seqs.append(seq[:-1])
            ts_upper.append(
                self.ts_bucketizer.get_bucket_indices(
                    self.ts_bucketizer.upper_bound - torch.Tensor(tsd.as_py())
                )[:-1]
            )
            ts_lower.append(
                self.ts_bucketizer.get_bucket_indices(
                    torch.Tensor(tsd.as_py()) - self.ts_bucketizer.lower_bound
                )[:-1]
            )
            sd_upper.append(
                self.loc_bucketizer.get_bucket_indices(
                    self.loc_bucketizer.upper_bound - torch.Tensor(disd.as_py())
                )[:-1]
            )
            sd_lower.append(
                self.loc_bucketizer.get_bucket_indices(
                    torch.Tensor(disd.as_py()) - self.loc_bucketizer.lower_bound
                )[:-1]
            )
            valid_lengths.append(len(loc_seqs[-1]))
            labels.append(seq[1:])
            if valid_lengths[-1] < 1:
                raise ValueError("The length of sequence must be greater than 1.")

        mask = valid_lengths_to_mask(valid_lengths)

        return STLSTMSample(
            loc_seq=pad_sequence(
                loc_seqs, batch_first=True, padding_value=self.tokenizer.pad
            ),
            td_upper_seq=pad_sequence(ts_upper, batch_first=True, padding_value=0),
            td_lower_seq=pad_sequence(ts_lower, batch_first=True, padding_value=0),
            sd_upper_seq=pad_sequence(sd_upper, batch_first=True, padding_value=0),
            sd_lower_seq=pad_sequence(sd_lower, batch_first=True, padding_value=0),
            valid_lengths=valid_lengths,
            labels=pad_sequence(
                labels, batch_first=True, padding_value=self.tokenizer.pad
            ),
            mask=mask,
        )


@dataclass
class HSTLSTMDataModule(BaseLocSeqDataModule):
    num_train_batches: int = 10
    num_val_batches: int = 10

    num_train_buckets: int = 10
    num_val_buckets: int = 10

    def __post_init__(self):
        super().__post_init__()

    def setup(self, stage: str):
        super().setup(stage=stage)
        self.train_sampler = SessionSampler(
            ds=self.train_ds,
            num_batches=self.num_train_batches,
            batch_size=self.train_batch_size,
        )

    def collate_function(self, ds: LocSeqDataset):
        # 每个元素是一个用户对应的样本所在的行下标
        session_indices: List[List[int]] = []

        user_sessions = []
        previous_user = ds.entity_id[0]
        for idx in range(len(ds)):
            user_id = ds.entity_id[idx]
            if user_id != previous_user:
                session_indices.append(user_sessions)
                user_sessions = []
            user_sessions.append(idx)
            previous_user = user_id
        session_indices.append(user_sessions)

        # 当前这个批次里面session数最多的样本的session数是多少
        max_session_length = max(
            len(user_sessions) for user_sessions in session_indices
        )

        # 每一项都是一个batch
        samples = [[] for _ in range(max_session_length)]

        # 遍历每个用户的session序列，然后将session组成batch放到samples里面
        for user_sessions in session_indices:
            for session_idx, sample_idx in enumerate(user_sessions):
                samples[session_idx].append(
                    (
                        self.tokenizer.tokenize_loc_seq(
                            ds.seq[sample_idx], return_as="pt"
                        ),
                        ds.ts_delta[sample_idx].as_py(),
                        ds.dis_delta[sample_idx].as_py(),
                    )
                )

        # 这里是把samples里面的batch挑出来之后，对位置序列、tsd、disd进行pad
        loc_samples, ts_delta_samples, dis_delta_samples = [], [], []
        valid_lengths = []
        for session in samples:
            loc_seq, ts_delta, dis_delta = zip(*session)
            valid_lengths.append([len(t) for t in loc_seq])
            loc_samples.append(
                pad_sequence(
                    loc_seq, batch_first=True, padding_value=self.tokenizer.pad
                )
            )

            ts_delta_samples.append(
                pad_sequence(
                    [torch.Tensor(t) for t in ts_delta],
                    batch_first=True,
                    padding_value=0,
                )
            )

            dis_delta_samples.append(
                pad_sequence(
                    [torch.Tensor(t) for t in dis_delta],
                    batch_first=True,
                    padding_value=0,
                )
            )

        # 返回三个List[Tensor]，每个tensor是一个batch，并且做了padding
        # 再返回一个List[List[int]]，每个元素是session的valid_length
        return loc_samples, ts_delta_samples, dis_delta_samples, valid_lengths
