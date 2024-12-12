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

from torch.nn.utils.rnn import pad_sequence

from ...common.samples import GMVSAESample
from ...utils import valid_lengths_to_mask
from ..arrow import LocSeqDataset
from ..sampler.bucket import BucketSampler
from .abstract import BaseLocSeqDataModule


@dataclass
class GMVSAEDataModule(BaseLocSeqDataModule):
    num_train_batches: int = 10
    num_val_batches: int = 10

    num_train_buckets: int = 10
    num_val_buckets: int = 10

    def __post_init__(self):
        super().__post_init__()

    def setup(self, stage: str):
        super().setup(stage=stage)

        if self.train_ds:
            self.train_sampler = BucketSampler(
                ds=self.train_ds,
                num_buckets=self.num_train_buckets,
                num_batches=self.num_train_batches,
                batch_size=self.train_batch_size,
            )
        if self.val_ds:
            self.val_sampler = BucketSampler(
                ds=self.val_ds,
                num_buckets=self.num_val_buckets,
                num_batches=self.num_val_batches,
                batch_size=self.val_batch_size,
            )

    def collate_function(self, ds: LocSeqDataset) -> GMVSAESample:
        """
        返回5项
        1. 编码器的序列
        2. 编码器的长度
        3. 解码器的输入序列
        4. 解码器输入序列的长度
        5. 解码器解码序列的label
        6. 解码器需要计算损失的mask
        """
        samples, with_be_token, encoder_lengths = [], [], []
        for loc_seq in ds.seq:
            samples.append(self.tokenizer.tokenize_loc_seq(loc_seq, return_as="pt"))
            with_be_token.append(
                self.tokenizer.tokenize_loc_seq(
                    loc_seq, add_bos=True, add_eos=True, return_as="pt"
                )
            )
            encoder_lengths.append(len(samples[-1]))

        encoder_seq = pad_sequence(
            samples, batch_first=True, padding_value=self.tokenizer.pad
        )
        with_be_token_seq = pad_sequence(
            with_be_token, batch_first=True, padding_value=self.tokenizer.pad
        )
        decoder_lengths = [i + 1 for i in encoder_lengths]
        decoder_seq = with_be_token_seq[:, :-1]
        decoder_labels = with_be_token_seq[:, 1:]

        mask = valid_lengths_to_mask(decoder_lengths)

        return GMVSAESample(
            encoder_seq=encoder_seq,
            encoder_lengths=encoder_lengths,
            decoder_seq=decoder_seq,
            decoder_lengths=decoder_lengths,
            decoder_labels=decoder_labels,
            mask=mask,
        )
