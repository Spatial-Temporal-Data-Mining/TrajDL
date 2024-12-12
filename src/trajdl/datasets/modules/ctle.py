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
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from ..arrow import LocSeqDataset
from .abstract import BaseLocSeqDataModule


@dataclass
class CTLEDataModule(BaseLocSeqDataModule):
    mask_prob: float = 0.2

    def __post_init__(self):
        super().__post_init__()

    def collate_function(self, ds: LocSeqDataset):
        # TODO: update doc
        loc_seq_cols = ds.seq
        ts_seq_cols = ds.ts_seq

        loc_list: List[torch.LongTensor] = []
        ts_list: List[torch.LongTensor] = []

        for idx in range(len(ds)):
            loc_list.append(
                self.tokenizer.tokenize_loc_seq(
                    loc_seq=loc_seq_cols[idx], return_as="pt"
                )
            )
            ts_list.append(torch.LongTensor(ts_seq_cols[idx].as_py()))

        loc_src = pad_sequence(
            loc_list, batch_first=True, padding_value=self.tokenizer.pad
        )
        ts_src = pad_sequence(ts_list, batch_first=True, padding_value=0)
        mask = (torch.rand(size=loc_src.shape) < self.mask_prob) & (
            loc_src != self.tokenizer.pad
        )

        return loc_src, ts_src, mask
