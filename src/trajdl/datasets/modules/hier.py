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
from datetime import datetime
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from ..arrow import LocSeqDataset
from .abstract import BaseLocSeqDataModule


@dataclass
class HIERDataModule(BaseLocSeqDataModule):
    def __post_init__(self):
        super().__post_init__()

    def collate_function(self, ds: LocSeqDataset):
        # TODO: update doc
        loc_seq_cols = ds.seq
        ts_seq_cols = ds.ts_seq

        # 需要处理出week、hour、duration
        loc_list: List[torch.LongTensor] = []
        weekday_list: List[torch.LongTensor] = []
        hour_list: List[torch.LongTensor] = []
        duration_list: List[torch.LongTensor] = []
        lengths: List[int] = []

        for idx in range(len(ds)):
            loc_list.append(
                self.tokenizer.tokenize_loc_seq(
                    loc_seq=loc_seq_cols[idx], return_as="pt"
                )
            )
            ts_list = ts_seq_cols[idx].as_py()

            # transform ts into datetime
            datetime_list = [datetime.fromtimestamp(ts) for ts in ts_list]

            # weekday
            weekday_list.append(
                torch.LongTensor([date.weekday() for date in datetime_list])
            )

            # hour
            hour_list.append(torch.LongTensor([date.hour for date in datetime_list]))

            # duration
            duration_list.append(
                torch.LongTensor(
                    [
                        (ts_list[i + 1] - ts_list[i]) % (24 * 60 * 60) // 60 // 60
                        for i in range(len(ts_list) - 1)
                    ]
                )
            )

            lengths.append(len(loc_seq_cols[idx]))

        loc_src = pad_sequence(
            [i[:-1] for i in loc_list],
            batch_first=True,
            padding_value=self.tokenizer.pad,
        )
        week_src = pad_sequence(
            [i[:-1] for i in weekday_list], batch_first=True, padding_value=0
        )
        hour_src = pad_sequence(
            [i[:-1] for i in hour_list], batch_first=True, padding_value=0
        )
        duration_src = pad_sequence(duration_list, batch_first=True, padding_value=0)
        lengths = [i - 1 for i in lengths]

        targets = pad_sequence(
            [i[1:] for i in loc_list],
            batch_first=True,
            padding_value=self.tokenizer.pad,
        )
        return loc_src, week_src, hour_src, duration_src, lengths, targets
