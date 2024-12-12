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

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ...common.samples import TULERSample
from ..arrow import LocSeqDataset
from .abstract import BaseLocSeqDataModule


@dataclass
class TULERDataModule(BaseLocSeqDataModule):
    user_map: Optional[Union[str, Dict[str, int]]] = None

    def __post_init__(self):
        super().__post_init__()

    def setup(self, stage: str):
        super().setup(stage=stage)

        if isinstance(self.user_map, str):
            # key是user_id, idx是从0开始的下标
            with open(self.user_map, "rb") as f:
                self.user_map: Dict[str, int] = pickle.load(f)
        elif not isinstance(self.user_map, dict):
            raise ValueError(
                "user_map should be path of user_map or the user_map instance"
            )

    def collate_function(self, ds: LocSeqDataset):
        # TODO: update doc
        loc_seq_cols = ds.seq
        id_cols = ds.entity_id

        idx_sorted = np.argsort([len(loc_seq) for loc_seq in loc_seq_cols])[::-1]

        samples: List[torch.LongTensor] = []
        seq_len: List[int] = []
        labels: List[int] = []

        for idx in idx_sorted:
            samples.append(
                self.tokenizer.tokenize_loc_seq(
                    loc_seq=loc_seq_cols[idx], return_as="pt"
                )
            )
            seq_len.append(samples[-1].shape[0])
            labels.append(self.user_map[id_cols[idx].as_py()])

        return TULERSample(
            src=pad_sequence(
                samples, batch_first=True, padding_value=self.tokenizer.pad
            ),
            seq_len=seq_len,
            labels=torch.LongTensor(labels),
        )
