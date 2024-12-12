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

from typing import List, Tuple, Union

import numpy as np
import pyarrow as pa
from torch.utils.data import Dataset


class T2VECDataset(Dataset):
    def __init__(self, src_table: pa.Table, trg_table: pa.Table):
        super().__init__()
        self.src_table = src_table
        self.trg_table = trg_table

    def __len__(self) -> int:
        return len(self.src_table)

    def __getitem__(self, idx: int) -> Tuple[pa.ListScalar, pa.ListScalar]:
        return (
            self.src_table.column("src")[idx],
            self.trg_table.column("trg")[
                self.src_table.column("label_idx")[idx].as_py()
            ],
        )

    def __getitems__(self, indices: Union[np.ndarray, List[int]]):
        if all(isinstance(tmp, (int, np.integer)) for tmp in indices):
            new_indices = indices
        else:
            new_indices = (idx for index_list in indices for idx in index_list)
        return [self[i] for i in new_indices]
