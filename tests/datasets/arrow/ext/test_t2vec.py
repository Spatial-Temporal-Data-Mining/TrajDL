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

import random

import numpy as np
import pyarrow as pa

from trajdl.datasets.arrow.ext.t2vec import T2VECDataset


def test_t2vecdataset():
    max_seq_length = 10
    max_num_locs = 100
    for _ in range(10):
        num_labels = random.randint(1, 10)
        ratio = random.randint(1, 20)

        trg_table = pa.table(
            {
                "trg": [
                    [
                        str(random.randint(0, max_num_locs))
                        for _ in range(random.randint(1, max_seq_length))
                    ]
                    for _ in range(num_labels)
                ]
            }
        )

        src_table = pa.table(
            {
                "src": [
                    [
                        str(random.randint(0, max_num_locs))
                        for _ in range(random.randint(1, max_seq_length))
                    ]
                    for _ in range(num_labels * ratio)
                ],
                "label_idx": [
                    random.randint(0, num_labels - 1) for _ in range(num_labels * ratio)
                ],
            }
        )

        ds = T2VECDataset(src_table=src_table, trg_table=trg_table)
        assert len(ds) == num_labels * ratio
        assert isinstance(ds[0], tuple)
        assert isinstance(ds[0][0], pa.ListScalar)
        assert isinstance(ds[0][1], pa.ListScalar)

        indices1 = [0]
        result = ds.__getitems__(indices1)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert isinstance(result[0][0], pa.ListScalar)
        assert isinstance(result[0][1], pa.ListScalar)
        assert len(result) == len(indices1)

        indices2 = np.arange(random.randint(1, len(ds)))
        result = ds.__getitems__(indices2)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert isinstance(result[0][0], pa.ListScalar)
        assert isinstance(result[0][1], pa.ListScalar)
        assert len(result) == len(indices2)

        indices3 = [indices2]
        result = ds.__getitems__(indices3)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert isinstance(result[0][0], pa.ListScalar)
        assert isinstance(result[0][1], pa.ListScalar)
        assert len(result) == sum(1 for i in indices3 for _ in i)
