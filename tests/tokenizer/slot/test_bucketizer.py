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
import re

import pytest
import torch

from trajdl.tokenizers.slot import Bucketizer


def test_bucketizer(tmp_path):
    lower_bound, upper_bound, num_buckets = 10, 50, 10
    bucketizer = Bucketizer(
        lower_bound=lower_bound, upper_bound=upper_bound, num_buckets=10
    )
    path = str(tmp_path / "bucketizer.pkl")

    bucketizer.save(path)

    for buc in [bucketizer, Bucketizer.load(path)]:
        for value in [-1, 51]:
            with pytest.warns(
                RuntimeWarning,
                match=re.escape(
                    f"Value {value} is out of bounds ({lower_bound}, {upper_bound})"
                ),
            ):
                buc.get_bucket_index(value)

        assert buc.get_bucket_index(lower_bound) == 0
        assert buc.get_bucket_index(upper_bound) == num_buckets - 1
        assert buc.get_bucket_index(20) == 2

        tensor_result = buc.get_bucket_indices(
            torch.Tensor(
                [lower_bound - 100, lower_bound, upper_bound, 20, upper_bound + 100]
            )
        )
        assert tensor_result.shape == (5,)
        assert tensor_result[0] == 0
        assert tensor_result[1] == 0
        assert tensor_result[2] == num_buckets - 1
        assert tensor_result[3] == 2
        assert tensor_result[4] == num_buckets - 1

        test_sample = [
            random.randint(lower_bound - 50, upper_bound + 50) for _ in range(100)
        ]
        tensor_result = buc.get_bucket_indices(torch.Tensor(test_sample))
        for idx, sample in enumerate(test_sample):
            assert tensor_result[idx].item() == buc.get_bucket_index(sample)
