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

import numpy as np
import pytest

from trajdl.metrics.tul import TULMetrics


def test_tul():
    # TODO: update this test case to verify the correctness
    with pytest.raises(
        ValueError, match=re.escape("`num_users` should be greater than 1.")
    ):
        TULMetrics(num_users=1, topk=5)

    with pytest.raises(
        ValueError, match=re.escape("`num_users` should be greater than `topk`.")
    ):
        TULMetrics(num_users=5, topk=5)

    batch_size = random.randint(1, 32)
    num_users, topk = 10, 5
    for _ in range(10):
        metric = TULMetrics(num_users=num_users, topk=topk)
        metric.reset()

        preds = np.random.uniform(size=(batch_size, num_users))
        targets = np.random.randint(low=0, high=num_users, size=(batch_size,))
        metric.update(preds=preds, targets=targets)

    result = metric.value()
    assert "acc" in result
    assert "acc_topk" in result
    assert "macro-f1" in result
