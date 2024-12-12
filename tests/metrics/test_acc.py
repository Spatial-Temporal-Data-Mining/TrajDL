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
import torch

from trajdl.metrics.acc import AccMetrics


def test_acc():
    m = AccMetrics()
    assert m.num_correct == 0
    assert m.num_samples == 0
    assert not m.value()

    num_samples = 0
    num_corrects = 0

    for _ in range(100):
        batch_size = random.randint(1, 128)
        pred = torch.randint(low=0, high=2, size=(batch_size,))
        label = torch.randint(low=0, high=2, size=(batch_size,))

        m.update(pred, label)

        num_samples += batch_size
        num_corrects += (pred == label).sum()

    assert "Acc" in m.value()
    assert m.num_samples == num_samples
    assert m.num_correct == num_corrects
    assert np.allclose(m.value()["Acc"], num_corrects * 100 / num_samples)

    m.reset()
    assert m.num_correct == 0
    assert m.num_samples == 0
    assert not m.value()
