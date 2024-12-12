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

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from trajdl.algorithms.loc_pred.rnn import RNNNextLocPredictor, select_last_k_elements


def test_rnn():
    padding_value = 99

    emb = nn.Embedding(100, 12)
    model = RNNNextLocPredictor(
        embedding_layer=emb,
        rnn_hidden_size=32,
        fc_hidden_size=32,
        output_size=100,
        num_layers=1,
        padding_value=padding_value,
    )

    src = pad_sequence(
        [
            torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7]),
            torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            torch.LongTensor([0, 1, 2, 3, 4]),
        ],
        batch_first=True,
        padding_value=padding_value,
    )
    lengths = [8, 10, 5]
    output = model(src, lengths)

    pred_len = 3
    first_indices = torch.arange(3).unsqueeze(dim=1)
    second_indices = (
        torch.LongTensor(lengths).unsqueeze(dim=1) - pred_len + torch.arange(pred_len)
    )
    print(output[first_indices, second_indices].shape)


def test_select_last_k_elements():
    for _ in range(100):
        batch_size, num_timesteps, feature_size = (
            random.randint(1, 32),
            random.randint(6, 12),
            random.randint(1, 64),
        )
        pred_len = 3

        x = torch.rand(size=(batch_size, num_timesteps, feature_size))
        lengths = [random.randint(4, num_timesteps) for _ in range(batch_size)]

        result1 = select_last_k_elements(x=x, lengths=lengths, k=pred_len)

        result2 = torch.cat(
            [
                x[idx : idx + 1, length - pred_len : length]
                for idx, length in enumerate(lengths)
            ],
            dim=0,
        )
        assert torch.allclose(result1, result2)

    lengths = [random.randint(4, num_timesteps) for _ in range(batch_size)]
    lengths[0] = -1
    with pytest.raises(ValueError):
        select_last_k_elements(x=x, lengths=lengths, k=pred_len)

    lengths[0] = 0
    with pytest.warns(
        RuntimeWarning,
        match="There is a 0 value in `lengths` parameter",
    ):
        select_last_k_elements(x=x, lengths=lengths, k=pred_len)
