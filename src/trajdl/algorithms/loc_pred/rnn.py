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

import warnings
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNNextLocPredictor(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        rnn_hidden_size: int,
        fc_hidden_size: int,
        output_size: int,
        num_layers: int,
        padding_value: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.add_module("embedding_layer", self.embedding_layer)

        self.rnn = nn.RNN(
            embedding_layer.embedding_dim,
            rnn_hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.out_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(rnn_hidden_size, fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, output_size),
        )

        self._pad = padding_value

    def forward(self, src: torch.LongTensor, lengths: List[int]):
        """
        Parameters
        ----------
        src: shape is (B, T)
        lengths: List[int]
        """

        # (B, T, C)
        emb = self.embedding_layer(src)

        inputs = pack_padded_sequence(
            input=emb, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        # (B, T, H), (B, H)
        output, _ = self.rnn(inputs)

        # (B, T, H), (B,)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=self._pad
        )

        return output


def select_last_k_elements(x: torch.Tensor, lengths: List[int], k: int) -> torch.Tensor:
    """
    选出序列里面最后k个元素，这里要考虑每个序列的实际长度，x是padding后的序列
    shape of x is (B, T, *)
    """
    if min(lengths) < 0:
        raise ValueError(
            f"The minimum of lengths should be a positive number or 0, not {min(lengths)}"
        )

    if min(lengths) == 0:
        warnings.warn("There is a 0 value in `lengths` parameter", RuntimeWarning)

    first_indices = torch.arange(x.shape[0]).unsqueeze(dim=1)
    second_indices = torch.LongTensor(lengths).unsqueeze(dim=1) - k + torch.arange(k)

    # (B, k, *)
    return x[first_indices, second_indices]
