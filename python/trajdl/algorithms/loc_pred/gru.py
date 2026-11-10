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

from typing import Optional

import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence

from ...utils import load_tokenizer


class GruLocPredictor(nn.Module):
    """Single-layer GRU next-location predictor replicating the TALE author's
    ``downstream/next_loc.py::LstmLocPredictor`` (which is a GRU despite the
    name) for strict author-protocol reproduction.

    Differences vs ``ErppNextLocPredictor``:
    - single-layer GRU (default hidden 256) instead of a 2-layer LSTM
      encoder/decoder with hidden 512;
    - consumes only the location-embedding sequence: no time feature
      concatenation, no decoder;
    - dropout on the embedded inputs, Tanh+Linear head;
    - weight init follows the author's ``weight_init``: orthogonal for GRU
      matrices, xavier_normal for Linear weights (the injected embedding
      layer is never re-initialized).

    Parameters
    ----------
    embedding_layer : nn.Module, optional
        Module mapping token ids (B, T) to embeddings (B, T, D); injected by
        the training framework when omitted.
    embedding_dim : int
    hidden_size : int, optional
        GRU hidden size, by default 256.
    output_size : int, optional
        Number of candidate locations; derived from ``tokenizer`` when omitted.
    tokenizer : str, optional
        Path of the tokenizer file, used to derive ``output_size``.
    dropout : float, optional
        Dropout on the embedded inputs, by default 0.1.
    """

    def __init__(
        self,
        embedding_layer: Optional[nn.Module] = None,
        *,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        output_size: Optional[int] = None,
        tokenizer: Optional[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if output_size is None:
            if tokenizer is None:
                raise ValueError(
                    "either output_size or tokenizer must be provided"
                )
            output_size = len(load_tokenizer(tokenizer))

        self.embedding_layer = embedding_layer
        if embedding_layer is not None:
            self.add_module("embedding_layer", embedding_layer)

        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, output_size))

        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for module in self.out_linear.modules():
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight.data)
                init.normal_(module.bias.data)

    def forward(
        self,
        full_seq: torch.LongTensor,
        valid_len: torch.LongTensor,
        pre_len: int,
        timestamp: torch.LongTensor,
    ) -> torch.Tensor:
        """Predict the next location (author protocol is ``pre_len = 1``).

        Parameters
        ----------
        full_seq : torch.LongTensor
            Token ids of shape (B, T); each row is the zero-padded history
            concatenated with the ``pre_len`` target locations.
        valid_len : torch.LongTensor
            Original sequence lengths of shape (B,), including the targets.
        pre_len : int
            Number of future locations to predict; must be 1.
        timestamp : torch.LongTensor
            Present for interface compatibility with
            ``ErppNextLocPredictor``; unused by this predictor.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, pre_len, output_size).
        """
        if pre_len != 1:
            raise ValueError(f"GruLocPredictor only supports pre_len=1, got {pre_len}")
        if self.embedding_layer is None:
            raise RuntimeError(
                "embedding_layer is not set; either pass one to the constructor "
                "or let the training framework inject it"
            )
        his_len = (valid_len - pre_len).tolist()
        emb = self.dropout(self.embedding_layer(full_seq))  # (B, T, D)
        packed = pack_padded_sequence(
            emb, his_len, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)  # hidden: (1, B, H)
        logits = self.out_linear(hidden.squeeze(0))  # (B, V)
        return logits.unsqueeze(1)  # (B, 1, V)
