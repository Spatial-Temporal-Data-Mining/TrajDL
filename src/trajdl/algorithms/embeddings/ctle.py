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

import math
from typing import Union

import numpy as np
import torch
from torch import nn

from ...tokenizers import AbstractTokenizer
from ...utils import load_tokenizer
from .base import BaseTokenEmbeddingLayer, SimpleEmbedding


class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim: int, max_len: int):
        super().__init__()

        # (max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)

        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # math.ceil(embedding_dim / 2)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * -(math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, embedding_dim)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        shape of x: (B, T, *)
        """

        # shape is (1, T, C)
        return self.pe[:, : x.size(1)]


class TemporalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        # (embedding_dim,)
        self.omega = nn.Parameter(
            (torch.Tensor(1 / 10 ** np.linspace(0, 9, embedding_dim))),
            requires_grad=True,
        )

        # (embedding_dim,)
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

        self.div_term = math.sqrt(1.0 / embedding_dim)

    def forward(self, x: torch.LongTensor):
        """
        x: shape is (B, T)
        """

        # (B, T, embedding_dim)
        time_encode = x.unsqueeze(-1) * self.omega.reshape(
            1, 1, -1
        ) + self.bias.reshape(1, 1, -1)
        time_encode = torch.cos(time_encode)

        # (B, T, embedding_dim)
        return self.div_term * time_encode


class CTLETokenEmbedding(BaseTokenEmbeddingLayer):
    def __init__(
        self,
        embedding_type: str,
        tokenizer: AbstractTokenizer,
        embedding_dim: int,
        max_len: int,
    ):
        super().__init__()
        self._embedding_type = embedding_type

        if embedding_type == "pe":
            self.emb = PositionalEncoding(embedding_dim=embedding_dim, max_len=max_len)
        elif embedding_type == "tp":
            self.emb = TemporalEncoding(embedding_dim=embedding_dim)
        else:
            raise ValueError("`embedding_type` only support {'pe', 'tp'}")

        self.loc_emb = SimpleEmbedding(tokenizer=tokenizer, embedding_dim=embedding_dim)

    @property
    def embedding_type(self) -> str:
        return self._embedding_type

    def forward(self, masked_tokens: torch.LongTensor, ts_src: torch.LongTensor):
        """
        masked_tokens: shape is (B, T)
        ts_src: shape is (B, T)

        """
        # (B, T, C)
        loc_emb = self.loc_emb(masked_tokens)

        # (1, T, C) if embedding_type == "pe"
        # (B, T, C) if embedding_type == "tp"
        extra_emb = (
            self.emb(masked_tokens) if self.embedding_type == "pe" else self.emb(ts_src)
        )

        # (B, T, C)
        return extra_emb + loc_emb


class CTLETokenEmbeddingWithTransformer(BaseTokenEmbeddingLayer):
    def __init__(
        self,
        embedding_type: str,
        embedding_dim: int,
        max_len: int,
        num_layers: int,
        n_heads: int,
        tokenizer: Union[str, AbstractTokenizer],
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        tokenizer = load_tokenizer(tokenizer=tokenizer)

        self.mask_value = tokenizer.mask
        self.pad_value = tokenizer.pad

        self.emb = CTLETokenEmbedding(
            embedding_type=embedding_type,
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            max_len=max_len,
        )

        # 做一个多头注意力机制的编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        # 多头注意力机制的编码层
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim, eps=1e-6),
        )

    def forward(
        self,
        loc_src: torch.LongTensor,
        ts_src: torch.LongTensor,
        mask: torch.BoolTensor,
    ):
        masked_tensor = loc_src.clone()
        masked_tensor[mask] = self.mask_value

        # (B, T, C)
        emb = self.emb(masked_tensor, ts_src)

        # (B, T, C)
        return self.encoder(
            emb,
            mask=None,
            src_key_padding_mask=masked_tensor == self.pad_value,
        )
