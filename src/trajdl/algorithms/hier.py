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

from typing import List, Union

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..common.enum import LossEnum
from ..grid.hierarchy import HierarchyGridSystem
from ..tokenizers import AbstractTokenizer


class HIERSpatialEmbedding(nn.Module):
    def __init__(self, num_vocab: int, sizes: List[int], h_grid: HierarchyGridSystem):
        super().__init__()
        self.sizes = sizes
        self.check(h_grid=h_grid)
        self.loc_emb = nn.Embedding(num_vocab, sum(sizes))
        self.cum_sum = [0] + np.cumsum(sizes).tolist()

    def check(self, h_grid: HierarchyGridSystem):
        if len(self.sizes) != len(h_grid.grid_id_cols):
            raise ValueError(
                "Number of columns of grid and length of sizes should be equal."
            )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.loc_emb(x)

    def hierarchical_avg(self, h_grid: HierarchyGridSystem) -> None:
        self.check(h_grid=h_grid)

        cols = h_grid.grid_id_cols
        with torch.no_grad():
            for idx, col in enumerate(cols[:-1]):
                agg_cols = cols[: idx + 1]
                col_start = self.cum_sum[idx]
                col_end = self.cum_sum[idx + 1]
                for _, sub_df in h_grid.all_grids.groupby(agg_cols):
                    row_start = sub_df.index.min()
                    row_end = sub_df.index.max() + 1
                    mean_weight = self.loc_emb.weight[
                        row_start:row_end, col_start:col_end
                    ].mean(dim=0, keepdims=True)
                    self.loc_emb.weight[row_start:row_end, col_start:col_end] = (
                        mean_weight
                    )

    @property
    def weight(self):
        return self.loc_emb.weight


class HIEREmbedding(nn.Module):
    def __init__(
        self,
        h_grid: HierarchyGridSystem,
        location_embedding_dims: List[int],
        num_vocab: int,
        week_embedding_dim: int,
        hour_embedding_dim: int,
        duration_embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.num_vocab = num_vocab
        self.loc_embedding_dim = sum(location_embedding_dims)
        self._embedding_dim = (
            self.loc_embedding_dim
            + week_embedding_dim
            + hour_embedding_dim
            + duration_embedding_dim
        )

        self.loc_emb = HIERSpatialEmbedding(
            num_vocab=num_vocab, sizes=location_embedding_dims, h_grid=h_grid
        )
        self.week_emb = nn.Embedding(7, week_embedding_dim)
        self.hour_emb = nn.Embedding(24, hour_embedding_dim)
        self.duration_emb = nn.Embedding(24, duration_embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(
        self,
        src: torch.LongTensor,
        week: torch.LongTensor,
        hour: torch.LongTensor,
        duration: torch.LongTensor,
    ):
        """
        Parameters
        ----------
        src, week, hour, duration: shape is (B, T)
        """
        token_emb = self.loc_emb(src)
        week_emb = self.week_emb(week)
        hour_emb = self.hour_emb(hour)
        duration_emb = self.duration_emb(duration)

        return self.dropout(
            torch.cat([token_emb, week_emb, hour_emb, duration_emb], dim=-1)
        )


class HIER(L.LightningModule):
    def __init__(
        self,
        tokenizer_path: str,
        hidden_size: int,
        num_layers: int,
        h_grid: HierarchyGridSystem,
        location_embedding_dims: List[int],
        week_embedding_dim: int = 4,
        hour_embedding_dim: int = 4,
        duration_embedding_dim: int = 4,
        dropout: float = 0.1,
        reduction: Union[str, LossEnum] = "mean",
    ):
        super().__init__()
        self.save_hyperparameters()

        tokenizer: AbstractTokenizer = AbstractTokenizer.load_pretrained(tokenizer_path)
        self.emb = HIEREmbedding(
            h_grid=h_grid,
            location_embedding_dims=location_embedding_dims,
            num_vocab=len(tokenizer),
            week_embedding_dim=week_embedding_dim,
            hour_embedding_dim=hour_embedding_dim,
            duration_embedding_dim=duration_embedding_dim,
            dropout=dropout,
        )
        self.rnn = nn.LSTM(
            self.emb.embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, sum(location_embedding_dims)), nn.LeakyReLU()
        )
        self._pad = tokenizer.pad

        self.loss = nn.CrossEntropyLoss(reduction="none")

        self._loss_reduction = LossEnum.parse(reduction)

    @property
    def loss_reduction(self) -> LossEnum:
        return self._loss_reduction

    def forward(
        self,
        src: torch.LongTensor,
        week: torch.LongTensor,
        hour: torch.LongTensor,
        duration: torch.LongTensor,
        lengths: List[int],
    ):
        # (B, T, C)
        emb = self.emb(src, week, hour, duration)

        inputs = pack_padded_sequence(
            emb, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        output, _ = self.rnn(inputs)

        # (B, T, C), (B,)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=self._pad
        )

        # (B, T, C)
        r = self.linear(output)

        # (B, T, V)
        return torch.matmul(r, self.emb.loc_emb.weight.swapaxes(0, 1))

    def forward_with_loss(self, batch):
        src, week, hour, duration, lengths, targets = batch

        # (B, T, C)
        emb = self.forward(src, week, hour, duration, lengths)

        # 不要计算padding部分的损失
        mask = targets != self._pad

        # (B, T)
        loss_value = mask * self.loss(
            emb.reshape(-1, emb.shape[-1]), targets.reshape(-1)
        ).reshape(emb.shape[0], emb.shape[1])

        loss = loss_value
        if self.loss_reduction == LossEnum.SUM:
            loss = loss_value.sum()
        elif self.loss_reduction == LossEnum.MEAN:
            loss = loss_value.sum() / mask.sum()

        return loss, len(lengths)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, batch_size = self.forward_with_loss(batch)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, batch_size = self.forward_with_loss(batch)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss
