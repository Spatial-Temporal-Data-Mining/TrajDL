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

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from ...common.enum import LossEnum
from ...common.samples import STLSTMSample
from ...loss.sampled_softmax import SampledSoftmaxLoss
from ...metrics.acc import AccMetrics
from ...tokenizers import AbstractTokenizer
from ...tokenizers.slot import Bucketizer
from ...utils import load_bucketizer, load_tokenizer
from ..abstract import BaseLightningModel
from ..embeddings.base import BaseTokenEmbeddingLayer, SimpleEmbedding
from .rnn import select_last_k_elements


class STLSTM(nn.Module):
    def __init__(
        self,
        tokenizer: AbstractTokenizer,
        embedding_dim: int,
        hidden_size: int,
        ts_bucketizer: Bucketizer,
        loc_bucketizer: Bucketizer,
        loc_emb_layer: Optional[BaseTokenEmbeddingLayer] = None,
    ):
        super().__init__()
        self.loc_emb = (
            loc_emb_layer
            if loc_emb_layer is not None
            else SimpleEmbedding(tokenizer=tokenizer, embedding_dim=embedding_dim)
        )
        self.temporal_upper_emb = nn.Embedding(
            num_embeddings=ts_bucketizer.num_buckets, embedding_dim=embedding_dim
        )
        self.temporal_lower_emb = nn.Embedding(
            num_embeddings=ts_bucketizer.num_buckets, embedding_dim=embedding_dim
        )
        self.spatial_upper_emb = nn.Embedding(
            num_embeddings=loc_bucketizer.num_buckets, embedding_dim=embedding_dim
        )
        self.spatial_lower_emb = nn.Embedding(
            num_embeddings=loc_bucketizer.num_buckets, embedding_dim=embedding_dim
        )
        self.temporal_ln = nn.Linear(
            in_features=embedding_dim, out_features=3 * hidden_size, bias=False
        )
        self.spatial_ln = nn.Linear(
            in_features=embedding_dim, out_features=3 * hidden_size, bias=False
        )
        self.input_weight = nn.Linear(
            in_features=embedding_dim, out_features=4 * hidden_size, bias=True
        )
        self.hidden_weight = nn.Linear(
            in_features=hidden_size, out_features=4 * hidden_size, bias=False
        )
        self._hidden_size = hidden_size
        self._td_upper = ts_bucketizer.upper_bound
        self._td_lower = ts_bucketizer.lower_bound
        self._sd_upper = loc_bucketizer.upper_bound
        self._sd_lower = loc_bucketizer.lower_bound

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def td_upper(self) -> float:
        return self._td_upper

    @property
    def td_lower(self) -> float:
        return self._td_lower

    @property
    def sd_upper(self) -> float:
        return self._sd_upper

    @property
    def sd_lower(self) -> float:
        return self._sd_lower

    def cell_step(
        self,
        loc: torch.LongTensor,
        td_upper: torch.LongTensor,
        td_lower: torch.LongTensor,
        sd_upper: torch.LongTensor,
        sd_lower: torch.LongTensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        first_step: bool = False,
    ):
        """
        loc, shape is (B,)，位置batch
        td_upper, td_lower: shape is (B,)，这个是两个时间步之间的时间差被上界减去和减去下界的batch
        sd_upper, sd_lower: shape is (B,)，这个是两个时间步之间的空间位移被上界减去和减去下界的batch
        hidden: [hidden, cell], shape: (B, H), (B, H)
        first_step: bool, optional
            default False
        """

        # (B, C)
        loc_emb = self.loc_emb(loc)

        # (B, 4 * H)
        tmp = self.input_weight(loc_emb) + self.hidden_weight(hidden[0])

        # (B, H), (B, H), (B, H), (B, H)
        i_t, f_t, o_t, g_t = tmp.chunk(4, 1)

        # (B, C)
        q = (self.temporal_upper_emb(td_upper) + self.temporal_lower_emb(td_lower)) / (
            self.td_upper - self.td_lower
        )
        if first_step:
            q = torch.zeros_like(q, device=q.device)

        # (B, C)
        s = (self.spatial_upper_emb(sd_upper) + self.spatial_lower_emb(sd_lower)) / (
            self.sd_upper - self.sd_lower
        )
        if first_step:
            s = torch.zeros_like(s, device=s.device)

        # (B, 3 * H)
        F = self.temporal_ln(q) + self.spatial_ln(s)

        # (B, H), (B, H), (B, H)
        i_F, f_F, o_F = F.chunk(3, 1)

        i_t = torch.sigmoid(i_t + i_F)
        f_t = torch.sigmoid(f_t + f_F)
        o_t = torch.sigmoid(o_t + o_F)
        g_t = torch.tanh(g_t)

        # (B, H)
        c_t = f_t * hidden[1] + i_t * g_t

        # (B, H)
        h_t = o_t * torch.tanh(c_t)

        # (B, H), (B, H)
        return h_t, c_t

    def forward(
        self,
        sample: STLSTMSample,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        sample: STLSTMSample
            输入序列
        hidden: Tuple[torch.Tensor, torch.Tensor], optional
            (hidden, cell), shape: (B, H), (B, H)
        """
        batch_size, num_timesteps = sample.loc_seq.shape

        hidden = (
            hidden
            if hidden
            else self.init_hidden(batch_size=batch_size, device=sample.loc_seq.device)
        )

        all_hiddens = []
        for ts_idx in range(num_timesteps):
            hidden = self.cell_step(
                loc=sample.loc_seq[:, ts_idx],
                td_upper=sample.td_upper_seq[:, ts_idx],
                td_lower=sample.td_lower_seq[:, ts_idx],
                sd_upper=sample.sd_upper_seq[:, ts_idx],
                sd_lower=sample.sd_lower_seq[:, ts_idx],
                hidden=hidden,
                first_step=ts_idx == 0,
            )

            all_hiddens.append(hidden)

        # (B, T + 1, H)
        all_h = torch.stack([hidden[0]] + [h for h, _ in all_hiddens], dim=1)

        # (B, T + 1, H)
        all_c = torch.stack([hidden[1]] + [c for _, c in all_hiddens], dim=1)

        valid_lengths_p1 = [length + 1 for length in sample.valid_lengths]

        last_h = select_last_k_elements(x=all_h, lengths=valid_lengths_p1, k=1).squeeze(
            dim=1
        )
        last_c = select_last_k_elements(x=all_c, lengths=valid_lengths_p1, k=1).squeeze(
            dim=1
        )
        return all_h[:, 1:], (last_h, last_c)

    def init_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(size=(batch_size, self.hidden_size), device=device),
            torch.zeros(size=(batch_size, self.hidden_size), device=device),
        )


class STLSTMModule(BaseLightningModel):
    def __init__(
        self,
        tokenizer: Union[str, AbstractTokenizer],
        embedding_dim: int,
        hidden_size: int,
        ts_bucketizer: Union[str, Bucketizer],
        loc_bucketizer: Union[str, Bucketizer],
        reduction: Union[str, LossEnum] = "mean",
        use_sampled_softmax: bool = True,
        num_neg_samples: int = 64,
        loc_emb_layer: Optional[BaseTokenEmbeddingLayer] = None,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
    ):
        super().__init__(optimizer_type=optimizer_type, learning_rate=learning_rate)
        self.save_hyperparameters()

        self.metrics = AccMetrics()

        tokenizer = load_tokenizer(tokenizer=tokenizer)
        self._hidden_size = hidden_size

        if loc_emb_layer:
            embedding_dim = loc_emb_layer.embedding_dim

        self.stlstm = STLSTM(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            ts_bucketizer=load_bucketizer(ts_bucketizer),
            loc_bucketizer=load_bucketizer(loc_bucketizer),
            loc_emb_layer=loc_emb_layer,
        )

        self._loss_reduction = LossEnum.parse(reduction)

        self._num_locs = len(tokenizer)
        self._use_sampled_softmax = use_sampled_softmax

        if use_sampled_softmax:
            self.w = torch.nn.Parameter(
                torch.randn(self._num_locs, hidden_size) / np.sqrt(hidden_size)
            )
            self.b = torch.nn.Parameter(torch.zeros(size=(self._num_locs, 1)))
            self.loss = SampledSoftmaxLoss(
                weights=self.w,
                bias=self.b,
                num_words=self._num_locs,
                num_samples=num_neg_samples,
                reduction=self._loss_reduction.value,
                use_sampled_softmax_in_eval=False,
            )
        else:
            self.projector = nn.Linear(
                in_features=hidden_size, out_features=len(tokenizer)
            )
            self.loss = nn.CrossEntropyLoss(reduction="none")

    @property
    def use_sampled_softmax(self) -> bool:
        return self._use_sampled_softmax

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_locs(self) -> int:
        return self._num_locs

    @property
    def loss_reduction(self) -> LossEnum:
        return self._loss_reduction

    def encode(self, sample: STLSTMSample):
        """
        以0为隐藏状态和细胞状态作为初始化，将一个session的前N-1个位置作为输入，输出后N-1个位置的隐藏状态

        Parameters
        ----------
        loc_seq: torch.LongTensor
            位置序列

        Returns
        ----------
        output: torch.Tensor
            shape is (B, T, H)

        """
        # output shape is (B, T, H)
        output, _ = self.stlstm(sample)

        return output

    def forward(self, sample: STLSTMSample, k: int = 1):
        """
        推理，选择最后k个时间步的输出

        Parameters
        ----------
        sample: STLSTMSample
            推理时的样本

        k: int, optional
            取每条序列的最后k个时间步进行输出，默认值是1

        Returns
        ----------
        torch.Tensor, shape is (B, num_locs)

        """
        # shape is (B, T, H)
        output = self.encode(sample)

        valid_lengths = sample.valid_lengths

        # shape is (B, H)
        last_pred = select_last_k_elements(
            x=output, lengths=valid_lengths, k=k
        ).squeeze(dim=1)

        if self.use_sampled_softmax:
            # (B, num_locs)
            last_pred = torch.matmul(
                last_pred, self.w.transpose(0, 1)
            ) + self.b.transpose(0, 1)
        else:
            # (B, num_locs)
            last_pred = self.projector(last_pred)

        # (B, num_locs)
        return torch.softmax(last_pred, dim=-1)

    def compute_loss(self, sample: STLSTMSample):
        """
        计算一个batch的loss

        Parameters
        ----------
        sample: STLSTMSample
            需要计算损失的样本
        """
        # (B, T, H)
        hidden = self.encode(sample)

        # (B, T)
        label = sample.labels
        batch_size = sample.batch_size

        if self.use_sampled_softmax:
            loss = self.loss(
                hidden.reshape(-1, self.hidden_size),
                label.reshape(
                    -1,
                ),
                sample.mask.reshape(
                    -1,
                ),
            )

            return loss, batch_size
        else:
            # (B, T)
            loss = (
                self.loss(
                    self.projector(hidden).reshape(-1, self.num_locs),
                    label.reshape(
                        -1,
                    ),
                ).reshape(label.shape)
                * sample.mask
            )

            if self.loss_reduction == LossEnum.SUM:
                loss = loss.sum()
            elif self.loss_reduction == LossEnum.MEAN:
                loss = loss.sum() / sample.mask.sum()

            return loss, batch_size

    def training_step(self, batch: STLSTMSample, batch_idx: int):
        loss, batch_size = self.compute_loss(batch)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch: STLSTMSample, batch_idx: int) -> torch.Tensor:
        loss, batch_size = self.compute_loss(batch)
        self.log("val_loss", loss, batch_size=batch_size)

        # (B,)
        pred = self.forward(sample=batch).argmax(dim=-1)
        # (B,)
        label = select_last_k_elements(batch.labels, batch.valid_lengths, k=1).squeeze(
            dim=1
        )
        self.metrics.update(pred=pred, label=label)
        return loss

    def test_step(self, batch: STLSTMSample, batch_idx: int) -> None:
        # (B,)
        pred = self.forward(sample=batch).argmax(dim=-1)
        # (B,)
        label = select_last_k_elements(batch.labels, batch.valid_lengths, k=1).squeeze(
            dim=1
        )
        self.metrics.update(pred=pred, label=label)

    def on_validation_epoch_start(self):
        self.metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.metrics.value()
        for key, value in metrics.items():
            self.log(key, value, batch_size=1)

    def on_test_epoch_start(self):
        self.metrics.reset()

    def on_test_epoch_end(self):
        self.metrics.value()


class HSTLSTM(BaseLightningModel):
    def __init__(
        self,
        tokenizer: AbstractTokenizer,
        embedding_dim: int,
        hidden_size: int,
        ts_buckets: Bucketizer,
        loc_buckets: Bucketizer,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
    ):
        super().__init__(optimizer_type=optimizer_type, learning_rate=learning_rate)
        self.save_hyperparameters()

        self.stlstm_encoder = STLSTM(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            ts_bucketizer=ts_buckets,
            loc_bucketizer=loc_buckets,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        self.stlstm_decoder = STLSTM(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            ts_bucketizer=ts_buckets,
            loc_bucketizer=loc_buckets,
        )

    def forward(
        self,
        loc_sessions: List[torch.LongTensor],
        ts_upper_sessions: List[torch.LongTensor],
        ts_lower_sessions: List[torch.LongTensor],
        sd_upper_sessions: List[torch.LongTensor],
        sd_lower_sessions: List[torch.LongTensor],
        valid_lengths: List[List[int]],
    ):
        # session的个数
        num_sessions = len(loc_sessions)

        for session_idx in range(num_sessions - 1):
            pass

    def training_step(self, batch, batch_idx: int):
        pass
