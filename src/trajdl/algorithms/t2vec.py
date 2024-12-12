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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..common.samples import T2VECSample
from ..tokenizers.abstract import AbstractTokenizer
from ..utils import dist2weight, load_tokenizer
from .abstract import BaseLightningModel
from .embeddings.base import BaseTokenEmbeddingLayer, SimpleEmbedding, Word2VecEmbedding


class StackingGRU(nn.Module):
    """
    Multi-layer CRU Cell
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0
    ):
        super(StackingGRU, self).__init__()
        if num_layers <= 0:
            raise ValueError("`num_layers` must be greater than 0")
        self._num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for _ in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, src: torch.Tensor, hidden_state: torch.Tensor):
        """
        Parameters
        ----------
        src: (batch_size, input_size)
        hidden_state: (num_layers, batch_size, hidden_size)

        Output
        ----------
        output: (batch_size, hidden_size)
        hidden_states: (num_layers, batch, hidden_size)
        """

        hidden_states = []
        output = src
        for layer_idx, gru in enumerate(self.grus):
            hn_i = gru(output, hidden_state[layer_idx])
            hidden_states.append(hn_i)
            output = self.dropout(hn_i) if layer_idx != self._num_layers - 1 else hn_i

        # (num_layers, batch_size, hidden_size)
        hidden_states = torch.stack(hidden_states)
        return output, hidden_states


class GlobalAttention(nn.Module):
    r"""
    $$a = \sigma((W_1 q)H)$$
    $$c = \tanh(W_2 [a H, q])$$
    """

    def __init__(self, hidden_size: int):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query: torch.Tensor, context: torch.Tensor):
        """
        Parameters
        ----------
        query: (batch_size, hidden_size)
        context: (batch_size, seq_len, hidden_size)

        Output
        ----------
        output (batch_size, hidden_size)
        """
        # (batch_size, hidden_size) => (batch_size, hidden_size, 1)
        q1 = self.L1(query).unsqueeze(2)

        # (batch_size, seq_len)
        a = torch.bmm(context, q1).squeeze(2)

        # (batch_size, seq_len) => (batch_size, 1, seq_len)
        a = self.softmax(a).unsqueeze(1)

        # (batch_size, hidden_size)
        c = torch.bmm(a, context).squeeze(1)

        # (batch_size, hidden_size * 2)
        c = torch.cat([c, query], 1)

        # (batch_size, hidden_size)
        return self.tanh(self.L2(c))


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        embedding_layer: BaseTokenEmbeddingLayer,
        dropout: float = 0.0,
    ):
        super(DecoderWithAttention, self).__init__()
        self.embedding_layer = embedding_layer
        self.rnn = StackingGRU(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.LongTensor,
        hidden_state: torch.Tensor,
        all_encoder_hidden_states: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x: torch.LongTensor, shape is (batch_size, seq_length)
        hidden_state: (num_layers, batch_size, hidden_size)
        all_encoder_hidden_states: (batch_size, seq_len, hidden_size)

        Output
        ----------
        output: (batch_size, seq_len, hidden_size)
        """

        # (batch_size, seq_len) => (batch_size, seq_len, embedding_dim)
        embed = self.embedding_layer(x)

        output = []
        # split along the sequence length dimension
        for e in embed.split(1, dim=1):
            # (batch_size, 1, embedding_dim) => (batch_size, embedding_dim)
            e = e.squeeze(1)
            o, hidden_state = self.rnn(e, hidden_state)
            o = self.attention(o, all_encoder_hidden_states)
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output, dim=1)
        return output


class T2VECEncoder(nn.Module):
    def __init__(
        self,
        embedding_layer: BaseTokenEmbeddingLayer,
        padding_value: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if bidirectional and hidden_size % 2 != 0:
            raise ValueError(
                "`hidden_size` should be an even number greater than 0 when `bidirectional` is True"
            )

        self.emb = embedding_layer

        self._encoder_hidden_size = hidden_size // (2 if bidirectional else 1)
        self._num_layers = num_layers
        self._bidirectional_encoder = bidirectional

        self.encoder = nn.GRU(
            input_size=embedding_layer.embedding_dim,
            hidden_size=self._encoder_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self._padding_value = padding_value

    def forward(self, src: torch.LongTensor, src_lengths: List[int]):
        src = self.emb(src)

        x = pack_padded_sequence(
            input=src, lengths=src_lengths, batch_first=True, enforce_sorted=False
        )

        # all hidden states, last hidden state
        # shape of last_encoder_hidden_state is (2 * num_layers, batch_size, hidden_size // 2) if using bidirectional
        # else (num_layers, batch_size, hidden_size)
        all_encoder_hidden_states, last_encoder_hidden_state = self.encoder(x)

        # (batch_size, seq_length, hidden_size)
        all_encoder_hidden_states = pad_packed_sequence(
            all_encoder_hidden_states,
            batch_first=True,
            padding_value=self._padding_value,
        )[0]

        if self._bidirectional_encoder:
            # (num_layers, batch_size, hidden_size)
            decoder_init_hidden = (
                last_encoder_hidden_state.reshape(
                    self._num_layers, 2, -1, self._encoder_hidden_size
                )
                .swapaxes(1, 2)
                .reshape(self._num_layers, -1, 2 * self._encoder_hidden_size)
            )
        else:
            # (num_layers, batch_size, hidden_size)
            decoder_init_hidden = last_encoder_hidden_state

        # (num_layers, batch_size, hidden_size), (batch_size, seq_length, hidden_size)
        return decoder_init_hidden, all_encoder_hidden_states


class T2VEC(BaseLightningModel):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        tokenizer: Union[str, AbstractTokenizer],
        knn_indices_path: str,
        knn_distances_path: str,
        num_layers: int = 1,
        bidirectional_encoder: bool = False,
        embedding_path: Optional[str] = None,
        freeze_embedding: bool = False,
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        embedding_dim: int
            位置嵌入的size

        hidden_size: int
            RNN的隐藏状态的size

        tokenizer: Union[str, AbstractTokenizer]
            tokenizer的路径或者实例

        knn_indices_path: str
            最近邻矩阵的存储路径

        knn_distances_path: str
            最近邻距离矩阵的存储路径

        num_layers: int
            编码器和解码器的层数

        bidirectional_encoder: bool, optional
            编码器是否使用双向RNN，默认值是True

        embedding_path: Optional[str], optional
            预训练的位置嵌入的路径，默认值是None，没有传入的时候会使用SimpleEmbedding

        freeze_embedding: bool, optional
            是否冻结位置嵌入不训练，默认是False

        dropout: float, optional
            编码器和解码器使用的dropout，默认值是0.0

        """
        super().__init__()
        self.save_hyperparameters()

        if bidirectional_encoder and hidden_size % 2 != 0:
            raise ValueError(
                "`hidden_size` should be an even number greater than 0 when `bidirectional_encoder` is True"
            )

        self._bidirectional_encoder = bidirectional_encoder
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        # load location embedding
        tokenizer = load_tokenizer(tokenizer=tokenizer)
        vocab_size = len(tokenizer)
        self.embedding = (
            Word2VecEmbedding(tokenizer=tokenizer, model_path=embedding_path)
            if tokenizer and embedding_path
            else SimpleEmbedding(tokenizer=tokenizer, embedding_dim=embedding_dim)
        )

        if freeze_embedding:
            self.embedding.freeze_parameters()
        else:
            self.embedding.unfreeze_parameters()

        self.encoder = T2VECEncoder(
            embedding_layer=self.embedding,
            padding_value=tokenizer.pad,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional_encoder,
            dropout=dropout,
        )
        self.decoder = DecoderWithAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            embedding_layer=self.embedding,
            dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=vocab_size),
            nn.LogSoftmax(dim=1),
        )

        # (num_locations, k)
        V = torch.LongTensor(np.load(knn_indices_path))

        # (num_locations, k)
        D = np.load(knn_distances_path)

        dis_factor = 0.008
        D = dist2weight(D, tokenizer=tokenizer, dist_decay_speed=dis_factor)
        D = torch.FloatTensor(D)

        self.knn_indices = nn.Parameter(V, requires_grad=False)
        self.knn_distances = nn.Parameter(D, requires_grad=False)
        self.loss_fn = nn.KLDivLoss(reduction="sum")
        self._padding_value = tokenizer.pad

    def forward(self, batch: T2VECSample) -> torch.Tensor:
        """
        推理的逻辑

        Parameters
        ----------
        batch: T2VECSample

        Returns
        ----------
        torch.Tensor, shape is (batch_size, hidden_size)

        """

        # vec shape is (num_layers, batch_size, hidden_size)
        vec, _ = self.encode(batch)

        # only select hidden state of last layer
        # shape is (batch_size, hidden_size)
        return vec[-1]

    def encode(self, batch: T2VECSample) -> Tuple[torch.Tensor, torch.Tensor]:
        # (num_layers, batch_size, hidden_size), (batch_size, seq_length, hidden_size)
        return self.encoder(batch.src, batch.lengths)

    def compute_loss(self, batch: T2VECSample):
        """
        Parameters
        ----------
        batch: T2VECSample

        Returns
        ----------
        loss: torch.Tensor
            shape is (1,)

        batch_size: int

        """

        target = batch.target

        # (num_layers, batch_size, hidden_size), (batch_size, seq_length, hidden_size)
        decoder_init_hidden, all_encoder_hidden_states = self.encode(batch)

        if target.shape[1] <= 1:
            raise ValueError("seq_length of `target` shoud be greater than 1")

        # (batch_size, seq_length of target, hidden_size)
        output = self.decoder(
            target[:, :-1], decoder_init_hidden, all_encoder_hidden_states
        )

        # (batch_size * seq_length, vocab_size)
        pred = self.projector(output.reshape(-1, self._hidden_size))

        targets = target[:, 1:].reshape(
            -1,
        )

        # (batch_size * seq_length, k)
        indices = torch.index_select(self.knn_indices, dim=0, index=targets)

        # (batch_size * seq_length, k)
        output_distribution = torch.gather(pred, dim=1, index=indices)

        # (batch_size * seq_length, k)
        target_distribution = torch.index_select(
            self.knn_distances, dim=0, index=targets
        )

        loss = self.loss_fn(output_distribution, target_distribution) / batch.batch_size
        return loss, batch.batch_size

    def training_step(self, batch: T2VECSample, batch_idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
        batch: T2VECSample
        """

        loss, batch_size = self.compute_loss(batch)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch: T2VECSample, batch_idx: int) -> torch.Tensor:
        loss, batch_size = self.compute_loss(batch)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss
