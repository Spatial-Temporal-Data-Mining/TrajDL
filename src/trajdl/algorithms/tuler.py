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

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..common.samples import TULERSample
from ..metrics.tul import TULMetrics
from ..tokenizers.abstract import AbstractTokenizer
from ..utils import load_tokenizer
from .abstract import BaseLightningModel
from .embeddings.base import SimpleEmbedding, Word2VecEmbedding
from .loc_pred.rnn import select_last_k_elements


class TULER(BaseLightningModel):
    """
    TULER: Identifying Human Mobility via Trajectory Embeddings (IJCAI 2017)
    """

    def __init__(
        self,
        tokenizer: Union[str, Path, AbstractTokenizer],
        num_users: int,
        embedding_dim: int,
        hidden_dim: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.5,
        embedding_path: Optional[str] = None,
        freeze_embedding: bool = False,
        bidirectional: bool = True,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        topk: int = 5,
    ):
        """
        Parameters
        ----------
        tokenizer: Union[str, Path, AbstractTokenizer]
            tokenizer的路径或者tokenizer的实例

        num_users: int
            TUL任务中的用户数

        embedding_dim: int
            位置嵌入层的维数

        hidden_dim: int
            RNN隐藏单元的维数

        rnn_type: str, optional
            RNN的类型，目前只支持{'lstm', 'gru'}，默认值是'lstm'

        num_layers: int, optional
            RNN的层数，默认值是1

        dropout: float, optional
            RNN的输出会经过dropout，dropout参数，默认值是0.5

        embedding_path: Optional[str], optional
            预训练的位置嵌入模型的路径，默认值是None

        freeze_embedding: bool, optional
            是否冻结预训练的位置嵌入，默认值是False

        bidirectional: bool, optional
            是否使用双向RNN，默认值是True

        optimizer_type: str, optional
            优化器，默认值是'adam'

        learning_rate: float, optional
            学习率，默认值是1e-3

        topk: int, optional
            评估的时候需要评估acc@k的k值，默认值是5

        """
        super().__init__(optimizer_type=optimizer_type, learning_rate=learning_rate)
        self.save_hyperparameters()

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError("`rnn_type` only support {'lstm', 'gru'}")

        self.metrics = TULMetrics(
            num_users=num_users, topk=topk if num_users > topk else num_users - 1
        )

        self.n_hidden = hidden_dim

        tokenizer = load_tokenizer(tokenizer=tokenizer)

        # load location embedding
        self.embedding = (
            Word2VecEmbedding(tokenizer=tokenizer, model_path=embedding_path)
            if tokenizer and embedding_path
            else SimpleEmbedding(tokenizer=tokenizer, embedding_dim=embedding_dim)
        )

        if freeze_embedding:
            self.embedding.freeze_parameters()
        else:
            self.embedding.unfreeze_parameters()

        # a bias vector of each LSTM layer is structured like this:
        # [b_ig | b_fg | b_gg | b_og]
        # init bias of forget gate as 1
        for names in self.rnn._all_weights:
            for name in filter(lambda name: "bias" in name, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_users)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()
        self._padding_value = tokenizer.pad

    def forward(self, sample: TULERSample) -> torch.Tensor:
        """
        推理的过程，给定多条序列和实际长度，返回这些序列在预测用户时的logits

        Parameters
        ----------
        sample: TULERSample

        Returns
        ----------
        output: torch.Tensor
            shape is (B, num_users), 每条序列的logits

        """
        # embed token, (B, T, C)
        inputs = self.embedding(sample.src)

        # pack
        data = pack_padded_sequence(inputs, sample.seq_len, batch_first=True)

        # lstm
        outputs, _ = self.rnn(data)

        # pad pack, out_pad shape is (B, T, C)
        out_pad, lengths = pad_packed_sequence(
            outputs, batch_first=True, padding_value=self._padding_value
        )

        # dropout, (B, T, C)
        outputs = self.dropout(out_pad)

        # select last time step for each sample, (B, num_users)
        return self.fc(
            select_last_k_elements(outputs, lengths=sample.seq_len, k=1).squeeze(dim=1)
        )

    def compute_loss(self, batch: TULERSample, return_prediction: bool = False):
        # (B, num_users)
        outputs = self.forward(batch)

        # (1,)
        loss = self.loss(outputs, batch.labels)

        if return_prediction:
            return loss, batch.batch_size, outputs

        return loss, batch.batch_size

    def training_step(self, batch: TULERSample, batch_idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
        batch: TULERSample
            样本

        batch_idx: int
            lightning框架需要的batch_idx

        Returns
        ----------
        loss: torch.Tensor
            shape is (1,)

        """
        loss, batch_size = self.compute_loss(batch)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch: TULERSample, batch_idx: int) -> torch.Tensor:
        loss, batch_size, outputs = self.compute_loss(batch, return_prediction=True)
        self.log("val_loss", loss, batch_size=batch_size)
        self.metrics.update(
            preds=outputs.detach().cpu().numpy(),
            targets=batch.labels.detach().cpu().numpy(),
        )
        return loss

    def test_step(self, batch: TULERSample, batch_idx: int) -> None:
        outputs = self.forward(batch)
        self.metrics.update(
            preds=outputs.detach().cpu().numpy(),
            targets=batch.labels.detach().cpu().numpy(),
        )

    def on_validation_epoch_start(self):
        self.metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.metrics.value()
        for key, value in metrics.items():
            self.log(f"val_{key}", value, batch_size=1)

    def on_test_epoch_start(self):
        self.metrics.reset()

    def on_test_epoch_end(self):
        self.metrics.value()
