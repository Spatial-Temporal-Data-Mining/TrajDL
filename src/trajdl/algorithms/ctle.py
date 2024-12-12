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

from typing import Optional, Union

import torch
from torch import nn

from ..common.enum import Mode
from ..tokenizers import AbstractTokenizer
from ..utils import load_tokenizer
from .embeddings.ctle import CTLETokenEmbeddingWithTransformer
from .framework import PretrainTrainFramework


class MaskedLM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

        self.output_size = output_size

    def forward(
        self,
        src: torch.LongTensor,
        mask: torch.BoolTensor,
        transformer_output: torch.Tensor,
    ):
        """
        src: shape is (B, T)
        mask: mask is (B, T)
        x: shape is (B, T, C)
        """

        # 使用dropout和线性变化处理transformer生成的embedding (B, T, C)，这里要映射到token上，要做分类用, (B, T, V)
        transformer_pred = self.linear(self.dropout(transformer_output))

        # shape is (M,)
        original_tokens = src[mask]

        # shape is (M, V)
        pred = transformer_pred[mask]

        # shape is (M,)
        return self.loss_func(pred, original_tokens)


class CTLETrainingFramework(PretrainTrainFramework):
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
        predictor: Optional[nn.Module] = None,
        mode: str = "pretrain",
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            mode=mode, optimizer_type=optimizer_type, learning_rate=learning_rate
        )
        self.save_hyperparameters()

        tokenizer = load_tokenizer(tokenizer=tokenizer)

        self.ctle_emb = CTLETokenEmbeddingWithTransformer(
            embedding_type=embedding_type,
            embedding_dim=embedding_dim,
            max_len=max_len,
            num_layers=num_layers,
            n_heads=n_heads,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.mlm_loss = MaskedLM(
            input_size=embedding_dim,
            output_size=len(tokenizer),
            dropout=dropout,
        )
        self.mh_loss = MaskedLM(
            input_size=embedding_dim,
            output_size=24,
            dropout=dropout,
        )

        self.predictor_loss = nn.CrossEntropyLoss()

        if not self.mode == Mode.PRETRAIN:
            if predictor is None:
                raise ValueError(
                    "predictor should not be None when mode is set to 'pretrain'"
                )

        self.predictor = predictor

    def init_from_pretrained_ckpt(self, ckpt_folder: str):
        if self.mode == Mode.TRAIN:
            pass

    def forward(
        self,
        loc_src: torch.LongTensor,
        ts_src: torch.LongTensor,
        mask: torch.BoolTensor,
    ):
        if self.mode == Mode.PRETRAIN:
            return self.ctle_emb(loc_src=loc_src, ts_src=ts_src, mask=mask)
        elif self.mode == Mode.TRAIN:
            raise NotImplementedError("waiting for implementation")

    def compute_loss(
        self,
        loc_src: torch.LongTensor,
        ts_src: torch.LongTensor,
        mask: torch.BoolTensor,
    ):
        # shape is (B, T, C)
        output = self.ctle_emb(loc_src=loc_src, ts_src=ts_src, mask=mask)

        mlm_loss = self.mlm_loss(loc_src, mask, output)

        # torch.LongTensor, shape is (B, T)
        hour_src = ts_src % (24 * 60 * 60) // 3600

        mh_loss = self.mh_loss(hour_src, mask, output)

        if self.mode == Mode.PRETRAIN:
            return mlm_loss + mh_loss
        elif self.mode == Mode.TRAIN:
            self.predictor(
                src=loc_src,
            )
            raise ValueError("mode only support {'pretrain'}")

    def training_step(self, batch, batch_idx: int):
        loc_src, ts_src, mask = batch
        loss = self.compute_loss(loc_src, ts_src, mask)
        self.log("train_loss", loss, batch_size=loc_src.shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int):
        loc_src, ts_src, mask = batch
        loss = self.compute_loss(loc_src, ts_src, mask)
        self.log("val_loss", loss, batch_size=loc_src.shape[0])
        return loss
