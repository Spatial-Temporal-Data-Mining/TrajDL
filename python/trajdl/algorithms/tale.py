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

import os
import pickle
from typing import ClassVar, Optional, Tuple, Union

import torch
from torch import nn

from ..common.enum import Mode
from ..common.samples import NextLocPredSample, TALEPretrainSample
from ..tokenizers import AbstractTokenizer
from ..utils import find_best_checkpoint, load_tokenizer
from .embeddings.tale import SliceHuffmanForest, TaleEmbedding
from .framework import PretrainTrainFramework


class TaleTrainingFramework(PretrainTrainFramework):
    """
    TALE training framework.

    In ``pretrain`` mode it trains the time-aware hierarchical-softmax
    embedding from ``TaleDataModule`` batches; in ``train`` mode it loads
    the token embedding from a pre-train checkpoint and finetunes it
    together with an injected next-location predictor (the baseline
    ``train_next_loc`` also optimizes the embedding jointly).

    Parameters
    ----------
    tokenizer : Union[str, AbstractTokenizer]
        Path of the tokenizer or a tokenizer instance.
    embedding_dim : int
    forest_path : str, optional
        Path to the pickled ``SliceHuffmanForest`` built by the offline
        preprocessing step. Only loaded in ``pretrain`` mode; the downstream
        ``train`` mode uses the static token lookup exclusively and never
        touches the forest, so finetune/eval can run without ``forest.pkl``.
    mode : str, optional
        "pretrain" or "train".
    predictor : Optional[nn.Module], optional
        Injected by configuration in "train" mode; TALE itself does not
        depend on any concrete predictor implementation. If the predictor
        exposes an ``embedding_layer`` attribute that is ``None``, it is
        wired to this framework's ``tale_emb``.
    pre_len : int, optional
        Number of future locations to predict in "train" mode.
    pretrained_ckpt_folder : Optional[str], optional
        Folder containing the pre-train stage checkpoints; in "train" mode the
        best (lowest val_loss) checkpoint's ``tale_emb.token_emb`` weights are
        loaded into the embedding. Typically injected on the command line after
        the pre-train stage finishes.
    freeze_embedding : bool, optional
        In "train" mode, freeze ``tale_emb.token_emb`` after loading the
        pre-trained weights so only the predictor is trained. This matches the
        author's downstream protocol: ``StaticEmbed`` wraps the pretrained
        matrix with ``requires_grad=False`` and detaches its output, so the
        released TALE code never updates the embedding during finetune
        (only the ``embed=fc`` trainable baseline learns it jointly).
        By default False, which keeps the previous TrajDL behavior of
        finetuning the embedding jointly.
    """

    path_params: ClassVar[tuple[str, ...]] = (
        "tokenizer",
        "forest_path",
        "pretrained_ckpt_folder",
    )

    def __init__(
        self,
        tokenizer: Union[str, AbstractTokenizer],
        embedding_dim: int,
        forest_path: Optional[str] = None,
        mode: str = "pretrain",
        predictor: Optional[nn.Module] = None,
        pre_len: int = 3,
        pretrained_ckpt_folder: Optional[str] = None,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        freeze_embedding: bool = False,
    ):
        super().__init__(
            mode=mode, optimizer_type=optimizer_type, learning_rate=learning_rate
        )
        self.save_hyperparameters()

        tokenizer = load_tokenizer(tokenizer=tokenizer)
        forest = None
        if self.mode == Mode.PRETRAIN:
            if forest_path is None:
                raise ValueError("forest_path is required in 'pretrain' mode")
            with open(forest_path, "rb") as f:
                forest: SliceHuffmanForest = pickle.load(f)

        self.tale_emb = TaleEmbedding(
            tokenizer=tokenizer, embedding_dim=embedding_dim, forest=forest
        )

        if self.mode == Mode.TRAIN and predictor is None:
            raise ValueError("predictor should not be None when mode is set to 'train'")

        self.predictor = predictor
        self.pre_len = pre_len
        self.predictor_loss = nn.CrossEntropyLoss()

        if self.mode == Mode.TRAIN and self.predictor is not None:
            if (
                hasattr(self.predictor, "embedding_layer")
                and self.predictor.embedding_layer is None
            ):
                self.predictor.embedding_layer = self.tale_emb
            if pretrained_ckpt_folder is not None:
                self.init_from_pretrained_ckpt(pretrained_ckpt_folder)
            if freeze_embedding:
                self.tale_emb.token_emb.embedding.weight.requires_grad_(False)
                # predictor 自带独立嵌入层时（word2vec/random 基线），作者协议的
                # 冻结语义是"下游只用预训练嵌入、只训预测头"，必须一并冻结
                if (
                    hasattr(self.predictor, "embedding_layer")
                    and self.predictor.embedding_layer is not None
                    and self.predictor.embedding_layer is not self.tale_emb
                ):
                    for param in self.predictor.embedding_layer.parameters():
                        param.requires_grad_(False)

    def init_from_pretrained_ckpt(self, ckpt_folder: str):  # type: ignore[override]
        """
        Load the pre-trained token embedding (``tale_emb.token_emb``) from the
        best checkpoint of the pre-train stage.

        Only ``token_emb`` weights are loaded: ``inner_emb`` / ``slot_emb``
        are pre-train-only auxiliary parameters and are unused downstream.
        """
        if self.mode == Mode.TRAIN:
            ckpt_filename = find_best_checkpoint(ckpt_folder, is_maximizing=False)
            ckpt_path = os.path.join(ckpt_folder, ckpt_filename)
            print(f"load tale_emb.token_emb weights from {ckpt_path}")
            # torch>=2.6 默认 weights_only=True，无法反序列化 Lightning ckpt
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            token_emb_state = {
                key[len("tale_emb.token_emb.") :]: value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("tale_emb.token_emb.")
            }
            self.tale_emb.token_emb.load_state_dict(token_emb_state)

    def forward(self, sample: NextLocPredSample) -> torch.Tensor:
        """Downstream inference: next-location logits from the injected predictor."""
        if self.mode != Mode.TRAIN or self.predictor is None:
            raise ValueError("forward inference requires mode='train' and a predictor")
        return self.predictor(
            sample.full_seq, sample.valid_len, sample.pre_len, sample.ts_src
        )

    def compute_loss(
        self, batch: Union[TALEPretrainSample, NextLocPredSample]
    ) -> Tuple[torch.Tensor, int]:
        if self.mode == Mode.PRETRAIN:
            loss = self.tale_emb.compute_pretrain_loss(
                {
                    "pos_u": batch.pos_u,
                    "pos_w": batch.pos_w,
                    "neg_w": batch.neg_w,
                    "pos_s": batch.pos_s,
                    "prob": batch.prob,
                }
            )
            return loss, batch.pos_u.shape[0]
        elif self.mode == Mode.TRAIN:
            logits = self.predictor(
                batch.full_seq, batch.valid_len, batch.pre_len, batch.ts_src
            )
            loss = self.predictor_loss(
                logits.reshape(-1, logits.size(-1)),
                batch.trg_seq.reshape(-1),
            )
            return loss, batch.full_seq.shape[0]
        raise ValueError(f"mode only support {{'pretrain', 'train'}}, got {self.mode}")

    def training_step(self, batch, batch_idx: int):
        loss, batch_size = self.compute_loss(batch)
        # 预训练验证集可能整批被 skip_missing 跳空，batch_size=0 会让
        # Lightning 的加权平均变成 0/0=nan
        self.log("train_loss", loss, batch_size=max(1, batch_size))
        return loss

    def _next_loc_acc1(self, batch: NextLocPredSample) -> torch.Tensor:
        """Validation-stage acc@1 (single-math principle): the author-protocol
        model-selection metric (next_loc.py accuracy_score).
        """
        logits = self.predictor(
            batch.full_seq, batch.valid_len, batch.pre_len, batch.ts_src
        )
        return (logits.argmax(-1) == batch.trg_seq).float().mean()

    def validation_step(self, batch, batch_idx: int):
        loss, batch_size = self.compute_loss(batch)
        self.log("val_loss", loss, batch_size=max(1, batch_size))
        if self.mode == Mode.TRAIN:
            self.log(
                "val_acc1", self._next_loc_acc1(batch), batch_size=max(1, batch_size)
            )
        return loss
