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

"""TrajSimi model and Lightning training module.

This module provides:
- TrajSimiRegression: A regression head for trajectory similarity learning.
- TrajSimiLightning: Lightning module for fine-tuning TrajCL on downstream similarity tasks.
"""

from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.samples import TrajSimiSample
from ...tokenizers.trajcl import TrajCLTokenizer
from ..abstract import BaseLightningModel
from .model import TrajCLLightning


class TrajSimiRegression(nn.Module):
    """A simple MLP regression head for transforming trajectory embeddings.
    Matches the original TrajCL TrajSimiRegression implementation.

    Parameters
    ----------
    nin : int
        Input and output embedding dimension.
    """

    def __init__(self, nin: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nin, nin), nn.ReLU(), nn.Linear(nin, nin))

    def forward(self, trajs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        trajs : torch.Tensor
            Trajectory embeddings, shape (batch_size, emb_dim).

        Returns
        -------
        torch.Tensor
            Transformed and L2-normalized embeddings, shape (batch_size, emb_dim).
        """
        return F.normalize(self.enc(trajs), dim=1)


class TrajSimiLightning(BaseLightningModel):
    """Lightning module for training TrajSimi.

    Parameters
    ----------
    pretrained_ckpt_path : str
        Path to the pretrained TrajCLLightning checkpoint.
    tokenizer_path : str, optional
        Path to the tokenizer file. Required for loading the TrajCL checkpoint
        since tokenizer is excluded from saved hyperparameters.
    cell_embeddings_path : str, optional
        Path to cell embeddings .pt file. Required for loading the TrajCL
        checkpoint since cell_embeddings is excluded from saved hyperparameters.
    embedding_dim : int
        Dimension of cell embeddings (should match pretrained model).
    optimizer_type : str, optional
        Optimizer type, by default "adam".
    learning_rate : float, optional
        Learning rate, by default 1e-3.
    weight_decay : float, optional
        Weight decay, by default 0.0001.
    """

    path_params: ClassVar[tuple[str, ...]] = (
        "pretrained_ckpt_path",
        "tokenizer_path",
        "cell_embeddings_path",
    )

    def __init__(
        self,
        pretrained_ckpt_path: str,
        tokenizer_path: str | None = None,
        cell_embeddings_path: str | None = None,
        embedding_dim: int = 256,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0001,
    ):
        super().__init__(optimizer_type=optimizer_type, learning_rate=learning_rate)
        self.save_hyperparameters(ignore=["tokenizer_path", "cell_embeddings_path"])

        self._weight_decay = weight_decay

        # Load tokenizer and cell_embeddings for TrajCLLightning.load_from_checkpoint
        # These are excluded from saved hyperparameters in TrajCLLightning, so must be
        # provided explicitly when loading.
        load_kwargs = {}

        if tokenizer_path is not None:
            load_kwargs["tokenizer"] = TrajCLTokenizer.load_pretrained(tokenizer_path)

        if cell_embeddings_path is not None:
            load_kwargs["cell_embeddings"] = torch.load(
                cell_embeddings_path, weights_only=True
            )
            # Override cell_embeddings_path so the loaded model doesn't try to
            # load it again from the stale path saved in the checkpoint
            load_kwargs["cell_embeddings_path"] = None

        # Load pre-trained TrajCL model
        trajcl_lightning = TrajCLLightning.load_from_checkpoint(
            pretrained_ckpt_path, **load_kwargs
        )

        # We only need the embedding layer and the encoder_q
        self.embedding = trajcl_lightning.embedding
        self.encoder_q = trajcl_lightning.model.encoder_q

        # Regression head
        self.regression_head = TrajSimiRegression(embedding_dim)

        self.criterion = nn.MSELoss()

        # For validation
        self.validation_step_outputs = []

    def forward(self, batch: TrajSimiSample) -> torch.Tensor:
        """Encode trajectories for inference.

        Parameters
        ----------
        batch : TrajSimiSample
            Input batch.

        Returns
        -------
        torch.Tensor
            Trajectory embeddings.
        """
        # Get cell embeddings: (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        emb = self.embedding(batch.cell_seq)

        # Transpose to (seq_len, batch_size, emb_dim) for transformer
        cell_emb = emb.transpose(0, 1)

        # Create padding mask
        max_len = cell_emb.shape[0]
        lengths_tensor = torch.tensor(batch.lengths, device=cell_emb.device)
        src_padding_mask = (
            torch.arange(max_len, device=cell_emb.device)[None, :]
            >= lengths_tensor[:, None]
        )

        # Encode using query encoder
        traj_emb = self.encoder_q(
            src=cell_emb,
            attn_mask=None,
            src_padding_mask=src_padding_mask,
            src_len=lengths_tensor,
            srcspatial=batch.spatial_feat.transpose(0, 1),
        )

        # Pass through regression head
        return self.regression_head(traj_emb)

    def compute_loss(self, batch: TrajSimiSample):
        """Regression loss of the sub_simi targets against the in-batch pairwise L1 distance matrix (strict upper triangle).

        Returns
        -------
        (loss, batch_size)
        """
        outs = self(batch)
        pred_l1_simi = torch.cdist(outs, outs, p=1.0)
        # 严格上三角（对角线与下三角为自比较/对称冗余）
        mask = torch.triu(torch.ones_like(pred_l1_simi), diagonal=1) == 1
        loss = self.criterion(pred_l1_simi[mask], batch.sub_simi[mask])
        return loss, batch.batch_size

    def training_step(self, batch: TrajSimiSample, batch_idx: int) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : TrajSimiSample
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        loss, batch_size = self.compute_loss(batch)
        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: TrajSimiSample, batch_idx: int) -> None:
        """Validation step. Collects embeddings for global metric computation.

        Parameters
        ----------
        batch : TrajSimiSample
            Input batch.
        batch_idx : int
            Batch index.
        """
        outs = self(batch)
        self.validation_step_outputs.append(outs)

    def on_validation_epoch_end(self) -> None:
        """Compute global metrics (HR@5, HR@20) at the end of validation epoch."""
        if not self.validation_step_outputs:
            return

        traj_outs = torch.cat(self.validation_step_outputs, dim=0)
        self.validation_step_outputs.clear()

        # Compute global predicted distance matrix
        pred_l1_simi = torch.cdist(traj_outs, traj_outs, p=1.0)

        try:
            val_simi_matrix = self.trainer.datamodule.val_simi_matrix
        except AttributeError:
            # If not available, skip metric computation
            return

        truth_l1_simi = val_simi_matrix.to(pred_l1_simi.device)

        # Compute loss on the whole matrix
        mask = torch.triu(torch.ones_like(pred_l1_simi), diagonal=1) == 1
        pred_l1_simi_seq = pred_l1_simi[mask]
        truth_l1_simi_seq = truth_l1_simi[mask]

        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Compute Hitting Ratios
        hr5 = self.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
        hr20 = self.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
        hr20_in_5 = self.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 5)

        self.log("val_hr5", hr5, prog_bar=True, sync_dist=True)
        self.log("val_hr20", hr20, prog_bar=True, sync_dist=True)
        self.log("val_hr20_in_5", hr20_in_5, prog_bar=True, sync_dist=True)

    @staticmethod
    def hitting_ratio(
        preds: torch.Tensor, truths: torch.Tensor, pred_topk: int, truth_topk: int
    ) -> float:
        """Calculate hitting ratio.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted distance matrix, shape (N, N)
        truths : torch.Tensor
            Ground truth distance matrix, shape (N, N)
        pred_topk : int
            Top K for predictions
        truth_topk : int
            Top K for ground truth

        Returns
        -------
        float
            Hitting ratio
        """
        assert preds.shape == truths.shape
        assert pred_topk < preds.shape[1] and truth_topk < preds.shape[1]

        _, preds_k_idx = torch.topk(preds, pred_topk + 1, dim=1, largest=False)
        _, truths_k_idx = torch.topk(truths, truth_topk + 1, dim=1, largest=False)

        preds_k_idx = preds_k_idx.cpu().numpy()
        truths_k_idx = truths_k_idx.cpu().numpy()

        tp = 0
        N = preds.shape[0]
        for i in range(N):
            tp += np.intersect1d(preds_k_idx[i], truths_k_idx[i]).size

        return (tp - N) / (truth_topk * N)

    def configure_optimizers(self):
        """Configure optimizer and lr scheduler."""
        params = [
            {
                "params": self.regression_head.parameters(),
                "lr": self.learning_rate,
                "weight_decay": self._weight_decay,
            },
            {
                "params": self.encoder_q.parameters(),
                "lr": self.learning_rate,
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(params)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(params)
        elif self.optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(params)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
