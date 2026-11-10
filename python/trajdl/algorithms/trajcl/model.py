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

"""TrajCL model and Lightning training module.

This module provides:
- TrajCL: The main model combining DualSTB encoders with MoCo framework
- TrajCLLightning: Lightning module for training with TrajCLDataModule
"""

from typing import ClassVar, List, Tuple, Union

import torch
import torch.nn as nn

from ...common.samples import TrajCLSample
from ...tokenizers.abstract import AbstractTokenizer
from ...utils import load_tokenizer
from ..abstract import BaseLightningModel
from ..embeddings.base import Node2VecEmbedding, SimpleEmbedding
from .encoder import DualSTB
from .moco import MoCo


class TrajCL(nn.Module):
    """TrajCL model for trajectory contrastive learning.

    Combines dual-stream encoders (DualSTB) with MoCo contrastive framework
    for learning trajectory representations.

    Parameters
    ----------
    embedding_dim : int
        Dimension of cell embeddings (also encoder input/output dimension).
    hidden_dim : int
        Dimension of feedforward network in transformer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    attn_dropout : float
        Dropout rate for attention.
    pos_dropout : float
        Dropout rate for positional encoding.
    proj_dim : int
        Dimension of projection head output.
    queue_size : int
        Size of MoCo negative sample queue.
    temperature : float
        Temperature for contrastive loss.
    spatial_hidden : int, optional
        Hidden dimension for spatial stream, by default 32.
    spatial_layers : int, optional
        Number of layers for spatial stream, by default 1.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        attn_dropout: float,
        pos_dropout: float,
        proj_dim: int,
        queue_size: int,
        temperature: float,
        spatial_hidden: int = 32,
        spatial_layers: int = 1,
    ):
        super(TrajCL, self).__init__()

        # Create query encoder
        self.encoder_q = DualSTB(
            ninput=embedding_dim,
            nhidden=hidden_dim,
            nhead=num_heads,
            nlayer=num_layers,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            spatial_nhidden=spatial_hidden,
            spatial_nlayer=spatial_layers,
        )

        # Create key encoder (same architecture)
        self.encoder_k = DualSTB(
            ninput=embedding_dim,
            nhidden=hidden_dim,
            nhead=num_heads,
            nlayer=num_layers,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            spatial_nhidden=spatial_hidden,
            spatial_nlayer=spatial_layers,
        )

        # MoCo framework
        self.moco = MoCo(
            encoder_q=self.encoder_q,
            encoder_k=self.encoder_k,
            nemb=embedding_dim,
            nout=proj_dim,
            queue_size=queue_size,
            temperature=temperature,
        )

    def forward(
        self,
        cell_emb: torch.Tensor,
        spatial_feat: torch.Tensor,
        lengths: List[int],
    ) -> torch.Tensor:
        """Encode trajectory using query encoder (for inference).

        Parameters
        ----------
        cell_emb : torch.Tensor
            Cell embeddings, shape (seq_len, batch_size, emb_dim).
        spatial_feat : torch.Tensor
            Spatial features, shape (seq_len, batch_size, 4).
        lengths : List[int]
            Actual sequence lengths.

        Returns
        -------
        torch.Tensor
            Trajectory embeddings, shape (batch_size, emb_dim).
        """
        # Create padding mask
        max_len = cell_emb.shape[0]
        lengths_tensor = torch.tensor(lengths, device=cell_emb.device)
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
            srcspatial=spatial_feat,
        )

        return traj_emb

    def compute_contrastive(
        self,
        cell_emb1: torch.Tensor,
        spatial_feat1: torch.Tensor,
        lengths1: List[int],
        cell_emb2: torch.Tensor,
        spatial_feat2: torch.Tensor,
        lengths2: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute contrastive learning outputs.

        Parameters
        ----------
        cell_emb1, cell_emb2 : torch.Tensor
            Cell embeddings for view 1 and view 2.
        spatial_feat1, spatial_feat2 : torch.Tensor
            Spatial features for view 1 and view 2.
        lengths1, lengths2 : List[int]
            Sequence lengths for view 1 and view 2.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (logits, targets) for contrastive loss.
        """
        # Create padding masks
        max_len1 = cell_emb1.shape[0]
        lengths1_tensor = torch.tensor(lengths1, device=cell_emb1.device)
        src_padding_mask1 = (
            torch.arange(max_len1, device=cell_emb1.device)[None, :]
            >= lengths1_tensor[:, None]
        )

        max_len2 = cell_emb2.shape[0]
        lengths2_tensor = torch.tensor(lengths2, device=cell_emb2.device)
        src_padding_mask2 = (
            torch.arange(max_len2, device=cell_emb2.device)[None, :]
            >= lengths2_tensor[:, None]
        )

        # Prepare kwargs for encoders
        kwargs_q = {
            "src": cell_emb1,
            "attn_mask": None,
            "src_padding_mask": src_padding_mask1,
            "src_len": lengths1_tensor,
            "srcspatial": spatial_feat1,
        }

        kwargs_k = {
            "src": cell_emb2,
            "attn_mask": None,
            "src_padding_mask": src_padding_mask2,
            "src_len": lengths2_tensor,
            "srcspatial": spatial_feat2,
        }

        # Compute contrastive outputs
        logits, targets = self.moco(kwargs_q, kwargs_k)
        return logits, targets

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Parameters
        ----------
        logits : torch.Tensor
            Logits from compute_contrastive.
        targets : torch.Tensor
            Target labels.

        Returns
        -------
        torch.Tensor
            Contrastive loss.
        """
        return self.moco.loss(logits, targets)


class TrajCLLightning(BaseLightningModel):
    """Lightning module for training TrajCL.

    Parameters
    ----------
    tokenizer : Union[str, AbstractTokenizer]
        Tokenizer path or instance for vocabulary and grid system.
    embedding_dim : int
        Dimension of cell embeddings.
    hidden_dim : int
        Dimension of feedforward network.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    attn_dropout : float
        Dropout rate for attention.
    pos_dropout : float
        Dropout rate for positional encoding.
    proj_dim : int
        Projection dimension for contrastive learning.
    queue_size : int
        Size of MoCo queue.
    temperature : float
        Temperature for contrastive loss.
    cell_embeddings : torch.Tensor, optional
        Pre-trained cell embeddings from node2vec, by default None.
    cell_embeddings_path : str, optional
        Path to load cell embeddings from, by default None.
        If provided, takes precedence over cell_embeddings tensor.
    freeze_embedding : bool, optional
        Whether to freeze embeddings, by default False.
    spatial_hidden : int, optional
        Hidden dimension for spatial stream, by default 32.
    spatial_layers : int, optional
        Number of layers for spatial stream, by default 1.
    optimizer_type : str, optional
        Optimizer type, by default "adam".
    learning_rate : float, optional
        Learning rate, by default 1e-3.
    weight_decay : float, optional
        Weight decay, by default 0.0001.
    """

    path_params: ClassVar[tuple[str, ...]] = ("tokenizer", "cell_embeddings_path")

    def __init__(
        self,
        tokenizer: Union[str, AbstractTokenizer],
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        attn_dropout: float,
        pos_dropout: float,
        proj_dim: int,
        queue_size: int,
        temperature: float,
        cell_embeddings_path: str | None = None,
        freeze_embedding: bool = True,
        spatial_hidden: int = 32,
        spatial_layers: int = 1,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0001,
    ):
        super().__init__(optimizer_type=optimizer_type, learning_rate=learning_rate)
        self.save_hyperparameters(ignore=["tokenizer", "cell_embeddings"])

        tokenizer = load_tokenizer(tokenizer=tokenizer)

        self._weight_decay = weight_decay

        if cell_embeddings_path is not None:
            # Load cell embeddings from path if provided
            cell_embeddings = torch.load(cell_embeddings_path, weights_only=True)

            # Create embedding layer
            self.embedding = Node2VecEmbedding(
                tokenizer=tokenizer,
                cell_embeddings=cell_embeddings,
                freeze=freeze_embedding,
            )
        else:
            self.embedding = SimpleEmbedding(
                tokenizer=tokenizer,
                embedding_dim=embedding_dim,
            )
        if freeze_embedding:
            self.embedding.freeze_parameters()

        # Create model
        self.model = TrajCL(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            proj_dim=proj_dim,
            queue_size=queue_size,
            temperature=temperature,
            spatial_hidden=spatial_hidden,
            spatial_layers=spatial_layers,
        )

    def forward(self, batch: TrajCLSample) -> torch.Tensor:
        """Encode trajectories for inference.

        Parameters
        ----------
        batch : TrajCLSample
            Input batch containing view1 data.

        Returns
        -------
        torch.Tensor
            Trajectory embeddings.
        """
        # Get cell embeddings
        cell_emb = self._get_cell_embeddings(batch.view1_cell_seq, batch.view1_lengths)

        # Encode
        return self.model(
            cell_emb=cell_emb,
            spatial_feat=batch.view1_spatial_feat.transpose(0, 1),
            lengths=batch.view1_lengths,
        )

    def _get_cell_embeddings(
        self, cell_seq: torch.LongTensor, lengths: List[int]
    ) -> torch.Tensor:
        """Convert cell IDs to embeddings.

        Parameters
        ----------
        cell_seq : torch.LongTensor
            Cell IDs, shape (batch_size, seq_len).
        lengths : List[int]
            Sequence lengths.

        Returns
        -------
        torch.Tensor
            Cell embeddings, shape (seq_len, batch_size, emb_dim).
        """
        # (batch_size, seq_len, emb_dim)
        emb = self.embedding(cell_seq)
        # Transpose to (seq_len, batch_size, emb_dim) for transformer
        return emb.transpose(0, 1)

    def compute_loss(self, batch: TrajCLSample) -> Tuple[torch.Tensor, int]:
        """Compute contrastive loss.

        Parameters
        ----------
        batch : TrajCLSample
            Input batch containing two views.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Loss value and batch size.
        """
        # Get cell embeddings for both views
        cell_emb1 = self._get_cell_embeddings(batch.view1_cell_seq, batch.view1_lengths)
        cell_emb2 = self._get_cell_embeddings(batch.view2_cell_seq, batch.view2_lengths)

        # Spatial features: (batch_size, seq_len, 4) -> (seq_len, batch_size, 4)
        spatial1 = batch.view1_spatial_feat.transpose(0, 1)
        spatial2 = batch.view2_spatial_feat.transpose(0, 1)

        # Compute contrastive outputs
        logits, targets = self.model.compute_contrastive(
            cell_emb1=cell_emb1,
            spatial_feat1=spatial1,
            lengths1=batch.view1_lengths,
            cell_emb2=cell_emb2,
            spatial_feat2=spatial2,
            lengths2=batch.view2_lengths,
        )

        # Compute loss
        loss = self.model.loss(logits, targets)

        return loss, batch.batch_size

    def training_step(self, batch: TrajCLSample, batch_idx: int) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : TrajCLSample
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        loss, batch_size = self.compute_loss(batch)
        # Log loss for monitoring and early stopping.
        # Since drop_last=True and batch_size is fixed at 128, weighted average
        # is equivalent to simple average (matching original TrajCL's
        # `tool_funcs.mean(loss_ep)`). We pass batch_size to suppress
        # Lightning's ambiguous batch_size inference warning.
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: TrajCLSample, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Parameters
        ----------
        batch : TrajCLSample
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        loss, batch_size = self.compute_loss(batch)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and lr scheduler.

        Matches original TrajCL: Adam + StepLR(step_size=5, gamma=0.5).

        Returns
        -------
        dict
            Optimizer and lr_scheduler config.
        """
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self._weight_decay,
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self._weight_decay,
            )
        elif self.optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self._weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
