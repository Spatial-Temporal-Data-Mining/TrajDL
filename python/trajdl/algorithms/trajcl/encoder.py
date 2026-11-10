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

"""Encoder modules for TrajCL.

This module implements the dual-stream encoder architecture:
- PositionalEncoding: Standard sinusoidal positional encoding for transformers
- SpatialMSM: Spatial Micro-Spatial-Movement Model for processing local spatial features
- DualSTB: Dual-stream Spatio-Temporal Backbone that fuses structural and spatial streams
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..embeddings.position import SinusoidalPE


class PositionalEncoding(SinusoidalPE):
    """WS5 compat shim delegating to ``embeddings.position.SinusoidalPE``.

    Keeps the legacy ``PositionalEncoding(emb_size, dropout, maxlen=201)``
    call surface (positional and keyword) and the ``pos_embedding``
    state_dict key with its ``(maxlen, 1, emb)`` table shape; this is the
    trajcl-PE row of the WS5 preset table and carries no math itself.  The
    preset's ``forward_mode="add_dropout"`` also owns the ``dropout``
    submodule unconditionally, matching the legacy module tree.
    """

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 201):
        super().__init__(
            dim=emb_size,
            max_len=maxlen,
            layout="interleaved",
            store="table",
            buffer_name="pos_embedding",
            table_shape="col_major_T1C",
            forward_mode="add_dropout",
            dropout=dropout,
            freq_variant="exp_mul_div",
        )


class SpatialMSMLayer(nn.Module):
    """Single layer of the Spatial MSM encoder.

    A custom TransformerEncoderLayer that returns attention weights
    for use as spatial attention bias.

    Parameters
    ----------
    d_model : int
        Dimension of the model.
    nhead : int
        Number of attention heads.
    dim_feedforward : int, optional
        Dimension of the feedforward network, by default 2048.
    dropout : float, optional
        Dropout rate, by default 0.1.
    activation : str, optional
        Activation function, by default "relu".
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(SpatialMSMLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    @staticmethod
    def _get_activation_fn(activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        return F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass returning output and attention weights.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (seq_len, batch_size, d_model).
        src_mask : Tensor, optional
            Attention mask, by default None.
        src_key_padding_mask : Tensor, optional
            Key padding mask of shape (batch_size, seq_len), by default None.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Output tensor and attention weights (batch_size, seq_len, seq_len).
        """
        # src: (seq_len, batch_size, d_model)
        # src2: (seq_len, batch_size, d_model)
        # attn: (batch_size, seq_len, seq_len) — averaged across heads
        #   (average_attn_weights=True averages over nhead dimension)
        src2, attn = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # src: (seq_len, batch_size, d_model), attn: (batch_size, seq_len, seq_len)
        return src, attn


class SpatialMSMEncoder(nn.Module):
    """Multi-layer Spatial MSM encoder.

    Stacks multiple SpatialMSMLayer and returns the output and
    attention weights from the last layer.

    Parameters
    ----------
    encoder_layer : SpatialMSMLayer
        The encoder layer to stack.
    num_layers : int
        Number of layers to stack.
    norm : nn.Module, optional
        Normalization layer, by default None.
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer: SpatialMSMLayer, num_layers: int, norm=None):
        super(SpatialMSMEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through all layers.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (seq_len, batch_size, d_model).
        mask : Tensor, optional
            Attention mask, by default None.
        src_key_padding_mask : Tensor, optional
            Key padding mask, by default None.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Output tensor and attention weights from the last layer.
        """
        output = src
        attn = None

        for mod in self.layers:
            output, attn = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, attn


class SpatialMSM(nn.Module):
    """Spatial Micro-Spatial-Movement Model.

    A custom transformer designed to process local spatial features
    and extract attention weights for use as spatial bias in the
    structural stream.

    Parameters
    ----------
    ninput : int
        Input dimension (4 for spatial features).
    nhidden : int
        Hidden dimension for the feedforward network.
    nhead : int
        Number of attention heads.
    nlayer : int
        Number of transformer layers (default: 3, matching original TrajCL).
    attn_dropout : float
        Dropout rate for attention.
    pos_dropout : float
        Dropout rate for positional encoding.
    """

    def __init__(
        self,
        ninput: int,
        nhidden: int,
        nhead: int,
        nlayer: int = 3,  # Default 3 layers, matching original TrajCL
        attn_dropout: float = 0.1,
        pos_dropout: float = 0.1,
    ):
        super(SpatialMSM, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden

        # Original TrajCL applies PositionalEncoding directly to 4-dim spatial features
        # without an input projection layer (see dual_attention.py:121-124)
        self.pos_encoder = PositionalEncoding(ninput, pos_dropout)

        trans_encoder_layers = SpatialMSMLayer(ninput, nhead, nhidden, attn_dropout)
        self.trans_encoder = SpatialMSMEncoder(trans_encoder_layers, nlayer)

    def forward(
        self,
        src: Tensor,
        attn_mask: Optional[Tensor],
        src_padding_mask: Optional[Tensor],
        src_len: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Process spatial features and return embedding and attention.

        Parameters
        ----------
        src : Tensor
            Spatial features of shape (seq_len, batch_size, 4).
        attn_mask : Tensor, optional
            Attention mask.
        src_padding_mask : Tensor, optional
            Padding mask of shape (batch_size, seq_len).
        src_len : Tensor
            Actual lengths of each sequence, shape (batch_size,).

        Returns
        -------
        Tuple[Tensor, Tensor]
            - Output embedding of shape (batch_size, nhidden)
            - Attention weights of shape (batch_size, seq_len, seq_len)
        """
        # src: (seq_len, batch_size, 4)
        src = self.pos_encoder(src)
        # src: (seq_len, batch_size, 4) — positional encoding added

        # rtn: (seq_len, batch_size, 4), attn: (batch_size, seq_len, seq_len)
        # attn is the averaged attention weights from the last SpatialMSMLayer
        # (averaged across heads via average_attn_weights=True in MultiheadAttention)
        rtn, attn = self.trans_encoder(src, attn_mask, src_padding_mask)

        # Average pooling over valid (non-padding) positions
        # src_padding_mask: (batch_size, seq_len), True for padding positions
        # src_padding_mask.T: (seq_len, batch_size)
        # mask: (seq_len, batch_size, 4) — 1 for valid, 0 for padding
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        # rtn: (batch_size, 4) — sum over seq_len dim, then divide by actual lengths
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        return rtn, attn


class DualSTB(nn.Module):
    """Dual-stream Spatio-Temporal Backbone.

    A dual-stream architecture that processes structural information
    (grid cell embeddings) and spatial information (local geometric
    features) in parallel, fusing them through attention bias.

    Parameters
    ----------
    ninput : int
        Dimension of input embeddings (cell embedding dimension).
    nhidden : int
        Dimension of the feedforward network in transformer.
    nhead : int
        Number of attention heads.
    nlayer : int
        Number of transformer layers.
    attn_dropout : float
        Dropout rate for attention.
    pos_dropout : float
        Dropout rate for positional encoding.
    spatial_nhidden : int, optional
        Hidden dimension for spatial stream, by default 32.
    spatial_nlayer : int, optional
        Number of layers for spatial stream, by default 3 (matching original TrajCL).
    """

    def __init__(
        self,
        ninput: int,
        nhidden: int,
        nhead: int,
        nlayer: int,
        attn_dropout: float,
        pos_dropout: float,
        spatial_nhidden: int = 32,
        spatial_nlayer: int = 3,  # Default 3 layers, matching original TrajCL
    ):
        super(DualSTB, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(ninput, pos_dropout)

        # Structural stream: standard transformer encoder
        structural_attn_layers = nn.TransformerEncoderLayer(
            ninput, nhead, nhidden, attn_dropout, batch_first=False
        )
        self.structural_attn = nn.TransformerEncoder(structural_attn_layers, nlayer)

        # Spatial stream: custom MSM for extracting spatial attention
        self.spatial_attn = SpatialMSM(
            ninput=4,  # 4 spatial features
            nhidden=spatial_nhidden,
            nhead=1,
            nlayer=spatial_nlayer,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
        )

        # Learnable fusion weight
        self.gamma_param = nn.Parameter(data=torch.tensor(0.5), requires_grad=True)

    def forward(
        self,
        src: Tensor,
        attn_mask: Optional[Tensor],
        src_padding_mask: Tensor,
        src_len: Tensor,
        srcspatial: Tensor,
    ) -> Tensor:
        """Forward pass through dual-stream encoder.

        Parameters
        ----------
        src : Tensor
            Structural input (cell embeddings), shape (seq_len, batch_size, emb_dim).
        attn_mask : Tensor, optional
            Attention mask (typically None for TrajCL).
        src_padding_mask : Tensor
            Padding mask of shape (batch_size, seq_len).
        src_len : Tensor
            Actual lengths, shape (batch_size,).
        srcspatial : Tensor
            Spatial features, shape (seq_len, batch_size, 4).

        Returns
        -------
        Tensor
            Trajectory embeddings of shape (batch_size, emb_dim).
        """
        # --- Spatial stream ---
        # srcspatial: (seq_len, batch_size, 4)
        if srcspatial is not None:
            # attn_spatial: (batch_size, seq_len, seq_len)
            # This is the single-head averaged attention weight matrix from SpatialMSM.
            _, attn_spatial = self.spatial_attn(
                srcspatial, attn_mask, src_padding_mask, src_len
            )

            # Replicate spatial attention for multi-head structural attention.
            # PyTorch's TransformerEncoder accepts a 3D attn_mask of shape
            # (nhead * batch_size, seq_len, seq_len), which is added to the
            # attention scores before softmax.
            # The layout must be [b0h0, b0h1, ..., b1h0, b1h1, ...], i.e.
            # each batch item's nhead copies are contiguous.
            #
            # NOTE: The original TrajCL used repeat(nhead, 1, 1) which produces
            # [b0..bN, b0..bN, ...] — an incorrect layout that causes each
            # sample's embedding to depend on other samples in the same batch.
            # Full analysis: the TrajCL section of docs/supplementary/benchmarking.md.
            #
            # attn_spatial: (batch_size, seq_len, seq_len)
            #   -> (nhead * batch_size, seq_len, seq_len)
            attn_spatial = attn_spatial.repeat_interleave(self.nhead, dim=0)

            # gamma: scalar in (0, 10), controls spatial attention strength
            gamma = torch.sigmoid(self.gamma_param) * 10
            # attn_spatial: (nhead * batch_size, seq_len, seq_len)
            attn_spatial = gamma * attn_spatial
        else:
            attn_spatial = None

        # --- Structural stream with spatial bias ---
        # src: (seq_len, batch_size, emb_dim)
        src = self.pos_encoder(src)
        # src: (seq_len, batch_size, emb_dim) — positional encoding added

        # structural_attn is nn.TransformerEncoder (batch_first=False).
        # It takes:
        #   src:  (seq_len, batch_size, emb_dim)
        #   mask: (nhead * batch_size, seq_len, seq_len) — added to attn scores
        #   src_key_padding_mask: (batch_size, seq_len)
        # rtn: (seq_len, batch_size, emb_dim)
        rtn = self.structural_attn(src, attn_spatial, src_padding_mask)

        # Average pooling over valid (non-padding) positions
        # src_padding_mask: (batch_size, seq_len), True for padding positions
        # src_padding_mask.T: (seq_len, batch_size)
        # mask: (seq_len, batch_size, emb_dim) — 1 for valid, 0 for padding
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        # rtn: (batch_size, emb_dim) — sum over seq_len dim, then divide by lengths
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        return rtn
