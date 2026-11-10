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

"""Momentum Contrast (MoCo) framework for TrajCL.

Based on: https://arxiv.org/abs/1911.05722
Original implementation: https://github.com/facebookresearch/moco
"""

from typing import Dict

import torch
import torch.nn as nn


class Projector(nn.Module):
    """Projection head for contrastive learning.

    A two-layer MLP that projects encoder outputs to a lower-dimensional
    space where contrastive loss is computed.

    Parameters
    ----------
    nin : int
        Input dimension (encoder output dimension).
    nout : int
        Output dimension (projection dimension).
    """

    def __init__(self, nin: int, nout: int):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nin, nin),
            nn.ReLU(),
            nn.Linear(nin, nout),
        )
        # Match original TrajCL initialization:
        # xavier_normal_(weight, gain=1.414) + zeros_(bias).
        # This is critical for InfoNCE early-stage gradient signal; the
        # PyTorch default Kaiming Uniform produces a narrower projection
        # distribution that under-trains within a small number of epochs.
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.414)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input through MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nin).

        Returns
        -------
        torch.Tensor
            Projected tensor of shape (batch_size, nout).
        """
        return self.mlp(x)


class MoCo(nn.Module):
    """Momentum Contrast for contrastive learning.

    Maintains a queue of negative samples and uses a momentum-updated
    key encoder for stable training.

    Parameters
    ----------
    encoder_q : nn.Module
        Query encoder (trained with gradients).
    encoder_k : nn.Module
        Key encoder (momentum-updated, no gradients).
    nemb : int
        Dimension of encoder output.
    nout : int
        Dimension of projector output.
    queue_size : int
        Size of the negative sample queue.
    mmt : float, optional
        Momentum coefficient for key encoder update, by default 0.999.
    temperature : float, optional
        Temperature for softmax, by default 0.07.
    """

    def __init__(
        self,
        encoder_q: nn.Module,
        encoder_k: nn.Module,
        nemb: int,
        nout: int,
        queue_size: int,
        mmt: float = 0.999,
        temperature: float = 0.07,
    ):
        super(MoCo, self).__init__()

        self.queue_size = queue_size
        self.mmt = mmt
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss()

        # Create encoders
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # Create projection heads
        self.mlp_q = Projector(nemb, nout)
        self.mlp_k = Projector(nemb, nout)

        # Initialize encoder_k and mlp_k to match q
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(nout, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder with momentum."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.mmt + param_q.data * (1.0 - self.mmt)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1.0 - self.mmt)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new keys.

        Parameters
        ----------
        keys : torch.Tensor
            New keys to enqueue, shape (batch_size, nout).
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Handle queue wrap-around
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            # Split into two parts when wrapping around
            first_part = self.queue_size - ptr
            self.queue[:, ptr : self.queue_size] = keys.T[:, :first_part]
            self.queue[:, : batch_size - first_part] = keys.T[:, first_part:]

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(
        self, kwargs_q: Dict[str, torch.Tensor], kwargs_k: Dict[str, torch.Tensor]
    ) -> tuple:
        """Compute contrastive loss.

        Parameters
        ----------
        kwargs_q : dict
            Arguments for query encoder, containing:
                - src: structural input
                - attn_mask: attention mask
                - src_padding_mask: padding mask
                - src_len: sequence lengths
                - srcspatial: spatial features
        kwargs_k : dict
            Arguments for key encoder (same structure as kwargs_q).

        Returns
        -------
        tuple
            (logits, targets) for cross-entropy loss computation.
        """
        # Compute query features
        q = self.mlp_q(self.encoder_q(**kwargs_q))
        q = nn.functional.normalize(q, dim=1)

        # Compute key features (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.mlp_k(self.encoder_k(**kwargs_k))
            k = nn.functional.normalize(k, dim=1)

        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Concatenate: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.temperature

        # Labels: all positive keys are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Parameters
        ----------
        logits : torch.Tensor
            Logits from forward pass.
        targets : torch.Tensor
            Target labels (all zeros for contrastive learning).

        Returns
        -------
        torch.Tensor
            Cross-entropy loss.
        """
        return self.criterion(logits, targets)
