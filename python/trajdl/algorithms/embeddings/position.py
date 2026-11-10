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

"""Shared sinusoidal position / learnable Fourier time encodings (WS5 uplift).

Absorbs the six private encoding classes of ctle, trajcl, transfertraj and
trajmamba into two configurable modules.  The constructor parameters are a
**bitwise contract**: each consumer maps to one preset row of the WS5 design
table (the WS5 uplift design, finalized 2026-10-24) and its state_dict keys/tensors plus forward outputs must
stay bit-for-bit identical to the pre-uplift classes (``torch.equal``, not
tolerance) because live checkpoints and the npz/``.pt`` gold fixtures were
recorded against those exact bit patterns.

Why the frequency formulas are NOT unified (bit-pattern contract): the three
interleaved/halves families compute mathematically equivalent frequencies
with different float32 operation orders -- ``exp(arange * -(log(10000)/dim))``
(ctle) vs ``exp(arange * (-log(10000)) / dim)`` (trajcl, multiply-then-divide)
vs ``1 / (10000 ** (arange / dim))`` (transfertraj) -- and ``sin``/``cos`` of
the differently-rounded arguments diverge in the low bits.  Each
``freq_variant`` branch below therefore reproduces its legacy source formula
verbatim, in the original operation order; "tidying" them into one formula
would silently break every aligned checkpoint and is forbidden by the plan's
numerical red line.
"""

import math

import numpy as np
import torch
from torch import nn

_LAYOUTS = ("interleaved", "halves")
_STORES = ("table", "inv_freq")
_TABLE_SHAPES = ("row_major", "col_major_T1C", "raw")
_FORWARD_MODES = ("return", "add_dropout")
_FREQ_VARIANTS = ("exp_neg_ratio", "exp_mul_div", "inv_pow")


class SinusoidalPE(nn.Module):
    """Configurable sinusoidal position encoding absorbing five legacy shapes.

    Parameters (the WS5 preset surface; defaults reproduce the ctle PE):

    dim : int
        Encoding width (legacy ``embedding_dim`` / ``emb_size`` / ``hidden_size``).
    max_len : int, optional
        Table length; required when ``store="table"``, unused for ``inv_freq``.
    layout : {"interleaved", "halves"}
        ``interleaved`` fills ``[:, 0::2] = sin`` / ``[:, 1::2] = cos`` of a
        precomputed table; ``halves`` concatenates ``[sin | cos]`` at forward
        time from an inverse-frequency buffer.
    store : {"table", "inv_freq"}
        ``table`` precomputes the full ``(max_len, dim)`` table and registers
        it under ``buffer_name``; ``inv_freq`` registers only the ``(dim/2,)``
        frequency buffer and computes angles in ``forward``.
    buffer_name : str
        state_dict key of the registered buffer (contract surface: ``pe`` for
        ctle, ``pos_embedding`` for trajcl, ``inv_freq`` for transfertraj).
    table_shape : {"row_major", "col_major_T1C", "raw"}
        Per-consumer materialisation of the precomputed ``(max_len, dim)``
        table before registration: ``row_major`` applies ``unsqueeze(0)``
        giving ``(1, max_len, dim)`` (ctle), ``col_major_T1C`` applies
        ``unsqueeze(-2)`` giving ``(max_len, 1, dim)`` (trajcl), ``raw``
        stores the unshaped ``(max_len, dim)`` table.  Only ``row_major``
        has a defined ``return``-mode forward; ``raw`` and ``col_major_T1C``
        are storage materialisations (legacy pre-``unsqueeze`` / time-first
        states) and ``forward`` raises for them outside ``add_dropout``.
    forward_mode : {"return", "add_dropout"}
        ``return`` returns the sliced buffer (transfertraj recomputes from
        ``inv_freq``); ``add_dropout`` returns
        ``dropout(x + table[: x.size(0), :])`` for ``(T, B, E)`` inputs and
        creates the ``nn.Dropout`` submodule unconditionally -- mirroring the
        legacy trajcl module tree so parameter/module structure parity holds.
    dropout : float
        Dropout probability for the ``add_dropout`` mode.
    freq_variant : {"exp_neg_ratio", "exp_mul_div", "inv_pow"}
        Verbatim frequency formula of the absorbed legacy class; see the
        module docstring for why these must not be merged.

    Valid combos: ``interleaved`` requires a table-store formula
    (``exp_neg_ratio`` | ``exp_mul_div``); ``halves`` requires
    ``freq_variant="inv_pow"`` with ``store="inv_freq"``; ``add_dropout``
    requires ``store="table"`` with ``table_shape="col_major_T1C"`` (the
    only legacy time-first contract — no preset consumes a ``raw`` table in
    forward); ``return`` mode is likewise defined for ``row_major`` tables
    only and ``forward`` raises for the other shapes.
    """

    def __init__(
        self,
        dim,
        max_len=None,
        *,
        layout="interleaved",
        store="table",
        buffer_name="pe",
        table_shape="row_major",
        forward_mode="return",
        dropout=0.0,
        freq_variant="exp_neg_ratio",
    ):
        super().__init__()
        for name, value, allowed in (
            ("layout", layout, _LAYOUTS),
            ("store", store, _STORES),
            ("table_shape", table_shape, _TABLE_SHAPES),
            ("forward_mode", forward_mode, _FORWARD_MODES),
            ("freq_variant", freq_variant, _FREQ_VARIANTS),
        ):
            if value not in allowed:
                raise ValueError(f"`{name}` must be one of {allowed}, got {value!r}")
        if freq_variant in ("exp_neg_ratio", "exp_mul_div") and layout != "interleaved":
            raise ValueError(
                f"`freq_variant={freq_variant!r}` requires layout='interleaved'"
            )
        if freq_variant == "inv_pow" and (layout, store) != ("halves", "inv_freq"):
            raise ValueError(
                "`freq_variant='inv_pow'` requires layout='halves' and "
                "store='inv_freq'"
            )
        if store == "inv_freq" and layout != "halves":
            raise ValueError("store='inv_freq' requires layout='halves'")
        if store == "table" and max_len is None:
            raise ValueError("`max_len` is required when store='table'")
        if forward_mode == "add_dropout" and (
            store != "table" or table_shape != "col_major_T1C"
        ):
            raise ValueError(
                "forward_mode='add_dropout' requires store='table' and "
                "table_shape='col_major_T1C'"
            )

        self._buffer_name = buffer_name
        self._store = store
        self._table_shape = table_shape
        self._forward_mode = forward_mode

        if store == "table":
            if freq_variant == "exp_neg_ratio":
                # verbatim ctle PositionalEncoding (embeddings/ctle.py)
                pe = torch.zeros(max_len, dim)
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                table = pe
            else:  # freq_variant == "exp_mul_div"
                # verbatim trajcl PositionalEncoding (trajcl/encoder.py);
                # multiply-then-divide order differs from exp_neg_ratio
                den = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0)) / dim)
                pos = torch.arange(0, max_len).reshape(max_len, 1)
                pos_embedding = torch.zeros((max_len, dim))
                pos_embedding[:, 0::2] = torch.sin(pos * den)
                pos_embedding[:, 1::2] = torch.cos(pos * den)
                table = pos_embedding
            if table_shape == "row_major":
                table = table.unsqueeze(0)
            elif table_shape == "col_major_T1C":
                table = table.unsqueeze(-2)
            self.register_buffer(buffer_name, table, persistent=True)
            if forward_mode == "add_dropout":
                # created even for dropout == 0.0: the legacy trajcl class
                # always owns this submodule, keep the module tree identical
                self.dropout = nn.Dropout(dropout)
        else:  # store == "inv_freq"
            # verbatim transfertraj PositionalEncode (transfertraj/layers.py)
            pair_steps = torch.arange(0.0, dim, 2.0) / dim
            self.register_buffer(
                buffer_name, 1.0 / (10000.0**pair_steps), persistent=True
            )

    def forward(self, x):
        buffer = getattr(self, self._buffer_name)
        if self._store == "inv_freq":
            # verbatim transfertraj PositionalEncode.forward: x is (B, T)
            batch, seq_len = x.shape
            angles = torch.outer(x.reshape(-1), buffer)
            pos_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
            return pos_emb.view(batch, seq_len, -1)
        if self._forward_mode == "add_dropout":
            # verbatim trajcl.forward: x is (T, B, E), table is T-first
            return self.dropout(x + buffer[: x.size(0), :])
        if self._table_shape != "row_major":
            # `raw` / `col_major_T1C` are storage materialisations only: no
            # legacy class forwarded an un-unsqueezed (rank-2) or time-first
            # table in return mode, so no bitwise contract exists to pin and
            # the slice axis would be ill-defined (batch vs. time on a
            # (B, T, C) feed).  Fail loudly instead of guessing.
            raise ValueError(
                "forward_mode='return' is only defined for "
                f"table_shape='row_major', got {self._table_shape!r}"
            )
        # verbatim ctle.forward: x is (B, T, C), table is (1, T, C)
        return buffer[:, : x.size(1)]


class FourierEncode(nn.Module):
    """Trainable random-Fourier time encoding (ctle TE / transfertraj / trajmamba).

    Absorbs ctle ``TemporalEncoding`` and the identically-spelled
    transfertraj / trajmamba ``FourierEncode`` classes; the omega initialiser
    ``torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size)).float()`` was
    measured bitwise-equal (``torch.equal``) to the legacy ctle
    ``torch.Tensor(np)`` construction, so no init-variant parameter is needed.

    Parameters (the WS5 preset surface; all defaults reproduce the
    transfertraj/trajmamba spelling):

    embed_size : int
        Number of Fourier channels / output width.
    omega_name : str
        Attribute and state_dict key of the frequency parameter (ctle shim
        keeps ``omega``; rename only if a consumer needs it).
    bias_name : str
        Attribute and state_dict key of the bias parameter.

    ``div_term = math.sqrt(1.0 / embed_size)`` stays a plain float attribute
    (never in state_dict), exactly as in all three legacy classes.
    """

    def __init__(self, embed_size, *, omega_name="omega", bias_name="bias"):
        super().__init__()
        init_omega = 1 / (10 ** np.linspace(0, 9, embed_size))
        self.register_parameter(
            omega_name,
            nn.Parameter(torch.from_numpy(init_omega).float(), requires_grad=True),
        )
        self.register_parameter(
            bias_name,
            nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True),
        )
        self._omega_name = omega_name
        self._bias_name = bias_name
        self.div_term = math.sqrt(1.0 / embed_size)

    def forward(self, x):
        omega = getattr(self, self._omega_name)
        bias = getattr(self, self._bias_name)
        if x.dim() < 3:
            x = x.unsqueeze(-1)
        encode = x * omega.view(1, 1, -1) + bias.view(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode
