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

import re
from datetime import datetime

import pytest
import torch

from trajdl.algorithms.embeddings.ctle import (
    CTLETokenEmbedding,
    CTLETokenEmbeddingWithTransformer,
    PositionalEncoding,
    TemporalEncoding,
)
from trajdl.tokenizers.locseq import LocSeqTokenizer

START_TS = int(datetime.strptime("20000101000000", "%Y%m%d%H%M%S").timestamp())
END_TS = int(datetime.now().timestamp())


def test_pe():
    batch_size, seq_length, embedding_dim, max_len = 3, 12, 32, 100
    pe = PositionalEncoding(embedding_dim=embedding_dim, max_len=max_len)
    x = torch.randint(low=0, high=32, size=(batch_size, seq_length))
    assert pe(x).shape == (1, seq_length, embedding_dim)


def test_temporal_encoding():
    batch_size, seq_length, embedding_dim, max_len = 3, 12, 32, 100
    tp = TemporalEncoding(embedding_dim=embedding_dim)
    x = torch.randint(low=START_TS, high=END_TS, size=(batch_size, seq_length))
    assert tp(x).shape == (batch_size, seq_length, embedding_dim)


def test_ctle_emb(locseq_tokenizer: LocSeqTokenizer):
    batch_size, seq_length, embedding_dim, max_len = 3, 12, 32, 100

    x = torch.randint(low=0, high=len(locseq_tokenizer), size=(batch_size, seq_length))
    ts_src = torch.randint(low=START_TS, high=END_TS, size=(batch_size, seq_length))

    for type_ in ["pe", "tp"]:
        emb = CTLETokenEmbedding(
            embedding_type=type_,
            tokenizer=locseq_tokenizer,
            embedding_dim=embedding_dim,
            max_len=max_len,
        )
        assert emb(x, ts_src).shape == (batch_size, seq_length, embedding_dim)

    with pytest.raises(
        ValueError, match=re.escape("`embedding_type` only support {'pe', 'tp'}")
    ):
        CTLETokenEmbedding(
            embedding_type="test",
            tokenizer=locseq_tokenizer,
            embedding_dim=embedding_dim,
            max_len=max_len,
        )


def test_ctle_with_transformer(locseq_tokenizer: LocSeqTokenizer):
    batch_size, seq_length, embedding_dim, max_len, hidden_size = 3, 6, 32, 100, 64
    pad_value = locseq_tokenizer.pad

    loc_src = torch.LongTensor(
        [
            [0, 1, 2, 3, pad_value, pad_value],
            [1, 2, 3, pad_value, pad_value, pad_value],
            [5, 6, 7, 8, 9, 10],
        ]
    )
    ts_src = torch.LongTensor(
        [
            [1725516770, 1725516771, 1725516772, 1725516773, 0, 0],
            [1725516783, 1725516784, 1725516785, 0, 0, 0],
            [1725516783, 1725516784, 1725516785, 1725516786, 1725516787, 1725516788],
        ]
    )

    mask_prob = 0.2
    mask = (torch.rand(size=(batch_size, seq_length)) < mask_prob) & (
        loc_src != pad_value
    )

    for type_ in ["pe", "tp"]:
        emb = CTLETokenEmbeddingWithTransformer(
            embedding_type=type_,
            embedding_dim=embedding_dim,
            max_len=max_len,
            num_layers=2,
            n_heads=8,
            tokenizer=locseq_tokenizer,
            hidden_size=16,
            dropout=0.1,
        )

        assert emb(loc_src, ts_src, mask).shape == (
            batch_size,
            seq_length,
            embedding_dim,
        )
