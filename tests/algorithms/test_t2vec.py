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
import random
import re

import numpy as np
import pytest
import torch

from trajdl.algorithms.embeddings.base import SimpleEmbedding
from trajdl.algorithms.t2vec import (
    T2VEC,
    DecoderWithAttention,
    GlobalAttention,
    StackingGRU,
    T2VECEncoder,
)
from trajdl.common.samples import T2VECSample
from trajdl.tokenizers import T2VECTokenizer

from ..conftest import FOLDER


def test_stacking_gru():
    with pytest.raises(
        ValueError, match=re.escape("`num_layers` must be greater than 0")
    ):
        StackingGRU(input_size=3, hidden_size=3, num_layers=0, dropout=0)

    for _ in range(10):
        batch_size, input_size, hidden_size, num_layers = (
            random.randint(1, 32),
            random.randint(1, 32),
            random.randint(1, 32),
            random.randint(1, 4),
        )
        model = StackingGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
        )
        input_tensor = torch.rand(size=(batch_size, input_size))
        hidden = torch.rand(size=(num_layers, batch_size, hidden_size))
        output, out_hidden = model(input_tensor, hidden)
        assert output.shape == (batch_size, hidden_size)
        assert out_hidden.shape == (num_layers, batch_size, hidden_size)


def test_global_attention():
    for _ in range(10):
        batch_size, seq_length, hidden_size = (
            random.randint(1, 32),
            random.randint(1, 32),
            random.randint(1, 32),
        )
        model = GlobalAttention(hidden_size=hidden_size)
        query = torch.rand(size=(batch_size, hidden_size))
        context = torch.rand(size=(batch_size, seq_length, hidden_size))
        assert model(query, context).shape == (batch_size, hidden_size)


def test_decoder_with_attention(t2vec_tokenizer: T2VECTokenizer):
    for _ in range(10):
        batch_size, embedding_dim, hidden_size, num_layers = (
            random.randint(1, 32),
            random.randint(1, 32),
            random.randint(1, 32),
            random.randint(1, 4),
        )
        seq_length = random.randint(1, 32)

        embedding_layer = SimpleEmbedding(
            tokenizer=t2vec_tokenizer, embedding_dim=embedding_dim
        )
        model = DecoderWithAttention(
            hidden_size=hidden_size,
            num_layers=num_layers,
            embedding_layer=embedding_layer,
            dropout=0.0,
        )
        src = torch.randint(
            low=0, high=len(t2vec_tokenizer), size=(batch_size, seq_length)
        )
        hidden = torch.rand(size=(num_layers, batch_size, hidden_size))
        all_encoder_hidden_states = torch.rand(
            size=(batch_size, seq_length, hidden_size)
        )
        assert model(src, hidden, all_encoder_hidden_states).shape == (
            batch_size,
            seq_length,
            hidden_size,
        )


def test_t2vec_encoder(t2vec_tokenizer: T2VECTokenizer):
    for _ in range(10):
        for bidirectional in [True, False]:
            batch_size, embedding_dim, hidden_size, num_layers = (
                random.randint(1, 32),
                random.randint(1, 32),
                random.randint(1, 32),
                random.randint(1, 4),
            )
            seq_length = random.randint(1, 32)
            lengths = [random.randint(1, seq_length) for _ in range(batch_size)]
            lengths[0] = seq_length
            if bidirectional and hidden_size % 2 != 0:
                hidden_size += 1

            embedding_layer = SimpleEmbedding(
                tokenizer=t2vec_tokenizer, embedding_dim=embedding_dim
            )
            model = T2VECEncoder(
                embedding_layer=embedding_layer,
                padding_value=t2vec_tokenizer.pad,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=0.0,
            )

            src = torch.randint(
                low=0, high=len(t2vec_tokenizer), size=(batch_size, seq_length)
            )
            output = model(src, lengths)
            assert output[0].shape == (num_layers, batch_size, hidden_size)
            assert output[1].shape == (batch_size, seq_length, hidden_size)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`hidden_size` should be an even number greater than 0 when `bidirectional` is True"
        ),
    ):
        T2VECEncoder(
            embedding_layer=embedding_layer,
            padding_value=t2vec_tokenizer.pad,
            hidden_size=3,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.0,
        )


def test_t2vec(t2vec_tokenizer: T2VECTokenizer):
    for _ in range(10):
        for bidirectional_encoder in [True, False]:
            for freeze in [True, False]:
                batch_size, embedding_dim, hidden_size, num_layers = (
                    random.randint(1, 32),
                    random.randint(1, 32),
                    random.randint(1, 32),
                    random.randint(1, 4),
                )
                if bidirectional_encoder:
                    if hidden_size % 2 != 0:
                        hidden_size += 1

                seq_length = random.randint(1, 12)
                lengths = [random.randint(1, seq_length) for _ in range(batch_size)]
                lengths[0] = seq_length

                knn_indices_path = os.path.join(FOLDER, "test_knn_indices.npy")
                knn_distances_path = os.path.join(FOLDER, "test_knn_distances.npy")

                k = random.randint(1, 12)
                np.save(
                    knn_indices_path,
                    np.random.randint(
                        low=0, high=len(t2vec_tokenizer), size=(len(t2vec_tokenizer), k)
                    ),
                )
                np.save(
                    knn_distances_path,
                    np.random.uniform(low=0, high=1000, size=(len(t2vec_tokenizer), k)),
                )

                model = T2VEC(
                    embedding_dim=embedding_dim,
                    hidden_size=hidden_size,
                    tokenizer=t2vec_tokenizer,
                    knn_distances_path=knn_distances_path,
                    knn_indices_path=knn_indices_path,
                    num_layers=num_layers,
                    bidirectional_encoder=bidirectional_encoder,
                    embedding_path=None,
                    freeze_embedding=freeze,
                    dropout=0.0,
                )

                src = torch.randint(
                    low=0, high=len(t2vec_tokenizer), size=(batch_size, seq_length)
                )
                target = torch.randint(
                    low=0, high=len(t2vec_tokenizer), size=(batch_size, seq_length + 1)
                )

                batch = T2VECSample(src, lengths, target)
                output = model.compute_loss(batch)
                assert isinstance(output[0].item(), float)
                assert output[1] == batch_size

                assert isinstance(
                    model.training_step(batch=batch, batch_idx=0).item(), float
                )
                assert isinstance(
                    model.validation_step(batch=batch, batch_idx=0).item(), float
                )

                assert model(batch).shape == (batch_size, hidden_size)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`hidden_size` should be an even number greater than 0 when `bidirectional_encoder` is True"
        ),
    ):
        T2VEC(
            embedding_dim=embedding_dim,
            hidden_size=3,
            tokenizer=t2vec_tokenizer,
            knn_distances_path=knn_distances_path,
            knn_indices_path=knn_indices_path,
            num_layers=num_layers,
            bidirectional_encoder=True,
            embedding_path=None,
            dropout=0.0,
        )

    with pytest.raises(
        ValueError, match=re.escape("seq_length of `target` shoud be greater than 1")
    ):
        seq_length = 1
        lengths = [random.randint(1, seq_length) for _ in range(batch_size)]
        lengths[0] = seq_length
        src = torch.randint(
            low=0, high=len(t2vec_tokenizer), size=(batch_size, seq_length)
        )
        target = torch.randint(
            low=0, high=len(t2vec_tokenizer), size=(batch_size, seq_length)
        )

        batch = T2VECSample(src, lengths, target)
        output = model.compute_loss(batch)
