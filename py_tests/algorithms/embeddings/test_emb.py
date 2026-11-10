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
from typing import List

import numpy as np
import pytest
import torch
from gensim.models import Word2Vec
from torch import nn
from tqdm import tqdm
from trajdl import Trajectory
from trajdl.algorithms.embeddings.base import (
    BaseTokenEmbeddingLayer,
    Node2VecEmbedding,
    SimpleEmbedding,
    Word2VecEmbedding,
)
from trajdl.common.enum import TokenEnum
from trajdl.tokenizers import AbstractTokenizer, T2VECTokenizer

from ...conftest import FOLDER

EMBEDDING_DIM = 256
WORD2VEC_MODEL_PATH = os.path.join(FOLDER, "t2vec", "work2vec.model")


@pytest.fixture(scope="session", autouse=True)
def test_word2vec(
    t2vec_tokenizer: T2VECTokenizer, traj_samples: List[Trajectory]
) -> Word2Vec:
    processed_sentences = [
        t2vec_tokenizer.traj_to_loc_seq(traj, add_bos=True, add_eos=True)
        for traj in tqdm(traj_samples, desc="process sentence for word2vec training")
    ]

    for sentence in processed_sentences:
        assert sentence[0] == TokenEnum.BOS_TOKEN.value
        assert sentence[-1] == TokenEnum.EOS_TOKEN.value

    model = Word2Vec(
        sentences=processed_sentences,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=1,
        sg=1,
        workers=1,
        seed=42,
        epochs=1,
    )

    # 这个用例在word2vec的语料里面增加了bos, eos, unk，所以缺失的两个是mask和pad
    assert len(model.wv) == len(t2vec_tokenizer) - 2

    os.makedirs(os.path.split(WORD2VEC_MODEL_PATH)[0], exist_ok=True)
    model.save(WORD2VEC_MODEL_PATH)

    return model


class NetForTestingEmb(nn.Module):
    def __init__(self, emb_layer: BaseTokenEmbeddingLayer, out_features: int):
        super().__init__()
        self.emb = emb_layer
        self.fc = nn.Linear(
            in_features=emb_layer.embedding_dim, out_features=out_features
        )

    def forward(self, src: torch.LongTensor):
        return self.fc(self.emb(src))


def check_freeze(emb_layer: BaseTokenEmbeddingLayer, tokenizer: AbstractTokenizer):
    batch_size = random.randint(2, 12)
    seq_length = random.randint(2, 12)
    src = torch.randint(low=0, high=len(tokenizer), size=(batch_size, seq_length))
    out_features = 3
    labels = torch.rand(size=(batch_size, seq_length, out_features))

    output = emb_layer(src)
    assert output.shape == (batch_size, seq_length, emb_layer.embedding_dim)

    net = NetForTestingEmb(emb_layer=emb_layer, out_features=out_features)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

    # test unfreeze
    for _ in range(10):
        output = net(src)
        assert output.shape == (batch_size, seq_length, out_features)
        loss = ((labels - output) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
        optimizer.step()

    assert not emb_layer.is_frozen
    for param in net.emb.parameters():
        assert param.requires_grad is True

    # test freeze
    emb_layer.freeze_parameters()
    assert emb_layer.is_frozen

    for _ in range(10):
        output = net(src)
        assert output.shape == (batch_size, seq_length, out_features)
        loss = ((labels - output) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
        optimizer.step()

    for param in net.emb.parameters():
        assert param.requires_grad is False

    # test unfreeze
    emb_layer.unfreeze_parameters()
    assert not emb_layer.is_frozen

    for _ in range(10):
        output = net(src)
        assert output.shape == (batch_size, seq_length, out_features)
        loss = ((labels - output) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.1)
        optimizer.step()

    for param in net.emb.parameters():
        assert param.requires_grad is True


def test_simple_embedding(t2vec_tokenizer: T2VECTokenizer):
    emb = SimpleEmbedding(tokenizer=t2vec_tokenizer, embedding_dim=EMBEDDING_DIM)
    params = list(emb.parameters())
    assert params[0].data.shape == (len(t2vec_tokenizer), EMBEDDING_DIM)
    assert len(params) == 1
    assert emb.embedding_dim == EMBEDDING_DIM

    check_freeze(emb_layer=emb, tokenizer=t2vec_tokenizer)


def test_word2vec_embedding(t2vec_tokenizer: T2VECTokenizer, test_word2vec: Word2Vec):
    emb_layer = Word2VecEmbedding(
        tokenizer=t2vec_tokenizer, model_path=WORD2VEC_MODEL_PATH
    )
    assert emb_layer.embedding_dim == EMBEDDING_DIM

    for word in test_word2vec.wv.index_to_key:
        idx = t2vec_tokenizer.token_to_id(word)
        word2vec_emb = test_word2vec.wv[word]
        torch_emb = emb_layer(torch.LongTensor([idx]))

        assert word2vec_emb.shape == (EMBEDDING_DIM,)
        assert torch_emb.shape == (1, EMBEDDING_DIM)

        assert np.allclose(
            word2vec_emb,
            torch_emb.detach()
            .cpu()
            .numpy()
            .reshape(
                -1,
            ),
            rtol=1e-6,
            atol=1e-6,
        )

    check_freeze(emb_layer=emb_layer, tokenizer=t2vec_tokenizer)


def test_node2vec_embedding():
    """Test Node2VecEmbedding with a simple TrajCLTokenizer setup."""
    from trajdl import Coordinate, GridCoord, RectangleBoundary, SimpleGridSystem
    from trajdl.tokenizers.trajcl import TrajCLTokenizer

    # Create a small grid system
    gps_boundary = RectangleBoundary(
        min_point=Coordinate.Wgs84(-8.7, 41.0),
        max_point=Coordinate.Wgs84(-8.5, 41.2),
    )
    web_mercator_boundary = gps_boundary.to_web_mercator()
    web_grid = SimpleGridSystem(
        boundary=web_mercator_boundary,
        step_x=500.0,
        step_y=500.0,
        column_major=True,
    )

    # Fit tokenizer (vocab now covers the entire grid; sample trajectories
    # are no longer needed for fitting)
    tokenizer = TrajCLTokenizer.fit(web_grid=web_grid)

    # Create mock cell_embeddings (one embedding per cell in the grid)
    total_cells = web_grid.num_x_grids * web_grid.num_y_grids
    emb_dim = 64
    cell_embeddings = torch.randn(total_cells, emb_dim)

    # Test Node2VecEmbedding
    emb_layer = Node2VecEmbedding(
        tokenizer=tokenizer,
        cell_embeddings=cell_embeddings,
        freeze=False,
    )

    assert emb_layer.embedding_dim == emb_dim

    # Test that grid symbol embeddings are properly aligned
    for token_id in range(len(tokenizer)):
        token = tokenizer.id_to_token(token_id)

        if token.startswith("G_"):
            # Grid symbol - should match cell_embeddings
            grid_coord = GridCoord.from_grid_symbol(token)
            cell_id = web_grid.locate_by_grid_coordinate(grid_coord)

            # Get embedding from layer
            output = emb_layer(torch.LongTensor([token_id]))
            expected = cell_embeddings[cell_id]

            assert torch.allclose(output.squeeze(), expected)
        else:
            # Special token - should be zeros for PAD, random for others
            output = emb_layer(torch.LongTensor([token_id]))

            if token == tokenizer.id_to_token(tokenizer.pad):
                assert torch.allclose(output.squeeze(), torch.zeros(emb_dim))
            else:
                # Non-PAD special tokens should have random initialization (non-zero)
                pass  # Just verify it doesn't crash

    # Test forward pass with batch
    batch_token_ids = torch.randint(0, len(tokenizer), (4, 10))
    output = emb_layer(batch_token_ids)

    assert output.shape == (4, 10, emb_dim)

    # Test freeze/unfreeze
    check_freeze(emb_layer=emb_layer, tokenizer=tokenizer)
