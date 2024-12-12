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

import pytest
import torch

from trajdl.algorithms.hier import HIER, HIEREmbedding, HIERSpatialEmbedding
from trajdl.common.enum import LossEnum
from trajdl.grid import HierarchyGridSystem
from trajdl.tokenizers import SimpleTokenizer

from ..conftest import FOLDER


def test_hier(test_hierarchy: HierarchyGridSystem):
    tokenizer: SimpleTokenizer = test_hierarchy.build_simple_tokenizer()
    tokenizer_path = os.path.join(FOLDER, "hier", "tokenizer.pkl")
    tokenizer.save_pretrained(tokenizer_path)

    sizes = [12, 20, 32]

    for _ in range(10):
        num_vocab = len(tokenizer)
        loc_embedding_dim = random.randint(1, 64)
        week_embedding_dim = random.randint(1, 64)
        hour_embedding_dim = random.randint(1, 64)
        duration_embedding_dim = random.randint(1, 64)

        batch_size = random.randint(1, 64)
        seq_length = random.randint(1, 64)

        emb = HIERSpatialEmbedding(
            num_vocab=num_vocab, sizes=sizes, h_grid=test_hierarchy
        )
        x = torch.randint(low=0, high=num_vocab, size=(batch_size, seq_length))

        emb.hierarchical_avg(test_hierarchy)
        assert emb(x).shape == (batch_size, seq_length, sum(sizes))

        emb = HIEREmbedding(
            h_grid=test_hierarchy,
            location_embedding_dims=sizes,
            num_vocab=num_vocab,
            week_embedding_dim=week_embedding_dim,
            hour_embedding_dim=hour_embedding_dim,
            duration_embedding_dim=duration_embedding_dim,
            dropout=0.1,
        )

        lengths = [random.randint(1, seq_length) for _ in range(batch_size)]
        lengths[0] = seq_length

        src = torch.randint(low=0, high=num_vocab, size=(batch_size, seq_length))
        week = torch.randint(low=0, high=7, size=(batch_size, seq_length))
        hour = torch.randint(low=0, high=24, size=(batch_size, seq_length))
        duration = torch.randint(low=0, high=24, size=(batch_size, seq_length))
        targets = src

        emb_size = (
            sum(sizes)
            + week_embedding_dim
            + hour_embedding_dim
            + duration_embedding_dim
        )

        assert emb(src, week, hour, duration).shape == (
            batch_size,
            seq_length,
            emb_size,
        )
        assert emb.embedding_dim == emb_size

        model = HIER(
            tokenizer_path=tokenizer_path,
            hidden_size=64,
            num_layers=2,
            h_grid=test_hierarchy,
            location_embedding_dims=sizes,
        )
        assert model(src, week, hour, duration, lengths).shape == (
            batch_size,
            seq_length,
            num_vocab,
        )

        loss_value, b = model.forward_with_loss(
            (src, week, hour, duration, lengths, targets)
        )
        assert isinstance(loss_value.item(), float)
        assert b == batch_size

        assert isinstance(
            model.training_step(
                batch=(src, week, hour, duration, lengths, targets), batch_idx=0
            ).item(),
            float,
        )
        assert isinstance(
            model.validation_step(
                batch=(src, week, hour, duration, lengths, targets), batch_idx=0
            ).item(),
            float,
        )

    with pytest.raises(
        ValueError,
        match="Number of columns of grid and length of sizes should be equal.",
    ):
        HIERSpatialEmbedding(num_vocab=num_vocab, sizes=[1, 2], h_grid=test_hierarchy)

    reductions = [LossEnum.MEAN, LossEnum.SUM, "mean", "sum"]
    for reduction in reductions:
        model = HIER(
            tokenizer_path=tokenizer_path,
            hidden_size=64,
            num_layers=2,
            h_grid=test_hierarchy,
            location_embedding_dims=sizes,
            reduction=reduction,
        )
        loss_value, b = model.forward_with_loss(
            (src, week, hour, duration, lengths, targets)
        )
        assert isinstance(loss_value.item(), float)
        assert b == batch_size

    reductions = [LossEnum.NONE, "none"]
    for reduction in reductions:
        model = HIER(
            tokenizer_path=tokenizer_path,
            hidden_size=64,
            num_layers=2,
            h_grid=test_hierarchy,
            location_embedding_dims=sizes,
            reduction=reduction,
        )
        loss_value, b = model.forward_with_loss(
            (src, week, hour, duration, lengths, targets)
        )
        assert loss_value.shape == (batch_size, seq_length)
        assert b == batch_size
