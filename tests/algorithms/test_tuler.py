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

import random
import re

import pytest
import torch

from trajdl.algorithms.tuler import TULER
from trajdl.common.samples import TULERSample


def test_tuler(locseq_tokenizer):
    for _ in range(10):
        for rnn_type in ["lstm", "gru"]:
            for freeze in [True, False]:
                num_users = random.randint(2, 32)
                batch_size, max_seq_length = random.randint(1, 32), random.randint(
                    1, 12
                )

                src = torch.randint(
                    low=0, high=len(locseq_tokenizer), size=(batch_size, max_seq_length)
                )

                seq_len = [random.randint(1, max_seq_length) for _ in range(batch_size)]
                seq_len[0] = max_seq_length
                seq_len.sort(reverse=True)

                labels = torch.randint(low=0, high=num_users, size=(batch_size,))

                model = TULER(
                    tokenizer=locseq_tokenizer,
                    num_users=num_users,
                    embedding_dim=12,
                    hidden_dim=12,
                    rnn_type=rnn_type,
                    freeze_embedding=freeze,
                )

                batch = TULERSample(
                    src=src,
                    seq_len=seq_len,
                    labels=labels,
                )
                assert model(batch).shape == (batch_size, num_users)
                assert isinstance(
                    model.training_step(batch=batch, batch_idx=0).item(),
                    float,
                )
                assert isinstance(
                    model.validation_step(batch=batch, batch_idx=0).item(),
                    float,
                )

    with pytest.raises(
        ValueError, match=re.escape("`rnn_type` only support {'lstm', 'gru'}")
    ):
        TULER(
            tokenizer=locseq_tokenizer,
            num_users=3,
            embedding_dim=12,
            hidden_dim=12,
            rnn_type="test",
        )
