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

import torch

from trajdl.algorithms.loc_pred.stlstm import STLSTM, STLSTMModule, STLSTMSample
from trajdl.tokenizers import LocSeqTokenizer
from trajdl.tokenizers.slot import Bucketizer


def generate_stlstm_samples(
    locseq_tokenizer: LocSeqTokenizer,
    ts_bucketizer: Bucketizer,
    loc_bucketizer: Bucketizer,
    embedding_dim: int,
    hidden_size: int,
    batch_size: int,
    num_timesteps: int,
) -> STLSTMSample:
    num_locs = len(locseq_tokenizer)
    loc_seq = torch.randint(low=0, high=num_locs, size=(batch_size, num_timesteps))
    td_upper_seq = torch.randint(
        low=0, high=ts_bucketizer.num_buckets, size=(batch_size, num_timesteps)
    )
    td_lower_seq = torch.randint(
        low=0, high=ts_bucketizer.num_buckets, size=(batch_size, num_timesteps)
    )
    sd_upper_seq = torch.randint(
        low=0, high=loc_bucketizer.num_buckets, size=(batch_size, num_timesteps)
    )
    sd_lower_seq = torch.randint(
        low=0, high=loc_bucketizer.num_buckets, size=(batch_size, num_timesteps)
    )
    valid_lengths = [random.randint(0, num_timesteps) for _ in range(batch_size)]
    valid_lengths[0] = num_timesteps

    labels = torch.randint(low=0, high=num_locs, size=(batch_size, num_timesteps))

    mask = torch.zeros(size=(batch_size, num_timesteps))
    for idx, length in enumerate(valid_lengths):
        mask[idx, :length] = 1

    return STLSTMSample(
        loc_seq=loc_seq,
        td_upper_seq=td_upper_seq,
        td_lower_seq=td_lower_seq,
        sd_upper_seq=sd_upper_seq,
        sd_lower_seq=sd_lower_seq,
        valid_lengths=valid_lengths,
        labels=labels,
        mask=mask,
    )


def test_st_lstm(locseq_tokenizer: LocSeqTokenizer):
    embedding_dim = 32
    hidden_size = 64
    batch_size = 8
    num_timesteps = 12

    ts_bucketizer = Bucketizer(lower_bound=0.0, upper_bound=50.0, num_buckets=10)
    loc_bucketizer = Bucketizer(lower_bound=0.0, upper_bound=100.0, num_buckets=10)

    sample = generate_stlstm_samples(
        locseq_tokenizer=locseq_tokenizer,
        ts_bucketizer=ts_bucketizer,
        loc_bucketizer=loc_bucketizer,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
    )

    model = STLSTM(
        tokenizer=locseq_tokenizer,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        ts_bucketizer=ts_bucketizer,
        loc_bucketizer=loc_bucketizer,
    )

    output, hidden = model(sample)
    assert output.shape == (batch_size, num_timesteps, hidden_size)
    assert hidden[0].shape == hidden[1].shape
    assert hidden[0].shape == (batch_size, hidden_size)

    h, c = model.init_hidden(batch_size=batch_size, device=torch.device("cpu"))
    assert h.shape == (batch_size, hidden_size)
    assert c.shape == (batch_size, hidden_size)
    assert (h == 0).all()
    assert (c == 0).all()


def test_stlstm_module(locseq_tokenizer: LocSeqTokenizer):
    embedding_dim = 32
    hidden_size = 64
    ts_bucketizer = Bucketizer(lower_bound=0.0, upper_bound=50.0, num_buckets=10)
    loc_bucketizer = Bucketizer(lower_bound=0.0, upper_bound=100.0, num_buckets=10)

    batch_size = 8
    num_timesteps = 12

    batch_sample = generate_stlstm_samples(
        locseq_tokenizer=locseq_tokenizer,
        ts_bucketizer=ts_bucketizer,
        loc_bucketizer=loc_bucketizer,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
    )

    for reduction in ["sum", "mean"]:
        for use_sampled_softmax in [True, False]:
            net = STLSTMModule(
                tokenizer=locseq_tokenizer,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                ts_bucketizer=ts_bucketizer,
                loc_bucketizer=loc_bucketizer,
                reduction=reduction,
                use_sampled_softmax=use_sampled_softmax,
            )

            output = net(batch_sample)
            assert output.shape == (batch_size, len(locseq_tokenizer))

            loss_value, b = net.compute_loss(sample=batch_sample)
            assert isinstance(loss_value.detach().cpu().item(), float)
            assert b == batch_size

            assert isinstance(
                net.training_step(batch=batch_sample, batch_idx=0)
                .detach()
                .cpu()
                .item(),
                float,
            )
            assert isinstance(
                net.validation_step(batch=batch_sample, batch_idx=0)
                .detach()
                .cpu()
                .item(),
                float,
            )

            assert net.forward(sample=batch_sample).shape == (
                batch_size,
                len(locseq_tokenizer),
            )
