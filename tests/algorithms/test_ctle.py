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

import torch

from trajdl.algorithms.ctle import CTLETrainingFramework, MaskedLM
from trajdl.tokenizers.locseq import LocSeqTokenizer


def test_ctlm_mlm(locseq_tokenizer: LocSeqTokenizer):
    batch_size, seq_length, embedding_dim, max_len = 3, 6, 32, 100
    pad_value = locseq_tokenizer.pad

    src = torch.LongTensor(
        [
            [0, 1, 2, 3, pad_value, pad_value],
            [1, 2, 3, pad_value, pad_value, pad_value],
            [5, 6, 7, 8, 9, 10],
        ]
    )
    hour = torch.LongTensor(
        [
            [8, 9, 10, 11, pad_value, pad_value],
            [11, 12, 13, 14, pad_value, pad_value],
            [15, 16, 17, 18, 19, 20],
        ]
    )
    lengths = [4, 3, 6]

    mask_prob = 0.2
    mask = (torch.rand(size=(batch_size, seq_length)) < mask_prob) & (src != pad_value)

    x = torch.rand(size=(batch_size, seq_length, embedding_dim))

    mlm = MaskedLM(input_size=embedding_dim, output_size=len(locseq_tokenizer))
    assert isinstance(mlm(src, mask, x).item(), float)

    mh = MaskedLM(input_size=embedding_dim, output_size=24)
    assert isinstance(mh(hour, mask, x).item(), float)


def test_ctle(locseq_tokenizer: LocSeqTokenizer):
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

    ctle = CTLETrainingFramework(
        embedding_type="tp",
        embedding_dim=embedding_dim,
        max_len=max_len,
        num_layers=1,
        n_heads=8,
        tokenizer=locseq_tokenizer,
        hidden_size=hidden_size,
    )

    assert isinstance(ctle.compute_loss(loc_src, ts_src, mask).item(), float)
    assert isinstance(
        ctle.training_step(batch=(loc_src, ts_src, mask), batch_idx=0).item(), float
    )
    assert isinstance(
        ctle.validation_step(batch=(loc_src, ts_src, mask), batch_idx=0).item(), float
    )
