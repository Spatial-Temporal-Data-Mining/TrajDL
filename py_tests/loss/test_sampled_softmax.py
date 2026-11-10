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
from trajdl.loss.sampled_softmax import _EVAL_CHUNK_NUM_ELEMENTS, SampledSoftmaxLoss


def test_sampled_softmax():
    num_words = 100
    embedding_dim = 3
    weights = torch.nn.Parameter(torch.rand(size=(num_words, embedding_dim)))
    bias = torch.nn.Parameter(torch.zeros(size=(num_words, 1)))

    loss = SampledSoftmaxLoss(
        weights=weights,
        bias=bias,
        num_words=num_words,
        num_samples=64,
        reduction="mean",
    )
    embedding = torch.rand(size=(32, embedding_dim))
    targets = torch.randint(low=0, high=100, size=(32,))
    with torch.inference_mode():
        print(loss(embedding, targets))
    loss.train()
    print(loss(embedding, targets))


def test_forward_eval_chunked_equivalence(monkeypatch):
    """
    _forward_eval 分块计算与一次性全量计算逐位等价（STLSTM V100-16G OOM 修复的回归防线）
    """
    torch.manual_seed(42)
    num_words = 211
    embedding_dim = 8
    rows = 97

    weights = torch.nn.Parameter(torch.randn(num_words, embedding_dim))
    bias = torch.nn.Parameter(torch.randn(num_words, 1))
    loss = SampledSoftmaxLoss(
        weights=weights,
        bias=bias,
        num_words=num_words,
        num_samples=64,
        reduction="none",
    )
    loss.eval()

    embedding = torch.randn(rows, embedding_dim)
    targets = torch.randint(low=0, high=num_words, size=(rows,))
    mask = torch.rand(rows) > 0.3

    with torch.inference_mode():
        # 一次性全量参考实现（修复前的计算方式）
        logits = torch.matmul(embedding, loss.w.t()) + loss.b.t()
        expected = loss.eval_loss(logits, targets) * mask

        # 分块：强制小 chunk 覆盖多条边界（rows 不能被 chunk_rows 整除）
        monkeypatch.setattr(
            "trajdl.loss.sampled_softmax._EVAL_CHUNK_NUM_ELEMENTS", 1 << 10
        )
        chunked = loss._forward_eval(embedding, targets, mask)

        # 大 chunk（单块覆盖全部行，等价于修复前行为）
        monkeypatch.setattr(
            "trajdl.loss.sampled_softmax._EVAL_CHUNK_NUM_ELEMENTS",
            _EVAL_CHUNK_NUM_ELEMENTS,
        )
        single = loss._forward_eval(embedding, targets, mask)

    assert torch.equal(expected, single)
    assert torch.allclose(expected, chunked, rtol=0.0, atol=1e-5)
