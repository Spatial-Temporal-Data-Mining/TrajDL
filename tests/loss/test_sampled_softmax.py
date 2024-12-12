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

from trajdl.loss.sampled_softmax import SampledSoftmaxLoss


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
