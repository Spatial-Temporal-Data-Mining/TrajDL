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

from typing import Dict

import torch


class AccMetrics:
    def __init__(self):
        self.num_correct = 0
        self.num_samples = 0

    def update(self, pred: torch.LongTensor, label: torch.LongTensor):
        self.num_correct += (pred == label).sum().item()
        self.num_samples += pred.shape[0]

    def value(self) -> Dict[str, float]:
        if self.num_samples == 0:
            return {}

        acc = self.num_correct * 100 / self.num_samples
        print("acc: {:.2f}%".format(acc))
        return {"Acc": acc}

    def reset(self):
        self.num_correct = 0
        self.num_samples = 0
