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

from abc import ABC, abstractmethod
from typing import Union

import lightning as L

from ...common.enum import Mode
from ..abstract import BaseLightningModel


class PretrainTrainFramework(BaseLightningModel, ABC):
    """
    预训练+训练框架

    """

    def __init__(self, mode: str, optimizer_type="adam", learning_rate=1e-3):
        super().__init__(optimizer_type=optimizer_type, learning_rate=learning_rate)
        self.set_mode(mode)

    @property
    def mode(self) -> Mode:
        return self._mode

    def set_mode(self, mode: Union[Mode, str]) -> None:
        if isinstance(mode, Mode):
            self._mode = mode
        elif isinstance(mode, str):
            self._mode = Mode.from_string(mode)
        else:
            raise ValueError("`mode` should a str or a Mode instance")
        print(f"{self.mode} mode.")

    @abstractmethod
    def init_from_pretrained_ckpt(self):
        """
        这个方法是给定一个预训练checkpoint的目录，根据一些逻辑对训练阶段的模型进行初始化的工作
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        """
        这个方法需要根据mode进行loss的计算
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover
