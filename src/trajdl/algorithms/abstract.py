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

import lightning as L
import torch.optim as optim


class BaseLightningModel(L.LightningModule):
    """
    A base Lightning model that encapsulates optimizer configuration.

    Parameters
    ----------
    optimizer_type : str, optional
        The type of optimizer to use ("adam", "sgd", or "rmsprop"). Default is "adam".
    learning_rate : float, optional
        The learning rate for the optimizer. Default is 1e-3.
    """

    def __init__(self, optimizer_type="adam", learning_rate=1e-3):
        super(BaseLightningModel, self).__init__()
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        """
        Configures the optimizer for the model based on the specified optimizer type.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            The configured optimizer instance for training.

        Raises
        ------
        ValueError
            If an unsupported optimizer type is provided.
        """
        if self.optimizer_type == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        return optimizer
