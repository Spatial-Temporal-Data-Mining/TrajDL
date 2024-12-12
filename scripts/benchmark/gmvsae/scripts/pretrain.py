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

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI
from sklearn.cluster import KMeans

from trajdl.algorithms.gmvsae import GMVSAE
from trajdl.datasets.modules.gmvsae import GMVSAEDataModule


class GMVSAEPretrainLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--pretrained_mu_c_path", default="output/gmvsae/pretrained_init_mu_c.npy"
        )

    def after_fit(self):
        dataloader = self.datamodule.val_dataloader()
        with torch.inference_mode():
            z_list = self.trainer.predict(
                model=None, dataloaders=dataloader, ckpt_path="best"
            )
        z_list = np.concatenate([i.detach().cpu().numpy() for i in z_list])

        kmeans = KMeans(n_clusters=self.config.fit.model.mem_num, verbose=1)
        kmeans.fit(z_list)
        init_mu_c = kmeans.cluster_centers_

        np.save(self.config.fit.pretrained_mu_c_path, init_mu_c)


def pretrain():
    GMVSAEPretrainLightningCLI(model_class=GMVSAE, datamodule_class=GMVSAEDataModule)


if __name__ == "__main__":
    pretrain()
