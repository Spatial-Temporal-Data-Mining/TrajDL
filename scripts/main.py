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

from lightning.pytorch.cli import LightningCLI

from trajdl.algorithms.ctle import CTLETrainingFramework  # noqa: F401
from trajdl.algorithms.gmvsae import GMVSAE  # noqa: F401
from trajdl.algorithms.hier import HIER  # noqa: F401
from trajdl.algorithms.loc_pred.stlstm import STLSTMModule  # noqa: F401
from trajdl.algorithms.t2vec import T2VEC  # noqa: F401
from trajdl.algorithms.tuler import TULER  # noqa: F401
from trajdl.datasets.modules.ctle import CTLEDataModule  # noqa: F401
from trajdl.datasets.modules.gmvsae import GMVSAEDataModule  # noqa: F401
from trajdl.datasets.modules.hier import HIERDataModule  # noqa: F401
from trajdl.datasets.modules.stlstm import STLSTMDataModule  # noqa: F401
from trajdl.datasets.modules.t2vec import T2VECDataModule  # noqa: F401
from trajdl.datasets.modules.tuler import TULERDataModule  # noqa: F401


def cli_main():
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
