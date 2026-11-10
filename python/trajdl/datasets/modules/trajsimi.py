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

import math
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler

from ... import Trajectory, generate_spatial_features
from ...common.samples import TrajSimiSample
from ...tokenizers.trajcl import TrajCLTokenizer
from ..arrow import TrajectoryDataset
from .abstract import BaseTrajectoryDataModule


class TrajSimiSplitDataset(Dataset):
    """Simple map-style dataset for TrajSimi splits."""

    def __init__(self, trajs: Sequence[Trajectory]):
        self.trajs = trajs

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Tuple[int, Trajectory]:
        return idx, self.trajs[idx]  # type: ignore


class TrajSimiDataModule(BaseTrajectoryDataModule):
    """
    DataModule for TrajSimi downstream task.

    Loads the `val_trajs` dataset and the corresponding distance matrix
    from `traj_simi_dict_{metric}.pkl`.
    Splits the data into train, eval, and test exactly as in the baseline.
    """

    def __init__(
        self,
        data_dir: str,
        metric: str,
        tokenizer: Union[str, TrajCLTokenizer],
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        num_cpus: int = 4,
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_cpus=num_cpus,
        )
        self.data_dir = Path(data_dir)
        self.metric = metric

        if isinstance(tokenizer, str):
            self.tokenizer = TrajCLTokenizer.load_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        # These will be populated in setup()
        self.train_trajs: Sequence[Trajectory] = []
        self.val_trajs: Sequence[Trajectory] = []
        self.test_trajs: Sequence[Trajectory] = []

        self.train_simi_matrix: Optional[torch.FloatTensor] = None
        self.val_simi_matrix: Optional[torch.FloatTensor] = None
        self.test_simi_matrix: Optional[torch.FloatTensor] = None

    def setup(self, stage: Optional[str] = None):
        """Load data and split."""
        # 1. Load trajectories
        ds = TrajectoryDataset.load(self.data_dir / "val_trajs")

        all_trajs = list(ds.iter_as_seqs())
        total = len(all_trajs)

        # Proportional split (70%/10%/20%) for both full and small datasets
        n_train = int(total * 0.7)
        n_eval = int(total * 0.1)
        self.train_trajs = all_trajs[:n_train]
        self.val_trajs = all_trajs[n_train : n_train + n_eval]
        self.test_trajs = all_trajs[n_train + n_eval :]

        # 2. Load distance matrices
        pkl_path = self.data_dir / f"traj_simi_dict_{self.metric}.pkl"
        with open(pkl_path, "rb") as f:
            trains_simi, evals_simi, tests_simi, max_distance = pickle.load(f)

        # Normalize matrices
        def normalize(mat_list):
            mat = torch.tensor(mat_list, dtype=torch.float32)
            # Original TrajCL does: (simi + simi.T) / max_distance
            mat = (mat + mat.T) / max_distance
            return mat

        self.train_simi_matrix = normalize(trains_simi)
        self.val_simi_matrix = normalize(evals_simi)
        self.test_simi_matrix = normalize(tests_simi)

        # Setup Datasets
        self.train_ds = TrajSimiSplitDataset(self.train_trajs)
        self.val_ds = TrajSimiSplitDataset(self.val_trajs)
        self.test_ds = TrajSimiSplitDataset(self.test_trajs)

    def train_dataloader(self) -> DataLoader:
        # 完美对齐原始 TrajCL 的 (N/B)^2 批次采样逻辑
        n_samples = len(self.train_ds)
        counts = math.ceil((n_samples / self.train_batch_size) ** 2)
        total_samples = counts * self.train_batch_size

        sampler = RandomSampler(
            self.train_ds, replacement=True, num_samples=total_samples
        )

        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            sampler=sampler,
            collate_fn=self._train_collate_fn,
            num_workers=self.num_cpus,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self._eval_collate_fn,
            num_workers=self.num_cpus,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self._eval_collate_fn,
            num_workers=self.num_cpus,
            pin_memory=True,
            drop_last=False,
        )

    def collate_function(self, ds) -> TrajSimiSample:
        """
        Abstract method implementation - not used by TrajSimiDataModule.

        TrajSimiDataModule uses custom _train_collate_fn and _eval_collate_fn
        directly in train_dataloader/val_dataloader/test_dataloader.
        This method is required to satisfy the abstract base class.
        """
        raise NotImplementedError(
            "TrajSimiDataModule does not use collate_function. "
            "Use _train_collate_fn or _eval_collate_fn instead."
        )

    def _process_trajectories(
        self, trajs: Sequence[Trajectory]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        tokenizer = self.tokenizer
        tokens_list = []
        spatial_list = []
        lengths = []

        boundary = tokenizer.web_grid.boundary  # type: ignore
        step_x = tokenizer.web_grid.step_x  # type: ignore
        # Match original TrajCL behavior for spatial features
        ref_length = step_x * 11 / 1.414

        for traj in trajs:
            result = tokenizer.encode_traj(traj, return_as="pt", return_coords=True)  # type: ignore
            spatial_features = generate_spatial_features(
                result.coords,
                boundary.min_x,
                boundary.max_x,
                boundary.min_y,
                boundary.max_y,
                ref_length,
            )
            spatial_tensor = torch.tensor(spatial_features, dtype=torch.float32)

            tokens_list.append(result.token_ids)
            spatial_list.append(spatial_tensor)
            lengths.append(len(result.token_ids))

        cell_seq = pad_sequence(
            tokens_list, batch_first=True, padding_value=tokenizer.pad
        )

        max_len = cell_seq.shape[1]
        spatial_feat = torch.zeros(len(spatial_list), max_len, 4)
        for i, feat in enumerate(spatial_list):
            spatial_feat[i, : feat.shape[0], :] = feat

        return cell_seq, spatial_feat, lengths

    def _train_collate_fn(self, batch: List[Tuple[int, Trajectory]]) -> TrajSimiSample:
        indices = [item[0] for item in batch]
        trajs = [item[1] for item in batch]

        cell_seq, spatial_feat, lengths = self._process_trajectories(trajs)

        # Get sub-matrix
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        assert self.train_simi_matrix is not None
        sub_simi = self.train_simi_matrix[idx_tensor][:, idx_tensor]

        return TrajSimiSample(
            cell_seq=cell_seq,
            spatial_feat=spatial_feat,
            lengths=lengths,
            sub_simi=sub_simi,  # type: ignore
        )

    def _eval_collate_fn(self, batch: List[Tuple[int, Trajectory]]) -> TrajSimiSample:
        # For eval/test, we don't need the sub_simi per batch, because we compute global metric
        trajs = [item[1] for item in batch]
        cell_seq, spatial_feat, lengths = self._process_trajectories(trajs)

        return TrajSimiSample(
            cell_seq=cell_seq,
            spatial_feat=spatial_feat,
            lengths=lengths,
            sub_simi=None,
        )
