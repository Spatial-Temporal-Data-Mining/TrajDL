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
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from ... import (
    CoordinateSystem,
    Trajectory,
    TrajectoryView,
    generate_spatial_features,
    mask_trajectory,
    rdp,
    seed_augmentation_rng,
    subset_trajectory,
    truncated_gaussian_distort,
)
from ...common.samples import TrajCLSample
from ...tokenizers.trajcl import TrajCLTokenizer
from ..arrow import TrajectoryDataset
from .abstract import BaseTrajectoryDataModule


def trajcl_augment(traj: Trajectory, aug_type: str, **kwargs) -> Trajectory:
    """Apply a single augmentation to a trajectory.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory
    aug_type : str
        Augmentation type: 'subset', 'mask', 'shift', 'simplify', or 'straight'.
        Matches all five augmentation types from the original TrajCL implementation.
    **kwargs : dict
        Additional arguments for the augmentation function

    Returns
    -------
    Trajectory
        Augmented trajectory

    Raises
    ------
    ValueError
        If aug_type is not one of 'subset', 'mask', 'shift', 'simplify', 'straight'
    """
    if aug_type == "subset":
        # subset_ratio: fraction of points to KEEP (e.g., 0.7 means keep 70%)
        ratio = kwargs.get("subset_ratio", kwargs.get("ratio", 0.7))
        return subset_trajectory(traj, ratio)
    elif aug_type == "mask":
        # mask_ratio: fraction of points to REMOVE (e.g., 0.3 means remove 30%)
        ratio = kwargs.get("mask_ratio", kwargs.get("ratio", 0.3))
        min_length = kwargs.get("min_length", 2)
        return mask_trajectory(traj, ratio, min_length)
    elif aug_type == "shift":
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 0.5)
        factor = kwargs.get("factor", 100.0)
        bound = kwargs.get("bound", 100.0)
        # rate=1.0 means all points are distorted (same as original shift_trajectory behavior)
        return truncated_gaussian_distort(traj, mu, sigma, factor, bound, rate=1.0)
    elif aug_type == "simplify":
        # RDP simplification, matching TrajCL's traj_simp_dist=100
        epsilon = kwargs.get("simplify_epsilon", kwargs.get("epsilon", 100.0))
        return rdp(traj, epsilon)
    elif aug_type == "straight":
        # Identity augmentation (no-op), returns a copy of the input trajectory.
        # Used as an anchor view in contrastive learning.
        # The collate loop feeds TrajectoryView objects (from iter_as_seqs),
        # which have no to_container; convert through the coord-system-specific
        # container conversion instead.
        if traj.coord_system == CoordinateSystem.WebMercator:
            return traj.to_web_mercator()
        return traj.to_wgs84()
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def trajcl_random_augment(
    traj: Trajectory,
    aug_types: List[str],
    **kwargs,
) -> Trajectory:
    """Randomly select and apply one augmentation.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory (Web Mercator coordinate system)
    aug_types : List[str]
        List of augmentation types to randomly choose from
    **kwargs : dict
        Additional arguments for the augmentation function

    Returns
    -------
    Trajectory
        Augmented trajectory
    """
    aug_type = random.choice(aug_types)
    return trajcl_augment(traj, aug_type, **kwargs)


@dataclass
class TrajCLDataModule(BaseTrajectoryDataModule):
    """
    DataModule for TrajCL contrastive learning.

    This DataModule implements the collate function for TrajCL, which:
    1. Applies two augmentations to each trajectory to create two views
    2. Converts trajectories to token sequences
    3. Generates spatial features for each view
    4. Returns padded batches with TrajCLSample

    Sampling strategy matches original TrajCL: standard DataLoader with
    shuffle=True (training) and drop_last=True, iterating over the full
    dataset each epoch.

    Parameters
    ----------
    tokenizer : Union[str, TrajCLTokenizer]
        Path to tokenizer or tokenizer instance

    aug_types : List[str], optional
        List of augmentation types to randomly choose from.
        Default is ['subset', 'mask', 'shift']

    aug_kwargs : dict, optional
        Additional kwargs for augmentation functions.

    aug_seed : int, optional
        If set, seeds the shared RNG used by the Rust augmentation ops
        (subset/mask/shift), making the augmentation sequence identical
        across runs. Mirrors the original TrajCL protocol of seeding the
        global RNG.
    """

    aug_types: List[str] = None
    aug_kwargs: dict = None
    pin_memory: bool = True
    aug_seed: int = None

    def __post_init__(self):
        super().__post_init__()
        if self.aug_seed is not None:
            seed_augmentation_rng(self.aug_seed)
        if self.aug_types is None:
            self.aug_types = ["subset", "mask", "shift"]
        if self.aug_kwargs is None:
            self.aug_kwargs = {
                "subset_ratio": 0.7,
                "mask_ratio": 0.3,
                "min_length": 2,
                "mu": 0.0,
                "sigma": 0.5,
                "factor": 100.0,
                "bound": 100.0,
            }

    def setup(self, stage: str):
        super().setup(stage=stage)

    def train_dataloader(self):
        """Training dataloader for TrajCL.

        Original TrajCL uses ``shuffle=False, drop_last=True``.
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            collate_fn=self.collate_function,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_cpus,
            pin_memory=self.pin_memory,
        )

    def _process_single_view(
        self,
        traj: Union[Trajectory, TrajectoryView],
        tokenizer: TrajCLTokenizer,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, int]:
        """Process a single trajectory into token sequence and spatial features.

        Parameters
        ----------
        traj : Union[Trajectory, TrajectoryView]
            Input trajectory (Web Mercator)
        tokenizer : TrajCLTokenizer
            Tokenizer for encoding

        Returns
        -------
        Tuple[torch.LongTensor, torch.FloatTensor, int]
            (token_ids, spatial_features, length)
        """
        # Encode trajectory with coordinates
        # Note: tokenizer expects Trajectory, so convert if needed
        result = tokenizer.encode_traj(traj, return_as="pt", return_coords=True)

        # Generate spatial features
        # Original TrajCL uses trajcl_local_mask_sidelen = cell_size * 11 for distance normalization
        # ref_length = trajcl_local_mask_sidelen / sqrt(2) = step_x * 11 / sqrt(2)
        boundary = tokenizer.web_grid.boundary
        step_x = tokenizer.web_grid.step_x
        ref_length = step_x * 11 / 1.414  # Match original TrajCL behavior

        spatial_features = generate_spatial_features(
            result.coords,
            boundary.min_x,
            boundary.max_x,
            boundary.min_y,
            boundary.max_y,
            ref_length,
        )
        spatial_tensor = torch.tensor(spatial_features, dtype=torch.float32)

        return result.token_ids, spatial_tensor, len(result.token_ids)

    def collate_function(self, ds: TrajectoryDataset) -> TrajCLSample:
        """
        Collate function for TrajCL training.

        For each trajectory in the batch:
        1. Apply two random augmentations to create two views
        2. Encode each view to token sequences
        3. Generate spatial features for each view
        4. Pad sequences to batch tensor

        Parameters
        ----------
        ds : TrajectoryDataset
            Dataset containing trajectories

        Returns
        -------
        TrajCLSample
            Batch sample with two views
        """
        tokenizer: TrajCLTokenizer = self.tokenizer

        view1_tokens: List[torch.LongTensor] = []
        view1_spatial: List[torch.FloatTensor] = []
        view1_lengths: List[int] = []

        view2_tokens: List[torch.LongTensor] = []
        view2_spatial: List[torch.FloatTensor] = []
        view2_lengths: List[int] = []

        for traj_view in ds.iter_as_seqs():
            # Apply two augmentations driven by aug_types[0] and aug_types[1]
            # Matches original TrajCL: trajcl_aug1='mask', trajcl_aug2='subset'
            aug1 = trajcl_augment(traj_view, self.aug_types[0], **self.aug_kwargs)
            aug2 = trajcl_augment(traj_view, self.aug_types[1], **self.aug_kwargs)

            # Process both views
            tokens1, spatial1, len1 = self._process_single_view(aug1, tokenizer)
            tokens2, spatial2, len2 = self._process_single_view(aug2, tokenizer)

            view1_tokens.append(tokens1)
            view1_spatial.append(spatial1)
            view1_lengths.append(len1)

            view2_tokens.append(tokens2)
            view2_spatial.append(spatial2)
            view2_lengths.append(len2)

        # Pad sequences
        view1_cell_seq = pad_sequence(
            view1_tokens, batch_first=True, padding_value=tokenizer.pad
        )
        view2_cell_seq = pad_sequence(
            view2_tokens, batch_first=True, padding_value=tokenizer.pad
        )

        # Pad spatial features (use zeros for padding)
        max_len1 = view1_cell_seq.shape[1]
        max_len2 = view2_cell_seq.shape[1]

        view1_spatial_feat = torch.zeros(len(view1_spatial), max_len1, 4)
        for i, feat in enumerate(view1_spatial):
            view1_spatial_feat[i, : feat.shape[0], :] = feat

        view2_spatial_feat = torch.zeros(len(view2_spatial), max_len2, 4)
        for i, feat in enumerate(view2_spatial):
            view2_spatial_feat[i, : feat.shape[0], :] = feat

        return TrajCLSample(
            view1_cell_seq=view1_cell_seq,
            view1_spatial_feat=view1_spatial_feat,
            view1_lengths=view1_lengths,
            view2_cell_seq=view2_cell_seq,
            view2_spatial_feat=view2_spatial_feat,
            view2_lengths=view2_lengths,
        )

    def collate_function_for_inference(self, ds: TrajectoryDataset) -> TrajCLSample:
        """
        Collate function for inference (no augmentation).

        Uses the original trajectory for both views.

        Parameters
        ----------
        ds : TrajectoryDataset
            Dataset containing trajectories

        Returns
        -------
        TrajCLSample
            Batch sample with identical views
        """
        tokenizer: TrajCLTokenizer = self.tokenizer

        view1_tokens: List[torch.LongTensor] = []
        view1_spatial: List[torch.FloatTensor] = []
        view1_lengths: List[int] = []

        for traj_view in ds.iter_as_seqs():
            # Pass view directly to _process_single_view (it handles conversion internally)
            tokens, spatial, length = self._process_single_view(traj_view, tokenizer)

            view1_tokens.append(tokens)
            view1_spatial.append(spatial)
            view1_lengths.append(length)

        view1_cell_seq = pad_sequence(
            view1_tokens, batch_first=True, padding_value=tokenizer.pad
        )

        max_len1 = view1_cell_seq.shape[1]
        view1_spatial_feat = torch.zeros(len(view1_spatial), max_len1, 4)
        for i, feat in enumerate(view1_spatial):
            view1_spatial_feat[i, : feat.shape[0], :] = feat

        # For inference, both views are identical
        return TrajCLSample(
            view1_cell_seq=view1_cell_seq,
            view1_spatial_feat=view1_spatial_feat,
            view1_lengths=view1_lengths,
            view2_cell_seq=view1_cell_seq.clone(),
            view2_spatial_feat=view1_spatial_feat.clone(),
            view2_lengths=view1_lengths.copy(),
        )
