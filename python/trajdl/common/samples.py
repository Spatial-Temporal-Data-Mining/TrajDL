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

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TULERSample:
    """
    TULER input sample

    Parameters
    ----------
    src: torch.LongTensor
        shape is (B, T)

    seq_len: List[int]
        length of each sequence

    labels: torch.LongTensor, optional
        shape is (B,), label of each sequence, default is None

    """

    src: torch.LongTensor
    seq_len: List[int]
    labels: Optional[torch.LongTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.seq_len)


@dataclass
class T2VECSample:
    """
    T2VEC input sample

    Parameters
    ----------
    src: torch.LongTensor
        shape is (batch_size, seq_length)

    lengths: List[int]
        length of each src sequence

    target: torch.LongTensor, optional
        shape is (batch_size, seq_length'), default is None
    """

    src: torch.LongTensor
    lengths: List[int]
    target: Optional[torch.LongTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.lengths)


@dataclass
class GMVSAESample:
    """
    GMVSAE input sample

    Parameters
    ----------
    encoder_seq: torch.LongTensor
        shape is (B, T)

    encoder_lengths: List[int]
        length of each sequence

    decoder_seq: Optional[torch.LongTensor], optional
        shape is (B, T + 1), each of seq added BOS, default is None

    decoder_lengths: Optional[List[int]], optional
        encoder_lengths + 1, default is None

    decoder_labels: Optional[torch.LongTensor], optional
        shape is (B, T + 1), each of seq added EOS, default is None

    mask: Optional[torch.Tensor], optional
        shape is (B, T + 1), default is None

    """

    encoder_seq: torch.LongTensor
    encoder_lengths: List[int]
    decoder_seq: Optional[torch.LongTensor] = None
    decoder_lengths: Optional[List[int]] = None
    decoder_labels: Optional[torch.LongTensor] = None
    mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.encoder_lengths)


@dataclass
class STLSTMSample:
    """
    ST-LSTM input sample

    Parameters
    ----------
    loc_seq: torch.LongTensor
        shape is (B, T), location sequence

    td_upper_seq: torch.LongTensor
        shape is (B, T), batch of the upper bound minus the time difference
        between two consecutive timesteps

    td_lower_seq: torch.LongTensor
        shape is (B, T), batch of the time difference between two consecutive
        timesteps minus the lower bound

    sd_upper_seq: torch.LongTensor
        shape is (B, T), batch of the upper bound minus the spatial
        displacement between two consecutive timesteps

    sd_lower_seq: torch.LongTensor
        shape is (B, T), batch of the spatial displacement between two
        consecutive timesteps minus the lower bound

    valid_lengths: List[int]
        actual length of each sequence

    labels: torch.LongTensor, optional
        shape is (B, T), passed in during training; labels corresponding to
        the LSTM outputs

    mask: torch.LongTensor, optional
        shape is (B, T), passed in during training; mask over the LSTM
        outputs, where 1 marks a position whose loss is computed and 0 one
        whose loss is not
    """

    loc_seq: torch.LongTensor
    td_upper_seq: torch.LongTensor
    td_lower_seq: torch.LongTensor
    sd_upper_seq: torch.LongTensor
    sd_lower_seq: torch.LongTensor
    valid_lengths: List[int]
    labels: Optional[torch.LongTensor] = None
    mask: Optional[torch.LongTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.valid_lengths)


@dataclass
class TrajCLSample:
    """
    Sample of TrajCL for contrastive learning.

    Parameters
    ----------
    view1_cell_seq : torch.LongTensor
        shape is (B, T1), token IDs for view 1

    view1_spatial_feat : torch.FloatTensor
        shape is (B, T1, 4), spatial features for view 1

    view1_lengths : List[int]
        length of each sequence in view 1

    view2_cell_seq : torch.LongTensor
        shape is (B, T2), token IDs for view 2

    view2_spatial_feat : torch.FloatTensor
        shape is (B, T2, 4), spatial features for view 2

    view2_lengths : List[int]
        length of each sequence in view 2
    """

    view1_cell_seq: torch.LongTensor
    view1_spatial_feat: torch.FloatTensor
    view1_lengths: List[int]
    view2_cell_seq: torch.LongTensor
    view2_spatial_feat: torch.FloatTensor
    view2_lengths: List[int]

    @property
    def batch_size(self) -> int:
        return len(self.view1_lengths)


@dataclass
class TrajSimiSample:
    """
    Sample of TrajSimi downstream task.

    Parameters
    ----------
    cell_seq : torch.LongTensor
        shape is (B, T), token IDs for trajectories

    spatial_feat : torch.FloatTensor
        shape is (B, T, 4), spatial features for trajectories

    lengths : List[int]
        length of each sequence

    sub_simi : Optional[torch.FloatTensor]
        shape is (B, B), ground truth distance matrix for the batch, default is None
    """

    cell_seq: torch.LongTensor
    spatial_feat: torch.FloatTensor
    lengths: List[int]
    sub_simi: Optional[torch.FloatTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.lengths)


@dataclass
class TrjSRSample:
    """TrjSR input sample: a batch of LR and HR trajectory images.

    Parameters
    ----------
    lr_images : torch.FloatTensor
        Low-resolution images, shape ``(B, 1, H_lr, W_lr)``.

    hr_images : torch.FloatTensor
        High-resolution images, shape ``(B, 1, H_hr, W_hr)``.
    """

    lr_images: torch.FloatTensor
    hr_images: torch.FloatTensor

    @property
    def batch_size(self) -> int:
        return self.lr_images.size(0)


@dataclass
class TALEPretrainSample:
    """TALE pre-training batch: flat (context, slot) pair rows (row-level batch unit, #116 semantics).

    Parameters
    ----------
    pos_u : torch.LongTensor
        Context token ids, shape ``(B, W)`` (W=1 when ``indi_context=True``).

    pos_w : torch.LongTensor
        Huffman internal-node path for code=1, zero-padded, shape ``(B, L1)``.

    neg_w : torch.LongTensor
        Path for code=0, zero-padded, shape ``(B, L2)``.

    pos_s : torch.LongTensor
        Time-slot id, shape ``(B,)``.

    prob : torch.FloatTensor
        Share of the row's time slot (the loss is weighted per row), shape ``(B,)``.
    """

    pos_u: torch.LongTensor
    pos_w: torch.LongTensor
    neg_w: torch.LongTensor
    pos_s: torch.LongTensor
    prob: torch.FloatTensor

    @property
    def batch_size(self) -> int:
        return self.pos_u.size(0)


@dataclass
class NextLocPredSample:
    """Downstream next-loc prediction batch (produced by NextLocPredDataModule, shared by TALE/ERPP/GRU).

    Parameters
    ----------
    full_seq : torch.LongTensor
        Growing-prefix history, right-padded with 0 to the batch width, shape ``(B, T)``.

    valid_len : torch.LongTensor
        Length of the original unpadded sequence, shape ``(B,)``.

    trg_seq : torch.LongTensor
        Subsequent locations to be predicted, shape ``(B, pre_len)``.

    ts_src : torch.LongTensor
        Timestamps aligned with full_seq, shape ``(B, T)``.

    pre_len : int
        Number of future locations to predict.
    """

    full_seq: torch.LongTensor
    valid_len: torch.LongTensor
    trg_seq: torch.LongTensor
    ts_src: torch.LongTensor
    pre_len: int

    @property
    def batch_size(self) -> int:
        return self.full_seq.size(0)


@dataclass
class DeepMoveSample:
    """DeepMove batch (produced by DeepMoveDataModule).

    Semantics aligned with the authors' main.py::generate_input_long_history / run_simple:

    - long-history models (attn_avg_long_user / attn_local_long): the first
      ``src_hist_len`` columns of ``src`` are the long-term history (flattened
      in session order, not re-sorted by time), followed by the current
      session[:-1]; ``target`` = current session[1:].
    - simple models: ``src`` = current session[:-1], ``src_hist_len = 0``,
      the hist fields are empty placeholders.
    - The ``hist_*`` group fields are used only by attn_avg_long_user:
      ``hist_loc`` holds the flattened history loc ids, ``hist_group`` gives
      each record's mean-pool group id (adjacent records with equal tim48 are
      merged), and ``hist_tim_group`` is the tim48 of each group's first
      record (author semantics: the tim embedding of a group is taken from
      its first record).
    - Loss = per-session mean NLL over valid positions, then an equally
      weighted average across sessions.

    All id tensors use right padding with the tokenizer's pad id; masks mark
    valid positions.
    """

    src: torch.LongTensor  # (B, L)
    src_tim: torch.LongTensor  # (B, L)
    src_mask: torch.BoolTensor  # (B, L)
    src_hist_len: torch.LongTensor  # (B,)
    target: torch.LongTensor  # (B, T)
    target_mask: torch.BoolTensor  # (B, T)
    hist_loc: torch.LongTensor  # (B, H)
    hist_mask: torch.BoolTensor  # (B, H)
    hist_group: torch.LongTensor  # (B, H)
    hist_group_len: torch.LongTensor  # (B, G)
    hist_group_mask: torch.BoolTensor  # (B, G)
    hist_tim_group: torch.LongTensor  # (B, G)
    uid: torch.LongTensor  # (B,)

    @property
    def batch_size(self) -> int:
        return self.src.size(0)

    def to(self, device: torch.device) -> "DeepMoveSample":
        return DeepMoveSample(
            **{name: getattr(self, name).to(device) for name in self.__dataclass_fields__}
        )


@dataclass
class UniTrajSample:
    """
    UniTraj pre-training input sample

    Parameters
    ----------
    trajectory: torch.Tensor
        shape is (B, 2, T), normalized trajectory coordinates (after first-point centering)
    attention_mask: torch.BoolTensor
        shape is (B, T), valid-point mask
    original: torch.Tensor
        shape is (B, 2), first-point coordinates (the translation applied before centering)
    intervals: torch.Tensor
        shape is (B, T), time intervals
    indices: torch.LongTensor
        shape is (B, M), indices of the masked points (a list before stacking the variable-length rows)
    """

    trajectory: torch.Tensor
    attention_mask: torch.Tensor
    original: torch.Tensor
    intervals: torch.Tensor
    indices: torch.LongTensor

    @property
    def batch_size(self) -> int:
        return self.trajectory.size(0)


@dataclass
class BluePretrainSample:
    """
    BLUE pre-training input sample (the output of blue_collate with with_label=False)

    Parameters
    ----------
    x: torch.Tensor
        shape is (B, L, 6), [normalized lon/lat, fwd/bk dist(km), fwd/bk az, time]
    time_x: torch.Tensor
        shape is (B, L, 6), time features
    mask_y: torch.Tensor
        shape is (B, L), point-level mask (reconstruction target)
    traj_len_s5: torch.LongTensor
        shape is (B,), trajectory lengths at the s5 level
    patch_len_s3: torch.LongTensor
        flat (sum(traj_len_s3),), patch lengths at the s3 level
    traj_len_s3: torch.LongTensor
        shape is (B,), number of tokens at the s3 level
    patch_len_s2: torch.LongTensor
        flat (sum(traj_len_s2),), patch lengths at the s2 level
    traj_len_s2: torch.LongTensor
        shape is (B,), number of tokens at the s2 level
    """

    x: torch.Tensor
    time_x: torch.Tensor
    mask_y: torch.Tensor
    traj_len_s5: torch.LongTensor
    patch_len_s3: torch.LongTensor
    traj_len_s3: torch.LongTensor
    patch_len_s2: torch.LongTensor
    traj_len_s2: torch.LongTensor

    @property
    def batch_size(self) -> int:
        return self.x.size(0)


@dataclass
class TransferTrajSample:
    """
    TransferTraj pre-training input sample (product of transfertraj_pretrain_collate)

    Parameters
    ----------
    input_seq: torch.Tensor
        shape is (B, L, 4, 2), four-channel features with value + token dual channels
    target_seq: torch.Tensor
        shape is (B, L, 4, 2), reconstruction target
    positions: torch.LongTensor
        shape is (B, L), positional encodings
    first_point: torch.Tensor
        shape is (B, 2), absolute UTM first-point coordinates
    """

    input_seq: torch.Tensor
    target_seq: torch.Tensor
    positions: torch.LongTensor
    first_point: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.input_seq.size(0)


@dataclass
class CtlePretrainSample:
    """
    CTLE pre-training (MLM + hour prediction) input sample (product of CTLEDataModule.collate_function)

    Parameters
    ----------
    loc_src: torch.LongTensor
        shape is (B, T), location token sequence
    ts_src: torch.LongTensor
        shape is (B, T), timestamps in epoch seconds
    mask: torch.BoolTensor
        shape is (B, T), masked positions (reconstruction target)
    """

    loc_src: torch.LongTensor
    ts_src: torch.LongTensor
    mask: torch.BoolTensor

    @property
    def batch_size(self) -> int:
        return self.loc_src.size(0)
