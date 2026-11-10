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

"""T2VEC Pair Dataset: pairs degraded trajectories with original trajectories.

Both src (degraded) and trg (original) are stored as ``LocSeqDataset``
(Arrow-backed, tokenized at preprocessing time). The src dataset carries a
feature table with a ``trg_idx`` column that maps each degraded sequence back
to its corresponding original sequence.

Storing the tokenized sequences (instead of raw coordinates) means the grid
discretization and kd-tree vocabulary remapping run exactly once during
preprocessing; training-time consumers (sampler bucketing, collate) only read
token sequences.

Usage::

    from trajdl.datasets import LocSeqDataset
    from trajdl.datasets.arrow.ext.t2vec import T2VECPairDataset

    trg_dataset = LocSeqDataset.load("output/train_trg_ds.parquet")
    src_dataset = LocSeqDataset.load("output/train_src_ds.parquet")
    dataset = T2VECPairDataset(src_dataset, trg_dataset)

    src_view, trg_view = dataset[0]
    # src_view: LocSeqView — degraded token sequence (zero-copy view)
    # trg_view: LocSeqView — original token sequence (zero-copy view)
"""

from typing import List, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from .... import LocSeqView
from ..locseq import LocSeqDataset


class T2VECPairDataset(Dataset):
    """PyTorch Dataset that pairs degraded (src) and original (trg) loc seqs.

    Returns ``LocSeqView`` objects (zero-copy Arrow views) whose ``seq`` is a
    plain list of token strings that can be passed directly to the
    tokenizer's ``encode_locseq`` in the DataModule collate.

    Parameters
    ----------
    src_dataset : LocSeqDataset
        Degraded token sequence dataset. Must have a feature table with a
        ``trg_idx`` column (int64) mapping each degraded record to an
        original sequence index.
    trg_dataset : LocSeqDataset
        The original token sequence dataset.
    """

    FEATURE_COL_TRG_IDX = "trg_idx"

    def __init__(self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset):
        super().__init__()
        # Validate by trying to access the feature column;
        # get_feature raises ValueError if feature_table is None or column missing.
        src_dataset.get_feature(0, self.FEATURE_COL_TRG_IDX)

        self.src_dataset = src_dataset
        self.trg_dataset = trg_dataset

    def __len__(self) -> int:
        return len(self.src_dataset)

    def __getitem__(self, idx: int) -> Tuple[LocSeqView, LocSeqView]:
        """Return a (src_view, trg_view) pair as LocSeqView objects.

        Parameters
        ----------
        idx : int
            Index into the src (degraded) dataset.

        Returns
        -------
        Tuple[LocSeqView, LocSeqView]
            (degraded_loc_seq_view, original_loc_seq_view)
        """
        trg_idx = self.src_dataset.get_feature(idx, self.FEATURE_COL_TRG_IDX)
        return self.src_dataset.get_view(idx), self.trg_dataset.get_view(trg_idx)

    def __getitems__(
        self, indices: Union[np.ndarray, List[int], List[np.ndarray]]
    ) -> List[Tuple[LocSeqView, LocSeqView]]:
        """Batch retrieval for DataLoader with batched indexing.

        Handles both flat index lists and nested index arrays (yielded by
        ``T2VECSampler`` through ``BatchSampler`` with ``batch_size=1``,
        which wraps each yielded array in a one-element list).
        """
        if all(isinstance(tmp, (int, np.integer)) for tmp in indices):
            new_indices = indices
        else:
            new_indices = (idx for index_list in indices for idx in index_list)
        return [self[i] for i in new_indices]
