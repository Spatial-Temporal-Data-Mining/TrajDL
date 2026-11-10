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

"""Tests for T2VECPairDataset."""

from typing import List

import pyarrow as pa
import pytest
from trajdl import LocSeq, LocSeqView
from trajdl.datasets.arrow.ext.t2vec import T2VECPairDataset
from trajdl.datasets.arrow.locseq import LocSeqDataset


@pytest.fixture
def trg_dataset() -> LocSeqDataset:
    """Target dataset with 10 original loc seqs (20 tokens each)."""
    seqs = [
        LocSeq.from_list([f"loc_{i}_{j}" for j in range(20)], entity_id=str(i))
        for i in range(10)
    ]
    return LocSeqDataset.init_from_seqs(seqs)


@pytest.fixture
def src_dataset(trg_dataset: LocSeqDataset) -> LocSeqDataset:
    """Source (degraded) dataset: 3 degraded versions per trg seq (30 total)."""
    src_seqs: List[LocSeq] = []
    trg_idx_list: List[int] = []

    for trg_idx in range(len(trg_dataset)):
        tokens = trg_dataset.get_view(trg_idx).seq

        # Simulate 3 degradation variants per original sequence
        for variant in range(3):
            src_seqs.append(
                LocSeq.from_list(tokens[:15], entity_id=f"{trg_idx}-{variant}")
            )
            trg_idx_list.append(trg_idx)

    src_ds = LocSeqDataset.init_from_seqs(src_seqs)
    feature_table = pa.table({"trg_idx": pa.array(trg_idx_list, type=pa.int64())})
    src_ds.set_feature_table(feature_table)
    return src_ds


class TestT2VECPairDataset:
    def test_init_valid(self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset):
        dataset = T2VECPairDataset(src_dataset, trg_dataset)
        assert len(dataset) == 30  # 10 trg * 3 degradations

    def test_init_no_feature_table(self, trg_dataset: LocSeqDataset):
        """Should raise ValueError if src_dataset has no feature table."""
        with pytest.raises(ValueError):
            T2VECPairDataset(trg_dataset, trg_dataset)

    def test_init_missing_trg_idx_column(self, trg_dataset: LocSeqDataset):
        """Should raise ValueError if feature table lacks trg_idx column."""
        src_ds = LocSeqDataset.init_from_seqs(
            [LocSeq.from_list(["a", "b"]) for _ in range(len(trg_dataset))]
        )
        src_ds.set_feature_table(
            pa.table(
                {"wrong_col": pa.array(list(range(len(trg_dataset))), type=pa.int64())}
            )
        )
        with pytest.raises(ValueError):
            T2VECPairDataset(src_ds, trg_dataset)

    def test_getitem_returns_locseq_views(
        self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset
    ):
        dataset = T2VECPairDataset(src_dataset, trg_dataset)
        src_view, trg_view = dataset[0]
        assert isinstance(src_view, LocSeqView)
        assert isinstance(trg_view, LocSeqView)

    def test_getitem_correct_trg_mapping(
        self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset
    ):
        """Each src record should map to the correct trg loc seq."""
        dataset = T2VECPairDataset(src_dataset, trg_dataset)

        for src_idx in range(len(dataset)):
            src_view, trg_view = dataset[src_idx]
            expected_trg_idx = src_dataset.get_feature(src_idx, "trg_idx")
            expected_trg_view = trg_dataset.get_view(expected_trg_idx)

            assert len(trg_view) == len(expected_trg_view)
            assert trg_view.seq == expected_trg_view.seq

    def test_getitems_batch(
        self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset
    ):
        dataset = T2VECPairDataset(src_dataset, trg_dataset)
        results = dataset.__getitems__([0, 5, 15, 29])
        assert len(results) == 4
        for src_view, trg_view in results:
            assert isinstance(src_view, LocSeqView)
            assert isinstance(trg_view, LocSeqView)

    def test_src_view_length(
        self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset
    ):
        """Source views should have the expected degraded length (15 tokens)."""
        dataset = T2VECPairDataset(src_dataset, trg_dataset)
        src_view, _ = dataset[0]
        assert len(src_view) == 15

    def test_trg_view_length(
        self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset
    ):
        """Target views should have the original full length (20 tokens)."""
        dataset = T2VECPairDataset(src_dataset, trg_dataset)
        _, trg_view = dataset[0]
        assert len(trg_view) == 20

    def test_one_to_many_mapping(
        self, src_dataset: LocSeqDataset, trg_dataset: LocSeqDataset
    ):
        """Multiple src records should map to the same trg loc seq."""
        dataset = T2VECPairDataset(src_dataset, trg_dataset)

        # First 3 src entries should all map to trg[0]
        for idx in range(3):
            _, trg_view = dataset[idx]
            expected_view = trg_dataset.get_view(0)
            assert trg_view.seq == expected_view.seq
