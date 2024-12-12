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

import os
import random
import re
from datetime import datetime
from typing import List

import numpy as np
import polars as pl
import pytest

from trajdl.common.enum import ArrowColName
from trajdl.datasets.arrow import BaseArrowDataset
from trajdl.datasets.arrow.locseq import LocSeqDataset
from trajdl.datasets.arrow.traj import TrajectoryDataset
from trajdl.datasets.base import BaseSeq
from trajdl.datasets.base.locseq import LocSeq
from trajdl.datasets.base.traj import Trajectory

from ...conftest import FOLDER


def construct_ds() -> List[LocSeqDataset]:
    test_ds = []
    for add_entity_id in [True, False]:
        for add_ts in [True, False]:
            seq_length = random.randint(0, 10)
            now_ts = int(datetime.now().timestamp())
            loc_seqs = [
                LocSeq(
                    [str(loc_idx) for loc_idx in range(seq_length)],
                    entity_id=str(i) if add_entity_id else None,
                    ts_seq=(
                        [now_ts + ts_idx for ts_idx in range(seq_length)]
                        if add_ts
                        else None
                    ),
                )
                for i in range(10)
            ]

            test_ds.append(LocSeqDataset.init_from_loc_seqs(seqs=loc_seqs))
    return test_ds


def check_col_names(ds: BaseArrowDataset):
    column_names = ds.table.schema.names
    for col in ArrowColName:
        assert col.value in column_names


def test_arrow_datasets(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    test_ds = construct_ds()
    test_ds.append(test_locseq_dataset)
    test_ds.append(test_trajectory_dataset)

    for ds in [test_locseq_dataset, test_trajectory_dataset]:
        tmp_path = os.path.join(FOLDER, "test_ds")
        ds.save(tmp_path)
        assert os.path.isfile(tmp_path + ".parquet")
        test_ds.append(ds.__class__.init_from_parquet(tmp_path + ".parquet"))

    for ds in test_ds:
        check_col_names(ds)

        indices = [[0, 1], [1, 2]]
        indices2 = [np.array(i, dtype=np.int64) for i in indices]
        indices3 = [0, 1, 2]
        indices4 = [np.int64(0), np.int32(1), 2]

        assert len(ds[0]) == 1
        assert len(ds.__getitems__([np.array([0, 1], dtype=np.int64)])) == 2
        assert len(ds.__getitems__(indices)) == 4
        assert len(ds.__getitems__(indices2)) == 4
        assert len(ds.__getitems__(indices3)) == 3
        assert len(ds.__getitems__(indices4)) == 3

        for new_ds in [
            ds[0],
            ds.__getitems__(indices),
            ds.__getitems__(indices2),
            ds.__getitems__(indices3),
            ds.__getitems__(indices4),
        ]:
            check_col_names(ds=new_ds)

        for t in ds.iter_as_seqs():
            if isinstance(ds, LocSeqDataset):
                assert isinstance(t, LocSeq)
            elif isinstance(ds, TrajectoryDataset):
                assert isinstance(t, Trajectory)
            else:
                raise RuntimeError("Test cases do not cover.")


def test_init_from_different_types(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    for ds in [test_locseq_dataset, test_trajectory_dataset]:
        df = ds.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert df.height == len(ds)

        for col_enum in ArrowColName:
            arrow_table = df.drop(col_enum.value).to_arrow()
            with pytest.warns(
                RuntimeWarning, match=r"Field .* does not exist in the input table."
            ):
                tmp_ds = ds.__class__.init_from_arrow(arrow_table)
                tmp_table_col = tmp_ds.table.column(col_enum.value)
                for t in tmp_table_col:
                    assert t.as_py() is None

        ds.__class__.init_from_table(df)
        ds.__class__.init_from_table(df.to_pandas())
        ds.__class__.init_from_table(df.to_arrow())

        with pytest.raises(
            ValueError,
            match=re.escape(
                "`table` should be an instance of {'pyarrow.Table', 'polars.DataFrame', 'pandas.DataFrame'}"
            ),
        ):
            ds.__class__.init_from_table(df.to_dicts())


def test_arrow_datasets_properties(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    def check_func(
        ds: BaseArrowDataset,
        seqs: List[BaseSeq],
        attr_name: str,
        idx: int,
        all_close: bool,
    ) -> bool:
        property_from_arrow = getattr(ds, attr_name)[idx].as_py()
        property_from_py = getattr(seq, attr_name)

        if property_from_py is None:
            return property_from_arrow is None

        if isinstance(property_from_py, np.ndarray) and len(property_from_py) == 0:
            return not property_from_arrow

        if all_close:
            return np.allclose(property_from_arrow, property_from_py)
        else:
            return property_from_arrow == property_from_py

    for ds in [test_locseq_dataset, test_trajectory_dataset]:
        seqs = list(ds.iter_as_seqs())

        for idx, seq in enumerate(seqs):
            if isinstance(ds, TrajectoryDataset):
                assert check_func(ds, seqs, "seq", idx, all_close=True)
            elif isinstance(ds, LocSeqDataset):
                assert check_func(ds, seqs, "seq", idx, all_close=False)
            else:
                raise RuntimeError("Test cases do not cover.")

            assert check_func(ds, seqs, "entity_id", idx, all_close=False)
            assert check_func(ds, seqs, "ts_seq", idx, all_close=True)
            assert check_func(ds, seqs, "ts_delta", idx, all_close=True)
            assert check_func(ds, seqs, "dis_delta", idx, all_close=True)
            assert check_func(ds, seqs, "start_ts", idx, all_close=False)
