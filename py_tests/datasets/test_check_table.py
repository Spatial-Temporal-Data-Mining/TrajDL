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

"""Tests for the ``check_table`` validators of the Arrow datasets.

Covers the Arrow schema contract: the 4 core columns must
exist, ``LocSeqDataset`` additionally requires a ``list<string>``-shaped
``seq`` column, and ``TrajectoryDataset`` additionally requires the geometry
column (a list of ``(x, y)`` coordinate pairs). The validators are exercised
through the public constructor, which invokes ``check_table`` unless
``unsafe=True``.
"""

import pyarrow as pa
import pytest
from trajdl import CoordinateSystem
from trajdl.common.enum import ArrowColName
from trajdl.datasets.arrow.locseq import LocSeqDataset
from trajdl.datasets.arrow.schema import LOC_SEQ_ARROW_SCHEMA, TRAJ_ARROW_SCHEMA
from trajdl.datasets.arrow.traj import TrajectoryDataset


def _locseq_table(seq_array: pa.Array, drop_cols: tuple = ()) -> pa.Table:
    """Build a LocSeq-shaped table, optionally overriding seq / dropping columns."""
    arrays = {
        ArrowColName.SEQ.value: (
            seq_array
            if seq_array is not None
            else pa.array([["a", "b"], ["c"]], type=pa.large_list(pa.large_string()))
        ),
        ArrowColName.ENTITY_ID.value: pa.array(["e1", "e2"], type=pa.large_string()),
        ArrowColName.START_TS.value: pa.array([0, 1], type=pa.int64()),
        ArrowColName.TS_SEQ.value: pa.array(
            [[1, 2], [3]], type=pa.large_list(pa.int64())
        ),
    }
    for col in drop_cols:
        arrays.pop(col)
    return pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))


def _traj_table(seq_array: pa.Array | None = None, drop_cols: tuple = ()) -> pa.Table:
    """Build a Trajectory-shaped table, optionally overriding seq / dropping columns."""
    coord_type = pa.list_(pa.float64(), 2)
    arrays = {
        ArrowColName.SEQ.value: (
            seq_array
            if seq_array is not None
            else pa.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]],
                type=pa.large_list(coord_type),
            )
        ),
        ArrowColName.ENTITY_ID.value: pa.array(["e1", "e2"], type=pa.large_string()),
        ArrowColName.START_TS.value: pa.array([0, 1], type=pa.int64()),
        ArrowColName.TS_SEQ.value: pa.array(
            [[1, 2], [3]], type=pa.large_list(pa.int64())
        ),
    }
    for col in drop_cols:
        arrays.pop(col)
    return pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))


def test_locseq_check_table_accepts_compliant_table():
    """A table matching the canonical LocSeq schema must pass validation."""
    ds = LocSeqDataset(table=_locseq_table(None))
    assert len(ds) == 2


def test_locseq_check_table_missing_core_column():
    """A table missing the ts_seq column must be rejected, naming the column."""
    with pytest.raises(ValueError, match=ArrowColName.TS_SEQ.value):
        LocSeqDataset(table=_locseq_table(None, drop_cols=(ArrowColName.TS_SEQ.value,)))


def test_locseq_check_table_rejects_non_string_seq():
    """A ``list<int64>`` seq column must be rejected, naming the column."""
    with pytest.raises(ValueError, match=ArrowColName.SEQ.value):
        LocSeqDataset(
            table=_locseq_table(pa.array([[1, 2], [3]], type=pa.large_list(pa.int64())))
        )


def test_locseq_check_table_rejects_scalar_seq():
    """A non-list (scalar string) seq column must be rejected as well."""
    with pytest.raises(ValueError, match=ArrowColName.SEQ.value):
        LocSeqDataset(
            table=_locseq_table(pa.array(["e1", "e2"], type=pa.large_string()))
        )


def test_locseq_unsafe_skips_check_table():
    """``unsafe=True`` keeps being the documented escape hatch."""
    bad = _locseq_table(pa.array([[1, 2], [3]], type=pa.large_list(pa.int64())))
    assert LocSeqDataset(table=bad, unsafe=True).core_table is bad


def test_trajectory_check_table_accepts_compliant_table():
    """A table matching the canonical Trajectory schema must pass validation."""
    ds = TrajectoryDataset(table=_traj_table(), coord_system=CoordinateSystem.Wgs84)
    assert len(ds) == 2


def test_trajectory_check_table_missing_core_column():
    """A table missing the ts_seq column must be rejected, naming the column."""
    with pytest.raises(ValueError, match=ArrowColName.TS_SEQ.value):
        TrajectoryDataset(
            table=_traj_table(drop_cols=(ArrowColName.TS_SEQ.value,)),
            coord_system=CoordinateSystem.Wgs84,
        )


def test_trajectory_check_table_missing_geometry_column():
    """A seq column without the (x, y) geometry layout must be rejected."""
    # seq shaped as a string list (a LocSeq payload): no geometry column
    with pytest.raises(ValueError, match=ArrowColName.SEQ.value):
        TrajectoryDataset(
            table=_traj_table(
                pa.array([["a", "b"], ["c"]], type=pa.large_list(pa.large_string()))
            ),
            coord_system=CoordinateSystem.Wgs84,
        )


def test_check_table_matches_canonical_schemas():
    """Both validators must accept tables derived from their own schemas.

    The tables are built from LOC_SEQ_ARROW_SCHEMA / TRAJ_ARROW_SCHEMA
    themselves and fed through the public constructor (which runs
    ``check_table``), so the test goes red if the check predicates drift
    from the Rust-owned canonical schemas in either direction: a schema
    the validator no longer accepts, or a validator requiring columns or
    types the schema no longer provides. A minimal 1-row table is used
    (instead of ``schema.empty_table()``) so that any future row-count
    shortcut in the validators cannot mask such drift.
    """
    loc_seq_table = pa.Table.from_pylist(
        [
            {
                ArrowColName.SEQ.value: ["a", "b"],
                ArrowColName.ENTITY_ID.value: "e1",
                ArrowColName.START_TS.value: 0,
                ArrowColName.TS_SEQ.value: [1, 2],
            }
        ],
        schema=LOC_SEQ_ARROW_SCHEMA,
    )
    assert loc_seq_table.schema == LOC_SEQ_ARROW_SCHEMA
    ds = LocSeqDataset(table=loc_seq_table)
    assert len(ds) == 1

    traj_table = pa.Table.from_pylist(
        [
            {
                ArrowColName.SEQ.value: [[1.0, 2.0], [3.0, 4.0]],
                ArrowColName.ENTITY_ID.value: "e1",
                ArrowColName.START_TS.value: 0,
                ArrowColName.TS_SEQ.value: [1, 2],
            }
        ],
        schema=TRAJ_ARROW_SCHEMA,
    )
    assert traj_table.schema == TRAJ_ARROW_SCHEMA
    ds = TrajectoryDataset(table=traj_table, coord_system=CoordinateSystem.Wgs84)
    assert len(ds) == 1
