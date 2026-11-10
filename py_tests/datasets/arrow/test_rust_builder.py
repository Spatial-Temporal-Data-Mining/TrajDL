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

"""Tests for the Rust-side ``RecordBatch`` builders and the unified
``init_from_seqs`` loop.

Coverage:

* Schema source-of-truth: schemas exposed by Rust agree with the historical
  Python-side definitions (which now re-export the Rust ones).
* Equivalence: ``use_rust=True`` and ``use_rust=False`` paths produce
  byte-equal ``core_table``\s, both for ``list`` and ``generator`` inputs.
* Memory lifetime: dropping the Python-side input list and forcing GC must
  not invalidate the returned ``RecordBatch`` (validates Arc ref-counting
  across the FFI boundary).
* Type checking: a wrong-type item must fail the unified loop with a clear
  ``ValueError`` regardless of which path is used.
* LocSeq coordinate semantics: ``coord_system`` is always forced to
  ``None`` for ``LocSeqDataset`` even if the caller passes one.
* Edge cases: empty input, missing optional metadata, embedded nulls,
  cross-batch concatenation.
"""

from __future__ import annotations

import gc
import random
from typing import Iterable, List

import pyarrow as pa
import pytest
from trajdl import CoordinateSystem, LocSeq, Trajectory
from trajdl.datasets import LocSeqDataset, TrajectoryDataset
from trajdl.datasets.arrow.builder import (
    build_loc_seq_record_batch,
    build_traj_record_batch,
)
from trajdl.datasets.arrow.schema import (
    LOC_SEQ_ARROW_SCHEMA,
    TRAJ_ARROW_SCHEMA,
)


def _make_traj(
    coords,
    *,
    entity_id=None,
    start_ts=None,
    ts_seq=None,
    coord_system=CoordinateSystem.Wgs84,
):
    traj = Trajectory.from_list(coords, coord_system, entity_id=entity_id)
    if start_ts is not None:
        traj.start_ts = start_ts
    if ts_seq is not None:
        traj.ts_seq = ts_seq
    return traj


def _make_loc_seq(locs, *, entity_id=None, start_ts=None, ts_seq=None):
    seq = LocSeq.from_list(locs, entity_id=entity_id, ts_seq=ts_seq)
    if start_ts is not None:
        seq.start_ts = start_ts
    return seq


def _as_input(seqs: list, *, kind: str) -> Iterable:
    """Wrap a list as either a list or a generator, depending on ``kind``.

    ``init_from_seqs`` must treat both identically — the unified loop
    consumes any ``Iterable`` exactly once.
    """
    if kind == "list":
        return seqs
    if kind == "generator":
        return (s for s in seqs)
    raise AssertionError(f"unknown input kind: {kind}")


# ---------------------------------------------------------------------------
# Schema source-of-truth: Rust schemas are re-exported via Python module
# ---------------------------------------------------------------------------


def test_loc_seq_arrow_schema_is_well_formed():
    assert LOC_SEQ_ARROW_SCHEMA.names == ["seq", "entity_id", "start_ts", "ts_seq"]


def test_traj_arrow_schema_is_well_formed():
    assert TRAJ_ARROW_SCHEMA.names == ["seq", "entity_id", "start_ts", "ts_seq"]


def test_builder_is_importable_from_public_module():
    """Users should never need to touch ``trajdl_rust`` directly."""
    assert callable(build_traj_record_batch)
    assert callable(build_loc_seq_record_batch)


# ---------------------------------------------------------------------------
# Equivalence: use_rust=True ↔ use_rust=False, list ↔ generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_kind", ["list", "generator"])
@pytest.mark.parametrize("use_rust", [True, False])
def test_traj_init_from_seqs_basic(use_rust: bool, input_kind: str):
    trajs = [
        _make_traj(
            [[1.0, 2.0], [3.0, 4.0]],
            entity_id="user_a",
            start_ts=100,
            ts_seq=[100, 200],
        ),
        _make_traj([[5.0, 6.0]], entity_id="user_b"),
        _make_traj(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ts_seq=[1, 2, 3],
        ),
    ]
    reference = TrajectoryDataset.init_from_seqs(trajs, use_rust=False)
    actual = TrajectoryDataset.init_from_seqs(
        _as_input(trajs, kind=input_kind), use_rust=use_rust
    )
    assert actual.core_table.equals(reference.core_table)
    assert actual.coord_system == reference.coord_system == CoordinateSystem.Wgs84


@pytest.mark.parametrize("input_kind", ["list", "generator"])
@pytest.mark.parametrize("use_rust", [True, False])
def test_loc_seq_init_from_seqs_with_nulls_and_empty_rows(
    use_rust: bool, input_kind: str
):
    seqs = [
        _make_loc_seq(
            ["a", "b", None, "d"],
            entity_id="u1",
            ts_seq=[1, 2, 3, 4],
        ),
        _make_loc_seq(["x"], entity_id="u2"),
        _make_loc_seq([], entity_id="u3"),  # empty seq is still a valid row
    ]
    reference = LocSeqDataset.init_from_seqs(seqs, use_rust=False)
    actual = LocSeqDataset.init_from_seqs(
        _as_input(seqs, kind=input_kind), use_rust=use_rust
    )
    assert actual.core_table.equals(reference.core_table)
    # LocSeq has no coordinates → coord_system must always be None.
    assert actual.coord_system is None


@pytest.mark.parametrize("input_kind", ["list", "generator"])
@pytest.mark.parametrize("use_rust", [True, False])
def test_traj_init_from_seqs_large_random(use_rust: bool, input_kind: str):
    """Random stress: 5k trajectories with varied lengths and metadata.

    Mirrors the ``samples.py`` LR-generation workload at smaller scale.
    """
    rng = random.Random(20260506)
    trajs: List[Trajectory] = []
    for i in range(5000):
        n_points = rng.randint(1, 30)
        coords = [
            [rng.uniform(-180, 180), rng.uniform(-90, 90)] for _ in range(n_points)
        ]
        entity = f"u_{i}" if rng.random() < 0.7 else None
        ts_seq = (
            sorted(rng.randint(0, 1_000_000) for _ in range(n_points))
            if rng.random() < 0.5
            else None
        )
        trajs.append(_make_traj(coords, entity_id=entity, ts_seq=ts_seq))

    reference = TrajectoryDataset.init_from_seqs(trajs, use_rust=False)
    actual = TrajectoryDataset.init_from_seqs(
        _as_input(trajs, kind=input_kind), use_rust=use_rust
    )
    assert actual.core_table.equals(reference.core_table)


@pytest.mark.parametrize("use_rust", [True, False])
def test_chunk_size_boundary_alignment_against_default(use_rust: bool):
    """The Rust path produces one ``RecordBatch`` per ``chunk_size`` rows
    just like the Python path, including the "exact multiple" boundary
    where no trailing partial batch should appear."""
    trajs = [_make_traj([[float(i), float(i + 1)]]) for i in range(64)]
    chunk_size = 16
    ds = TrajectoryDataset.init_from_seqs(
        trajs, chunk_size=chunk_size, use_rust=use_rust
    )
    seq_col = ds.core_table.column("seq")
    # 64 rows / 16 per chunk = 4 batches, no trailing remainder.
    assert seq_col.num_chunks == 4


@pytest.mark.parametrize("use_rust", [True, False])
def test_loc_seq_dataset_has_no_coord_system(use_rust: bool):
    """LocSeq has no coordinates of its own → dataset coord_system is None."""
    seqs = [_make_loc_seq(["a"], entity_id="u")]
    ds = LocSeqDataset.init_from_seqs(seqs, use_rust=use_rust)
    assert ds.coord_system is None


# ---------------------------------------------------------------------------
# Type checking inside the unified loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Type checking inside the unified loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_rust", [True, False])
def test_traj_init_rejects_non_trajectory_item(use_rust: bool):
    seqs = [_make_traj([[1.0, 2.0]]), "not a trajectory"]
    with pytest.raises(ValueError, match="expected Trajectory"):
        TrajectoryDataset.init_from_seqs(seqs, use_rust=use_rust)


@pytest.mark.parametrize("use_rust", [True, False])
def test_loc_seq_init_rejects_trajectory_item(use_rust: bool):
    """Passing a Trajectory into LocSeqDataset must fail loudly, not
    silently produce a malformed table."""
    seqs = [_make_loc_seq(["a"]), _make_traj([[1.0, 2.0]])]
    with pytest.raises(ValueError, match="expected LocSeq"):
        LocSeqDataset.init_from_seqs(seqs, use_rust=use_rust)


@pytest.mark.parametrize("use_rust", [True, False])
def test_traj_init_rejects_mixed_coord_systems(use_rust: bool):
    trajs = [
        _make_traj([[1.0, 2.0]], coord_system=CoordinateSystem.Wgs84),
        _make_traj([[3.0, 4.0]], coord_system=CoordinateSystem.WebMercator),
    ]
    with pytest.raises(ValueError, match="only 1 coordinate system"):
        TrajectoryDataset.init_from_seqs(trajs, use_rust=use_rust)


@pytest.mark.parametrize("use_rust", [True, False])
def test_traj_init_empty_input_raises(use_rust: bool):
    """Empty input cannot infer a coord_system → consistent error on
    both paths."""
    with pytest.raises(ValueError, match="only 1 coordinate system"):
        TrajectoryDataset.init_from_seqs([], use_rust=use_rust)


# ---------------------------------------------------------------------------
# coord_system inference (single source of truth: the inputs themselves)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_rust", [True, False])
def test_traj_init_infers_non_default_coord_system(use_rust: bool):
    """Inference picks up whatever coord_system the inputs carry — there
    is no separate ``coord_system`` parameter on ``init_from_seqs``;
    the inputs are the single source of truth."""
    trajs = [
        _make_traj([[1.0, 2.0]], coord_system=CoordinateSystem.WebMercator),
        _make_traj([[3.0, 4.0]], coord_system=CoordinateSystem.WebMercator),
    ]
    ds = TrajectoryDataset.init_from_seqs(trajs, use_rust=use_rust)
    assert ds.coord_system == CoordinateSystem.WebMercator


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        TrajectoryDataset.init_from_seqs([_make_traj([[1.0, 2.0]])], chunk_size=0)


# ---------------------------------------------------------------------------
# Memory lifetime: validate Arc ref-counting across FFI
# ---------------------------------------------------------------------------


def test_record_batch_outlives_python_input_list():
    """After dropping the Python-side input and GC'ing, the RecordBatch
    must still decode correctly, proving the Rust-allocated Arrow buffers
    are kept alive by Python's refcount on the returned object (Arc on
    the Rust side, release callback in the C Data Interface)."""
    trajs = [
        _make_traj(
            [[float(i), float(i + 1)] for i in range(20)],
            entity_id=f"user_{j}",
            start_ts=j,
            ts_seq=list(range(20)),
        )
        for j in range(100)
    ]

    batch = build_traj_record_batch(trajs)

    # Drop everything we can on the Python side.
    del trajs
    gc.collect()

    # If the buffers were freed prematurely we'd see a use-after-free here.
    assert batch.num_rows == 100
    seq_col = batch.column(0)
    first = seq_col[0].as_py()
    assert len(first) == 20
    assert first[0] == [0.0, 1.0]
    assert first[-1] == [19.0, 20.0]

    entity_col = batch.column(1)
    assert entity_col[0].as_py() == "user_0"
    assert entity_col[99].as_py() == "user_99"


def test_record_batch_can_be_concatenated_into_table():
    """Multiple Rust-built RecordBatches must concat into a valid
    ``pa.Table`` without copy-back."""
    b1 = build_traj_record_batch([_make_traj([[1.0, 2.0]], entity_id="a")])
    b2 = build_traj_record_batch([_make_traj([[3.0, 4.0], [5.0, 6.0]], entity_id="b")])

    table = pa.Table.from_batches([b1, b2])
    assert table.num_rows == 2
    assert table.schema.equals(TRAJ_ARROW_SCHEMA)
    assert table.column("entity_id").to_pylist() == ["a", "b"]


# ---------------------------------------------------------------------------
# Edge cases on the public builder API
# ---------------------------------------------------------------------------


def test_traj_builder_empty_list_returns_empty_batch():
    batch = build_traj_record_batch([])
    assert batch.num_rows == 0
    assert batch.schema.equals(TRAJ_ARROW_SCHEMA)


def test_loc_seq_builder_empty_list_returns_empty_batch():
    batch = build_loc_seq_record_batch([])
    assert batch.num_rows == 0
    assert batch.schema.equals(LOC_SEQ_ARROW_SCHEMA)


def test_traj_builder_handles_all_optional_metadata_missing():
    trajs = [
        _make_traj([[1.0, 2.0], [3.0, 4.0]]),  # no entity_id, no ts_seq
    ]
    batch = build_traj_record_batch(trajs)
    assert batch.column("entity_id")[0].as_py() is None
    assert batch.column("start_ts")[0].as_py() is None
    assert batch.column("ts_seq")[0].as_py() is None
    assert batch.column("seq")[0].as_py() == [[1.0, 2.0], [3.0, 4.0]]


def test_loc_seq_builder_preserves_inner_nulls():
    seqs = [_make_loc_seq([None, "a", None, "b"], entity_id="u")]
    batch = build_loc_seq_record_batch(seqs)
    assert batch.column("seq")[0].as_py() == [None, "a", None, "b"]
