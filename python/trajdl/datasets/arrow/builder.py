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

"""Public Arrow ``RecordBatch`` builders for trajectory datasets.

Two layers live in this module:

* The bare functions :func:`build_traj_record_batch` /
  :func:`build_loc_seq_record_batch` are zero-cost re-exports of the
  Rust-side PyO3 functions. They consume a list of sequences and produce
  a :class:`pyarrow.RecordBatch` whose buffers are zero-copy shared
  through the Arrow C Data Interface (no data is copied; lifetime is
  managed by Arrow's release callback). User code should always import
  them from here rather than touching the ``trajdl_rust`` namespace.

* The :class:`RecordBatchBuilder` abstraction owns a per-chunk buffer
  and exposes ``append`` / ``build`` / ``clear`` / ``__len__``. Two
  implementations exist:

    - :class:`_RustRecordBatchBuilder` accumulates ``Trajectory`` /
      ``LocSeq`` references and flushes through the Rust functions
      above. Used by ``init_from_seqs(use_rust=True)``, the default.
    - :class:`_PythonRecordBatchBuilder` accumulates per-column Python
      lists and flushes via ``pa.RecordBatch.from_pydict``. Slower; kept
      as the equivalence-test reference and a portability fallback.

  Construction goes through :func:`make_record_batch_builder`, which
  picks the right implementation from ``schema`` + ``use_rust``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import pyarrow as pa

# Zero-overhead re-exports: the Rust binding is the implementation.
# Aliasing here lets every other module import the friendlier names
# without having to know about ``trajdl_rust``.
from ...trajdl_rust import build_loc_seq_record_batch_py as build_loc_seq_record_batch
from ...trajdl_rust import build_traj_record_batch_py as build_traj_record_batch
from .schema import GET_ARROW_FROM_RUST_SEQUENCE, TRAJ_ARROW_SCHEMA


class RecordBatchBuilder(ABC):
    """Per-chunk Arrow ``RecordBatch`` accumulator.

    A builder owns a mutable buffer that ``append`` grows one sequence
    at a time. ``build`` flushes the buffer into a single
    :class:`pyarrow.RecordBatch`; ``clear`` discards whatever is buffered
    without producing a batch. ``__len__`` reports the current row count
    so callers can decide when to flush (typically once per
    ``chunk_size`` rows).

    Implementations must guarantee that ``build`` against an empty buffer
    is a no-op-equivalent: it returns a zero-row batch with the canonical
    schema. This keeps the trailing-flush logic in callers trivial.
    """

    @abstractmethod
    def append(self, seq: Any) -> None:
        """Add one sequence object (``Trajectory`` or ``LocSeq``) to the buffer."""

    @abstractmethod
    def build(self) -> pa.RecordBatch:
        """Materialize whatever is currently buffered as a ``RecordBatch``."""

    @abstractmethod
    def clear(self) -> None:
        """Drop the buffered rows without producing a batch."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of rows currently buffered."""


class _RustRecordBatchBuilder(RecordBatchBuilder):
    """Buffer ``Trajectory`` / ``LocSeq`` refs and flush through Rust.

    The actual ``Vec -> RecordBatch`` conversion happens in
    ``src/datasets/builder.rs``; this Python wrapper just owns the
    intermediate Python list and dispatches to the right Rust function.
    """

    def __init__(self, schema: pa.Schema) -> None:
        # Pick the Rust function once at construction time so the hot
        # ``build`` path is a single attribute lookup.
        self._rust_build = (
            build_traj_record_batch
            if schema == TRAJ_ARROW_SCHEMA
            else build_loc_seq_record_batch
        )
        self._buf: list = []

    def append(self, seq: Any) -> None:
        self._buf.append(seq)

    def build(self) -> pa.RecordBatch:
        return self._rust_build(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


class _PythonRecordBatchBuilder(RecordBatchBuilder):
    """Per-column Python-list buffer flushed via ``pa.RecordBatch.from_pydict``."""

    def __init__(self, schema: pa.Schema) -> None:
        self._schema = schema
        self._cols: dict[str, list] = defaultdict(list)

    def append(self, seq: Any) -> None:
        # The transform map keys mirror the schema field names; using it
        # here keeps the Python path bug-for-bug compatible with how the
        # original ``init_from_seqs`` shaped its ``chunk_data`` dict.
        for col_name, transform_func in GET_ARROW_FROM_RUST_SEQUENCE.items():
            self._cols[col_name].append(transform_func(seq))

    def build(self) -> pa.RecordBatch:
        # Cast back to a plain dict because ``from_pydict`` types its
        # signature on ``Mapping[str, list]`` and a defaultdict shows up
        # as ``defaultdict[str, list]`` to mypy.
        return pa.RecordBatch.from_pydict(dict(self._cols), schema=self._schema)

    def clear(self) -> None:
        self._cols.clear()

    def __len__(self) -> int:
        # All column lists grow in lockstep, so any of them is fine.
        # Empty dict → 0 rows.
        if not self._cols:
            return 0
        return len(next(iter(self._cols.values())))


def make_record_batch_builder(
    schema: pa.Schema, *, use_rust: bool
) -> RecordBatchBuilder:
    """Pick the right :class:`RecordBatchBuilder` implementation.

    Parameters
    ----------
    schema
        Arrow schema of the dataset being constructed. Determines whether
        the Rust path dispatches to ``build_traj_record_batch`` or
        ``build_loc_seq_record_batch``.
    use_rust
        If ``True``, return the zero-copy Rust-backed builder (~10×
        faster on large inputs). If ``False``, return the Python-only
        builder used for equivalence testing and as a portability
        fallback.
    """
    if use_rust:
        return _RustRecordBatchBuilder(schema)
    return _PythonRecordBatchBuilder(schema)


__all__ = [
    "build_traj_record_batch",
    "build_loc_seq_record_batch",
    "RecordBatchBuilder",
    "make_record_batch_builder",
]
