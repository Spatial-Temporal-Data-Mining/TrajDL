# Copyright 2026 All authors of TrajDL
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

"""High-performance Arrow→numpy/torch conversion primitives (framework-level
reuse, product of the 2026-09-23 zero-copy audit).

Contract: the hot path forbids ``as_py()`` /
``to_pylist()`` / ``to_pandas()``; primitive columns default to
``zero_copy_only=True``, and null / multi-chunk / offset≠0 scenarios degrade
to an **explicit** copy (never silent). Variable-length columns are read
zero-copy through the flat values/offsets buffers plus a vectorized padded
write, replacing per-row ``as_py`` + ``pad_sequence``.
"""

import numpy as np
import pyarrow as pa
import torch


def _single_chunk(col: "pa.ChunkedArray | pa.Array") -> pa.Array:
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 0:
            return pa.array([], type=col.type)
        col = col.combine_chunks() if col.num_chunks > 1 else col.chunk(0)
    return col


def _numpy_zero_copy_fallback(arr: pa.Array) -> np.ndarray:
    try:
        return arr.to_numpy(zero_copy_only=True)
    except pa.ArrowInvalid:
        # 显式降级：offset≠0（切片）或 null 存在时零拷贝不可达
        return arr.to_numpy(zero_copy_only=False)


def primitive_to_numpy(col: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """Primitive column → numpy: zero-copy by default, explicit copy degradation when zero-copy is impossible."""
    return _numpy_zero_copy_fallback(_single_chunk(col))


def primitive_to_tensor(col: "pa.ChunkedArray | pa.Array", dtype: torch.dtype | None = None) -> torch.Tensor:
    """Primitive column → torch tensor via the zero-copy path (the numpy view shares the Arrow memory)."""
    t = torch.from_numpy(np.ascontiguousarray(primitive_to_numpy(col)))
    return t if dtype is None else t.to(dtype)


def flat_values_offsets(col: "pa.ChunkedArray | pa.Array") -> tuple[np.ndarray, np.ndarray]:
    """Variable-length primitive list column → two numpy views ``(values, offsets)`` (zero-copy read).

    For a sliced array (e.g. the concat product of 1-row slices produced by
    the DataLoader ``__getitems__``), ``values`` is the full child buffer
    while ``offsets`` keeps absolute base addresses, so the two no longer
    cover the same span; we uniformly go through ``flatten()`` (keeping only
    the referenced span) plus base-address rebasing to restore the
    ``len(values) == offsets[-1]`` invariant (the zero-copy fast path for
    non-sliced arrays is unaffected).
    """
    arr = _single_chunk(col)
    values = _numpy_zero_copy_fallback(arr.flatten())
    offsets = _numpy_zero_copy_fallback(arr.offsets).astype(np.int64, copy=False)
    if len(offsets) == 0:
        offsets = np.zeros(1, dtype=np.int64)
    if len(values) != int(offsets[-1] - offsets[0]):
        raise ValueError(
            f"varlen list column inconsistent: len(values)={len(values)} "
            f"but offsets cover [{offsets[0]}, {offsets[-1]})"
        )
    if offsets[0] != 0:
        offsets = offsets - offsets[0]
    return values, offsets


def varlen_to_padded(
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    pad_value: int | float = 0,
    min_len: int = 0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Flat values + offsets → a single padded matrix (vectorized scatter write, zero Python loops)."""
    offsets = offsets.astype(np.int64, copy=False)
    lens = np.diff(offsets)
    n = len(lens)
    max_len = max(int(lens.max(initial=0)), int(min_len))
    out = np.full((n, max_len), pad_value, dtype=values.dtype if dtype is None else dtype)
    if n and max_len and len(values):
        rows = np.repeat(np.arange(n), lens)
        pos = np.arange(int(offsets[-1])) - np.repeat(offsets[:-1], lens)
        out[rows, pos] = values
    return out


def matrix_to_arrow_column(arr: np.ndarray) -> "pa.FixedSizeListArray":
    """(N, W) contiguous matrix → ``fixed_size_list<W>`` column (underlying buffer shared zero-copy).

    A fixed-width matrix column is the zero-copy bidirectional form in which a
    (N, W) numpy matrix is expressed in Arrow IPC on disk: the values buffer is
    exactly the row-major flattening of the matrix.
    """
    arr = np.ascontiguousarray(arr)
    flat = pa.array(arr.reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, arr.shape[1])


def arrow_column_to_matrix(col: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """``fixed_size_list<W>`` column → (N, W) numpy zero-copy view.

    With multiple chunks / offset≠0 / nulls present, degrade explicitly to a
    copy (same rule as primitive_to_numpy).
    """
    arr = _single_chunk(col)
    values = arr.flatten() if isinstance(arr, pa.FixedSizeListArray) else arr
    flat = _numpy_zero_copy_fallback(values)
    width = arr.type.list_size if isinstance(arr, pa.FixedSizeListArray) else 1
    return flat.reshape(len(arr), width)


def list_child(col: "pa.ChunkedArray | pa.Array") -> pa.Array:
    """list / large_list column → child values Array (zero-copy view when offset=0 and no nulls)."""
    return _single_chunk(col).flatten()


def list_offsets(col: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """list / large_list column → offsets as numpy (int64 zero-copy for large_list)."""
    arr = _single_chunk(col)
    return _numpy_zero_copy_fallback(arr.offsets)
