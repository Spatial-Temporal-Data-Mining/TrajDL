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

"""convert.py 零拷贝原语的等价性与降级语义测试。"""

import numpy as np
import pyarrow as pa
import torch
from trajdl.datasets.arrow.convert import (
    arrow_column_to_matrix,
    flat_values_offsets,
    matrix_to_arrow_column,
    primitive_to_tensor,
    varlen_to_padded,
)


def test_primitive_zero_copy():
    col = pa.chunked_array([pa.array([1, 2, 3], type=pa.int64())])
    t = primitive_to_tensor(col)
    assert t.dtype == torch.int64
    assert t.tolist() == [1, 2, 3]


def test_primitive_slice_explicit_degrade():
    # offset≠0 时零拷贝不可达 → 显式拷贝降级，结果仍正确
    sliced = pa.array([10, 20, 30, 40]).slice(1, 3)
    assert primitive_to_tensor(sliced).tolist() == [20, 30, 40]


def test_multi_chunk_combine():
    col = pa.chunked_array([pa.array([1]), pa.array([2, 3])])
    assert primitive_to_tensor(col).tolist() == [1, 2, 3]


def test_varlen_padded_matches_as_py_reference():
    lst = pa.array([[1, 2], [3], [], [4, 5, 6]], type=pa.list_(pa.int64()))
    values, offsets = flat_values_offsets(pa.chunked_array([lst]))
    out = varlen_to_padded(values, offsets, pad_value=9)
    ref = np.full((4, 3), 9, dtype=np.int64)
    for i, row in enumerate(lst.to_pylist()):  # 参考实现故意用 as_py 语义
        ref[i, : len(row)] = row
    assert np.array_equal(out, ref)
    assert out.shape == (4, 3)


def test_varlen_padded_empty_and_min_len():
    values, offsets = flat_values_offsets(pa.chunked_array([pa.array([], type=pa.list_(pa.int64()))]))
    out = varlen_to_padded(values, offsets, pad_value=0, min_len=5)
    assert out.shape == (0, 5)


def test_flat_values_offsets_sliced_table():
    # 回归：DataLoader __getitems__ 对大表 slice 后 .values 与 .offsets 覆盖不一致
    # （云上 tuler 实证：values 40630 vs offsets[-1] 38425）→ flatten + 基址重定
    for list_type in (pa.list_(pa.int64()), pa.large_list(pa.int64())):
        lst = pa.array([[1, 2], [3], [], [4, 5, 6], [7]], type=list_type)
        table = pa.table({"col": lst})
        sliced = table.slice(1, 3)  # offset=1：values 是完整 child、offsets 带绝对基址
        values, offsets = flat_values_offsets(sliced["col"])
        assert offsets[0] == 0
        assert len(values) == int(offsets[-1])
        rows = [values[offsets[i] : offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
        assert rows == [[3], [], [4, 5, 6]]
        # 与未切片子集的 flat 读逐位一致
        v_ref, o_ref = flat_values_offsets(
            pa.chunked_array([pa.array([[3], [], [4, 5, 6]], type=list_type)])
        )
        assert np.array_equal(values, v_ref) and np.array_equal(offsets, o_ref)


def test_flat_values_offsets_null_slot_empty_range():
    # null 槽位（pyarrow 构造路径下 range 恒为空）保持一致读出；
    # flat_values_offsets 对覆盖不一致的病态数组显式报错（不变量防线）
    arr = pa.ListArray.from_arrays(
        pa.array([0, 2, None, 5], type=pa.int32()), pa.array([1, 2, 3, 4, 5])
    )
    values, offsets = flat_values_offsets(arr)
    assert values.tolist() == [1, 2, 3, 4, 5]
    assert len(values) == int(offsets[-1] - offsets[0])
    assert values[offsets[0] : offsets[1]].tolist() == [1, 2]


def test_matrix_column_roundtrip_zero_copy():
    m = np.arange(24, dtype=np.float32).reshape(3, 8)
    col = matrix_to_arrow_column(m)
    assert col.type == pa.list_(pa.float32(), 8)
    # 零拷贝：Arrow values 缓冲区与源矩阵共享内存
    back = arrow_column_to_matrix(pa.chunked_array([col]))
    assert np.array_equal(back, m)
    assert back.dtype == np.float32
    assert np.shares_memory(back, m)


def test_matrix_column_ipc_roundtrip(tmp_path):
    # 走 IPC 落盘/装载（memory-map）后数值逐位一致
    m = np.random.RandomState(0).randn(7, 6).astype(np.float64)
    col = matrix_to_arrow_column(m)
    schema = pa.schema([("points", col.type), ("offsets", pa.int64())])
    batch = pa.record_batch(
        [col, pa.array(np.arange(7, dtype=np.int64))], schema=schema
    )
    p = tmp_path / "t.arrow"
    with pa.OSFile(str(p), "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            writer.write_batch(batch)
    with pa.memory_map(str(p), "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    assert np.array_equal(arrow_column_to_matrix(table["points"]), m)
    assert primitive_to_tensor(table["offsets"]).tolist() == list(range(7))


def test_matrix_column_empty():
    col = matrix_to_arrow_column(np.zeros((0, 8), dtype=np.float32))
    back = arrow_column_to_matrix(col)
    assert back.shape == (0, 8)
