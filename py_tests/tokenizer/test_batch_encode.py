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

"""``AbstractTokenizer.encode_locseq_batch`` 等价性测试。

批量编码输出必须与逐行 ``encode_locseq(..., return_as="np")`` 逐位一致
（含 OOV / 空行 / BOS+EOS 组合 / 多 chunk 列 / ListScalar / 空列）。
"""

import numpy as np
import pyarrow as pa
import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from trajdl.datasets.arrow.convert import varlen_to_padded
from trajdl.tokenizers import SimpleTokenizer

SEQS = [
    ["a", "b", "a"],
    ["c", "a"],
    [],
    ["b", "b", "c", "a", "b"],
    ["d"],
]


@pytest.fixture
def tokenizer():
    return SimpleTokenizer.from_vocab(init_vocab={"a": 0, "b": 1})


def _reference(tokenizer, seqs, add_bos, add_eos):
    rows = [
        tokenizer.encode_locseq(s, add_bos=add_bos, add_eos=add_eos, return_as="np")
        for s in seqs
    ]
    flat = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(r) for r in rows])
    return flat, offsets


@pytest.mark.parametrize("add_bos", [False, True])
@pytest.mark.parametrize("add_eos", [False, True])
def test_batch_matches_per_row_arrow_column(tokenizer, add_bos, add_eos):
    col = pa.chunked_array(
        [
            pa.array(SEQS[:2], type=pa.list_(pa.string())),
            pa.array(SEQS[2:], type=pa.list_(pa.string())),
        ]
    )
    flat, offsets = tokenizer.encode_locseq_batch(col, add_bos=add_bos, add_eos=add_eos)
    ref_flat, ref_offsets = _reference(tokenizer, SEQS, add_bos, add_eos)
    np.testing.assert_array_equal(flat, ref_flat)
    np.testing.assert_array_equal(offsets, ref_offsets)


@pytest.mark.parametrize("add_bos", [False, True])
@pytest.mark.parametrize("add_eos", [False, True])
def test_batch_matches_per_row_list_input(tokenizer, add_bos, add_eos):
    flat, offsets = tokenizer.encode_locseq_batch(
        [list(s) for s in SEQS], add_bos=add_bos, add_eos=add_eos
    )
    ref_flat, ref_offsets = _reference(tokenizer, SEQS, add_bos, add_eos)
    np.testing.assert_array_equal(flat, ref_flat)
    np.testing.assert_array_equal(offsets, ref_offsets)


@pytest.mark.parametrize("add_bos", [False, True])
@pytest.mark.parametrize("add_eos", [False, True])
def test_batch_matches_per_row_list_scalar_input(tokenizer, add_bos, add_eos):
    scalars = [
        pa.scalar(s, type=pa.list_(pa.string())) for s in SEQS
    ]
    flat, offsets = tokenizer.encode_locseq_batch(
        scalars, add_bos=add_bos, add_eos=add_eos
    )
    ref_flat, ref_offsets = _reference(tokenizer, SEQS, add_bos, add_eos)
    np.testing.assert_array_equal(flat, ref_flat)
    np.testing.assert_array_equal(offsets, ref_offsets)


@pytest.mark.parametrize("add_bos", [False, True])
@pytest.mark.parametrize("add_eos", [False, True])
def test_batch_matches_per_row_sliced_table(tokenizer, add_bos, add_eos):
    # 回归：DataLoader batch 是大表 slice 后的产物（.values 全 child /
    # .offsets 绝对基址），批量编码必须与未切片子集逐位一致
    table = pa.table({"loc": pa.array(SEQS, type=pa.list_(pa.string()))})
    for start, length in ((0, len(SEQS)), (1, 3), (2, 2)):
        col = table.slice(start, length)["loc"]
        flat, offsets = tokenizer.encode_locseq_batch(col, add_bos=add_bos, add_eos=add_eos)
        ref_flat, ref_offsets = _reference(tokenizer, SEQS[start : start + length], add_bos, add_eos)
        np.testing.assert_array_equal(flat, ref_flat)
        np.testing.assert_array_equal(offsets, ref_offsets)


def test_empty_column(tokenizer):
    col = pa.chunked_array([pa.array([], type=pa.list_(pa.string()))])
    for kwargs in ({}, {"add_bos": True}, {"add_eos": True}, {"add_bos": True, "add_eos": True}):
        flat, offsets = tokenizer.encode_locseq_batch(col, **kwargs)
        assert flat.shape == (0,) and flat.dtype == np.int64
        np.testing.assert_array_equal(offsets, [0])


def test_padded_matrix_matches_pad_sequence(tokenizer):
    col = pa.chunked_array([pa.array(SEQS, type=pa.list_(pa.string()))])
    flat, offsets = tokenizer.encode_locseq_batch(col)
    padded = torch.from_numpy(varlen_to_padded(flat, offsets, pad_value=tokenizer.pad))

    rows = [tokenizer.encode_locseq(s, return_as="pt") for s in SEQS]
    ref = pad_sequence(rows, batch_first=True, padding_value=tokenizer.pad)
    assert torch.equal(padded, ref)
