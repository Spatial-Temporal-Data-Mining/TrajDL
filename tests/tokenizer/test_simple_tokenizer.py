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

import pyarrow as pa
import pytest

from trajdl.common.enum import TokenEnum
from trajdl.tokenizers import SimpleTokenizer

from ..conftest import FOLDER
from . import load_tokenizer_test_func


def test_simple_tokenizer():
    with pytest.raises(ValueError):
        vocab = {"a": 1, "b": 2}
        SimpleTokenizer.build(init_vocab=vocab)

    with pytest.raises(ValueError):
        vocab = {"a": 0, "b": 0}
        SimpleTokenizer.build(init_vocab=vocab)

    with pytest.warns(
        RuntimeWarning,
        match=r"Token '.*' exist in vocab, tokenizer will not give this token a index automatically.",
    ):
        vocab = {"a": 0, "b": 1, "[BOS]": 2}
        SimpleTokenizer.build(init_vocab=vocab)

    string_array = pa.array(["a", "b", "a", "c"], type=pa.string())
    list_scalar_str = pa.scalar(string_array, type=pa.list_(pa.string()))
    seq = [
        TokenEnum.BOS_TOKEN.value,
        "a",
        "b",
        TokenEnum.EOS_TOKEN.value,
        TokenEnum.PAD_TOKEN.value,
    ]

    vocab = {"a": 0, "b": 1}
    tokenizer = SimpleTokenizer.build(init_vocab=vocab)
    path = os.path.join(FOLDER, "test_simple_tokenizer.pkl")
    tokenizer.save_pretrained(path)

    def test_tokenizer(tokenizer: SimpleTokenizer):
        assert tokenizer.tokenize_loc_seq(seq) == [
            tokenizer.bos,
            0,
            1,
            tokenizer.eos,
            tokenizer.pad,
        ]

        assert tokenizer.loc2idx("c") == tokenizer.unk

        assert tokenizer.tokenize_loc_seq(loc_seq=list_scalar_str) == [
            0,
            1,
            0,
            tokenizer.unk,
        ]

        assert (
            len(
                {
                    tokenizer.bos,
                    tokenizer.eos,
                    tokenizer.unk,
                    tokenizer.pad,
                    tokenizer.mask,
                }
            )
            == 5
        )

    tokenizers = [tokenizer, SimpleTokenizer.load_pretrained(path)]
    for tokenizer in tokenizers:
        test_tokenizer(tokenizer=tokenizer)

    load_tokenizer_test_func(tokenizer=tokenizer, path=path)
