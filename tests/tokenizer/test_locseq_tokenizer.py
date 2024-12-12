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
import re
from typing import List

import numpy as np
import pyarrow as pa
import pytest
import torch
from tqdm import tqdm

from trajdl.datasets import LocSeq
from trajdl.tokenizers.locseq import LocSeqTokenizer

from ..conftest import FOLDER
from . import load_tokenizer_test_func


def test_locseq_tokenizer(
    locseq_tokenizer: LocSeqTokenizer, test_locseqs: List[LocSeq]
):
    path = os.path.join(FOLDER, "test_locseq_tokenizer.pkl")
    locseq_tokenizer.save_pretrained(path)

    samples = test_locseqs[:5]
    string_array = pa.array(samples[0].seq, type=pa.string())
    pa_sample = pa.scalar(string_array, type=pa.list_(pa.string()))

    def test_tokenizer(tokenizer: LocSeqTokenizer):

        assert isinstance(tokenizer, LocSeqTokenizer)

        assert len(tokenizer) == 5672
        assert tokenizer.bos is not None
        assert tokenizer.eos is not None
        assert tokenizer.unk is not None
        assert tokenizer.pad is not None
        assert tokenizer.mask is not None

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

        for add_bos in [True, False]:
            for add_eos in [True, False]:
                for return_as in ["pt", "py", "np"]:
                    for traj in tqdm(samples, desc="tokenizing"):
                        r = tokenizer.tokenize_loc_seq(
                            traj, return_as=return_as, add_bos=add_bos, add_eos=add_eos
                        )
                        if return_as == "pt":
                            assert isinstance(r, torch.Tensor)
                        elif return_as == "np":
                            assert isinstance(r, np.ndarray)
                        elif return_as == "py":
                            assert isinstance(r, list)
                        if add_bos:
                            assert r[0] == tokenizer.bos
                        if add_eos:
                            assert r[-1] == tokenizer.eos

        with pytest.raises(
            ValueError, match=re.escape("`return_as` only supports {'py', 'np', 'pt'}")
        ):
            tokenizer.tokenize_loc_seq(samples[0], return_as="test")

        assert tokenizer.tokenize_loc_seq(
            loc_seq=pa_sample
        ) == tokenizer.tokenize_loc_seq(loc_seq=samples[0])

    tokenizers = [locseq_tokenizer, LocSeqTokenizer.load_pretrained(path)]
    for tokenizer in tokenizers:
        test_tokenizer(tokenizer=tokenizer)

    load_tokenizer_test_func(tokenizer=locseq_tokenizer, path=path)
