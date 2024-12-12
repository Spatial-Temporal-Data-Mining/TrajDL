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
import pytest
import torch

from trajdl.datasets import Trajectory
from trajdl.tokenizers.t2vec import T2VECTokenizer

from ..conftest import FOLDER
from . import load_tokenizer_test_func


def test_t2vec_tokenizer(
    t2vec_tokenizer: T2VECTokenizer, traj_samples: List[Trajectory]
):
    samples = traj_samples[:5]
    for traj in samples:
        t2vec_tokenizer.tokenize_traj(traj)

    path = os.path.join(FOLDER, "test_t2vec_tokenizer.pkl")
    t2vec_tokenizer.save_pretrained(path)

    with_bos_and_eos_result = [
        [
            1166,
            604,
            663,
            727,
            853,
            802,
            1085,
            1015,
            415,
            1124,
            416,
            239,
            48,
            24,
            307,
            779,
            81,
            1167,
        ],
        [
            1166,
            242,
            36,
            85,
            219,
            914,
            70,
            424,
            442,
            426,
            257,
            1044,
            1044,
            1044,
            1044,
            411,
            411,
            411,
            411,
            1167,
        ],
        [
            1166,
            292,
            209,
            327,
            127,
            579,
            604,
            878,
            852,
            908,
            1085,
            1015,
            679,
            368,
            1124,
            416,
            367,
            48,
            24,
            14,
            147,
            155,
            1001,
            135,
            206,
            490,
            80,
            58,
            274,
            1067,
            309,
            274,
            23,
            23,
            1115,
            909,
            909,
            669,
            669,
            805,
            805,
            1048,
            1048,
            677,
            967,
            975,
            415,
            329,
            917,
            1085,
            692,
            853,
            835,
            663,
            579,
            127,
            1167,
        ],
        [
            1166,
            1150,
            1150,
            1150,
            1156,
            743,
            320,
            547,
            356,
            261,
            0,
            38,
            2,
            3,
            304,
            534,
            943,
            929,
            527,
            689,
            872,
            775,
            854,
            791,
            353,
            256,
            1027,
            425,
            616,
            96,
            1167,
        ],
        [
            1166,
            641,
            641,
            655,
            1011,
            1040,
            213,
            605,
            571,
            774,
            774,
            1155,
            1155,
            780,
            780,
            780,
            780,
            780,
            780,
            780,
            891,
            891,
            891,
            891,
            891,
            891,
            1167,
        ],
    ]

    without_bos_and_eos_result = [
        [
            604,
            663,
            727,
            853,
            802,
            1085,
            1015,
            415,
            1124,
            416,
            239,
            48,
            24,
            307,
            779,
            81,
        ],
        [
            242,
            36,
            85,
            219,
            914,
            70,
            424,
            442,
            426,
            257,
            1044,
            1044,
            1044,
            1044,
            411,
            411,
            411,
            411,
        ],
        [
            292,
            209,
            327,
            127,
            579,
            604,
            878,
            852,
            908,
            1085,
            1015,
            679,
            368,
            1124,
            416,
            367,
            48,
            24,
            14,
            147,
            155,
            1001,
            135,
            206,
            490,
            80,
            58,
            274,
            1067,
            309,
            274,
            23,
            23,
            1115,
            909,
            909,
            669,
            669,
            805,
            805,
            1048,
            1048,
            677,
            967,
            975,
            415,
            329,
            917,
            1085,
            692,
            853,
            835,
            663,
            579,
            127,
        ],
        [
            1150,
            1150,
            1150,
            1156,
            743,
            320,
            547,
            356,
            261,
            0,
            38,
            2,
            3,
            304,
            534,
            943,
            929,
            527,
            689,
            872,
            775,
            854,
            791,
            353,
            256,
            1027,
            425,
            616,
            96,
        ],
        [
            641,
            641,
            655,
            1011,
            1040,
            213,
            605,
            571,
            774,
            774,
            1155,
            1155,
            780,
            780,
            780,
            780,
            780,
            780,
            780,
            891,
            891,
            891,
            891,
            891,
            891,
        ],
    ]

    def test_tokenizer(tokenizer: T2VECTokenizer, has_nearest_hot_map: bool):
        assert isinstance(tokenizer, T2VECTokenizer)

        assert len(tokenizer) == 1171
        assert (len(tokenizer.nearest_hot_map) != 0) is has_nearest_hot_map

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

        for add_start_end_token in [True, False]:
            for return_as in ["pt", "py", "np"]:
                for idx, traj in enumerate(samples):
                    r = tokenizer.tokenize_traj(
                        traj,
                        add_start_end_token=add_start_end_token,
                        return_as=return_as,
                    )
                    if return_as == "pt":
                        assert isinstance(r, torch.Tensor)
                    elif return_as == "np":
                        assert isinstance(r, np.ndarray)
                    elif return_as == "py":
                        assert isinstance(r, list)
                        if add_start_end_token:
                            assert r == with_bos_and_eos_result[idx]
                        else:
                            assert r == without_bos_and_eos_result[idx]
                    if add_start_end_token:
                        assert r[0] == tokenizer.bos
                        assert r[-1] == tokenizer.eos

        with pytest.raises(
            ValueError, match=re.escape("`return_as` could only be {'py', 'np', 'pt'}")
        ):
            tokenizer.tokenize_traj(samples[0], return_as="test")

        assert len(tokenizer.nearest_hot_map) != 0

    tokenizers = [
        (t2vec_tokenizer, True),
        (T2VECTokenizer.load_pretrained(path), False),
    ]
    for tokenizer, has_nearest_hot_map in tokenizers:
        test_tokenizer(tokenizer=tokenizer, has_nearest_hot_map=has_nearest_hot_map)

    load_tokenizer_test_func(tokenizer=t2vec_tokenizer, path=path)
