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

import re
from typing import List

import numpy as np
import orjson
import pytest
import torch
from trajdl import Trajectory
from trajdl.tokenizers.t2vec import T2VECTokenizer


def test_t2vec_tokenizer(
    t2vec_tokenizer: T2VECTokenizer, traj_samples: List[Trajectory]
):
    samples = traj_samples[:5]
    for traj in samples:
        t2vec_tokenizer.encode_traj(traj)

    # TODO: 新增序列化支持
    # path = os.path.join(FOLDER, "test_t2vec_tokenizer.pkl")
    # t2vec_tokenizer.save_pretrained(path)

    with open("py_tests/data/t2vec_tokenized_results.json", "rb") as f:
        with_bos_and_eos_result = orjson.loads(f.read())

    without_bos_and_eos_result = [label[1:-1] for label in with_bos_and_eos_result]

    def test_special_token(tokenizer: T2VECTokenizer):
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

    def test_tokenizer(tokenizer: T2VECTokenizer, has_nearest_hot_map: bool):
        assert isinstance(tokenizer, T2VECTokenizer)

        assert len(tokenizer) == 1171
        assert (len(tokenizer.nearest_hot_map) != 0) is has_nearest_hot_map
        test_special_token(tokenizer)

        for add_start_end_token in [True, False]:
            for return_as in ["pt", "py", "np"]:
                for idx, traj in enumerate(samples):
                    r = tokenizer.encode_traj(
                        traj,
                        add_bos=add_start_end_token,
                        add_eos=add_start_end_token,
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
            ValueError, match=re.escape("`return_as` only supports {'py', 'np', 'pt'}")
        ):
            tokenizer.encode_traj(samples[0], return_as="test")

        assert len(tokenizer.nearest_hot_map) != 0

    # TODO: 新增序列化支持
    # tokenizers = [
    #     (t2vec_tokenizer, True),
    #     (T2VECTokenizer.load_pretrained(path), False),
    # ]
    tokenizers = [
        (t2vec_tokenizer, True),
    ]
    for tokenizer, has_nearest_hot_map in tokenizers:
        test_tokenizer(tokenizer=tokenizer, has_nearest_hot_map=has_nearest_hot_map)

    # TODO: 增加序列化支持
    # load_tokenizer_test_func(tokenizer=t2vec_tokenizer, path=path)
