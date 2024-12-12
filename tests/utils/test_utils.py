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

from unittest.mock import patch

import numpy as np
import pytest
import torch

from trajdl.utils import (
    find_best_checkpoint,
    get_num_cpus,
    tiny_value_of_dtype,
    try_gpu,
    valid_lengths_to_mask,
)


@patch("torch.cuda.device_count", return_value=1)
def test_try_gpu(mock_device_count):
    device = try_gpu()
    assert device == torch.device("cuda:0")

    device = try_gpu(1)
    assert device == torch.device("cpu")


@patch("multiprocessing.cpu_count", return_value=4)
def test_get_num_cpus(mock_cpu_count):
    assert get_num_cpus() == 4


@patch(
    "os.listdir",
    return_value=[
        "model-epoch=001-val_loss=1.0.ckpt",
        "model-epoch=002-val_loss=0.8.ckpt",
        "model-epoch=003-val_loss=1.5.ckpt",
    ],
)
def test_find_best_checkpoint(mock_listdir):
    best_checkpoint = find_best_checkpoint("dummy_path", is_maximizing=False)
    assert best_checkpoint == "model-epoch=002-val_loss=0.8.ckpt"


@patch(
    "os.listdir",
    return_value=[
        "tuler-epoch=001-val_loss=1.0.ckpt",
        "tuler-epoch=002-val_loss=0.8.ckpt",
        "tuler-epoch=003-val_loss=1.5.ckpt",
    ],
)
def test_find_best_checkpoint_error(mock_listdir):
    with pytest.raises(ValueError):
        find_best_checkpoint("dummy_path")


@patch(
    "os.listdir",
    return_value=[
        "model-epoch=001-val_loss=1.0.ckpt",
        "model-epoch=002-val_loss=0.8.ckpt",
        "model-epoch=003-val_loss=1.5.ckpt",
    ],
)
def test_find_best_checkpoint_maximizing(mock_listdir):
    best_checkpoint = find_best_checkpoint("dummy_path", is_maximizing=True)
    assert best_checkpoint == "model-epoch=003-val_loss=1.5.ckpt"


def test_tiny_value_of_dtype():
    assert np.allclose(tiny_value_of_dtype(torch.float), 1e-13)
    assert np.allclose(tiny_value_of_dtype(torch.double), 1e-13)
    assert np.allclose(tiny_value_of_dtype(torch.half), 1e-4)

    with pytest.raises(TypeError):
        tiny_value_of_dtype(torch.int)


def test_valid_lengths_to_mask():
    # Test with varying lengths
    valid_lengths = [3, 2, 5]
    expected_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    mask = valid_lengths_to_mask(valid_lengths)
    assert torch.equal(mask, expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"

    # Test with all sequences of the same length
    valid_lengths = [4, 4, 4]
    expected_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )
    mask = valid_lengths_to_mask(valid_lengths)
    assert torch.equal(mask, expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"

    # Test with single sequence
    valid_lengths = [1]
    expected_mask = torch.tensor([[1.0]])
    mask = valid_lengths_to_mask(valid_lengths)
    assert torch.equal(mask, expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"

    # Test with empty input
    valid_lengths = []
    expected_mask = torch.empty((0, 0))
    mask = valid_lengths_to_mask(valid_lengths)
    assert torch.equal(mask, expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"

    # Test with single zero length
    valid_lengths = [0]
    expected_mask = torch.empty((1, 0))
    mask = valid_lengths_to_mask(valid_lengths)
    assert torch.equal(mask, expected_mask), f"Expected:\n{expected_mask}\nGot:\n{mask}"
