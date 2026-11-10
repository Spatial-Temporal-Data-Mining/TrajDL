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
    AUTO_WORKER_CAP_DEFAULT,
    _cgroup_cpu_quota,
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


@patch("trajdl.utils._cgroup_cpu_quota", return_value=None)
@patch("os.sched_getaffinity", return_value=set(range(4)))
def test_get_num_cpus_affinity(mock_affinity, mock_quota):
    assert get_num_cpus() == 4


@patch("trajdl.utils._cgroup_cpu_quota", return_value=8)
@patch("os.sched_getaffinity", return_value=set(range(96)))
def test_get_num_cpus_cgroup_quota(mock_affinity, mock_quota):
    # 云上容器形态：亲和性是宿主 96 核，真正的配额在 CFS quota（8 核）
    assert get_num_cpus() == 8


def test_get_num_cpus_fallback(monkeypatch):
    import os

    monkeypatch.setattr("trajdl.utils._cgroup_cpu_quota", lambda: None)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert get_num_cpus() == 3


def test_get_num_cpus_auto_cap(monkeypatch):
    # CPU-rich but shm-poor hosts (e.g. 184 visible cores, 8 GiB /dev/shm)
    # must not spawn one worker per core: DataLoader shm pressure scales
    # with the worker count and the workers die with SIGBUS.
    import os

    monkeypatch.setattr("trajdl.utils._cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(184)))
    monkeypatch.delenv("TRJDL_AUTO_WORKER_CAP", raising=False)
    assert get_num_cpus() == AUTO_WORKER_CAP_DEFAULT


def test_get_num_cpus_auto_cap_env_override(monkeypatch):
    import os

    monkeypatch.setattr("trajdl.utils._cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(184)))
    monkeypatch.setenv("TRJDL_AUTO_WORKER_CAP", "8")
    assert get_num_cpus() == 8
    # A non-positive cap disables clamping entirely (escape hatch).
    monkeypatch.setenv("TRJDL_AUTO_WORKER_CAP", "0")
    assert get_num_cpus() == 184


def test_get_num_cpus_auto_cap_below_quota(monkeypatch):
    # The cap must never raise a count already below it.
    import os

    monkeypatch.setattr("trajdl.utils._cgroup_cpu_quota", lambda: 8)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(96)))
    monkeypatch.delenv("TRJDL_AUTO_WORKER_CAP", raising=False)
    assert get_num_cpus() == 8


def _patch_cgroup_open(monkeypatch, files):
    import builtins
    import io

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        key = str(path)
        if key in files:
            if files[key] is None:
                raise FileNotFoundError(key)
            return io.StringIO(files[key])
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_cgroup_cpu_quota_v2(monkeypatch):
    _patch_cgroup_open(monkeypatch, {"/sys/fs/cgroup/cpu.max": "800000 100000\n"})
    assert _cgroup_cpu_quota() == 8


def test_cgroup_cpu_quota_v2_unlimited(monkeypatch):
    _patch_cgroup_open(monkeypatch, {"/sys/fs/cgroup/cpu.max": "max 100000\n"})
    assert _cgroup_cpu_quota() is None


def test_cgroup_cpu_quota_v1(monkeypatch):
    _patch_cgroup_open(
        monkeypatch,
        {
            "/sys/fs/cgroup/cpu.max": None,
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "1200000\n",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
        },
    )
    assert _cgroup_cpu_quota() == 12


def test_cgroup_cpu_quota_v1_unlimited(monkeypatch):
    _patch_cgroup_open(
        monkeypatch,
        {
            "/sys/fs/cgroup/cpu.max": None,
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1\n",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
        },
    )
    assert _cgroup_cpu_quota() is None


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
        "model-epoch=001-val_loss=1.0.ckpt",
        "model-epoch=002-val_loss=0.8.ckpt",
        "model-epoch=003-val_loss=1.5.ckpt",
    ],
)
def test_find_best_checkpoint_default_minimizes(mock_listdir):
    best_checkpoint = find_best_checkpoint("dummy_path")
    assert best_checkpoint == "model-epoch=002-val_loss=0.8.ckpt"


@patch(
    "os.listdir",
    return_value=[
        "tuler-epoch=001-val_loss=1.0.ckpt",
        "tuler-epoch=002-val_loss=0.8.ckpt",
        "tuler-epoch=003-val_loss=1.5.ckpt",
    ],
)
def test_find_best_checkpoint_non_model_prefix(mock_listdir):
    # 文件名前缀不限（tuler-/trajcl-…），旧版正则锁定 model- 前缀导致此类文件被漏掉
    best_checkpoint = find_best_checkpoint("dummy_path")
    assert best_checkpoint == "tuler-epoch=002-val_loss=0.8.ckpt"


@patch(
    "os.listdir",
    return_value=[
        "trajcl-epoch=019-train_loss=1.143000.ckpt",
        "trajcl-epoch=018-train_loss=1.200000.ckpt",
    ],
)
def test_find_best_checkpoint_train_loss_metric(mock_listdir):
    # train_loss 等任意指标段同样解析（trajcl 预训练无 val loop 的实际形态）
    best_checkpoint = find_best_checkpoint("dummy_path")
    assert best_checkpoint == "trajcl-epoch=019-train_loss=1.143000.ckpt"


@patch(
    "os.listdir",
    return_value=[
        "last.ckpt",
        "model.ckpt",
        "epoch=001-val_loss=1.0.ckpt",
    ],
)
def test_find_best_checkpoint_error(mock_listdir):
    # 无 `{name}-epoch={epoch}-{metric}={value}.ckpt` 形态的文件全部不匹配
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
