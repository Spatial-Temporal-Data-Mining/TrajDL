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

import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import yaml

from ..tokenizers.abstract import AbstractTokenizer
from ..tokenizers.slot import Bucketizer
from ..tokenizers.t2vec import T2VECTokenizer
from .callbacks import EpochLogPrinter

__all__ = ["EpochLogPrinter"]


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def _cgroup_cpu_quota() -> Optional[int]:
    """
    CPU quota (in cores) enforced by the container's cgroup, or None.

    Containers enforce the vCPU allocation through the cgroup CFS quota
    (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` on cgroup v1) while the
    affinity mask stays unrestricted (all host cores), so neither
    ``sched_getaffinity`` nor ``os.cpu_count`` can see it.
    """
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
        return None
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


# Upper bound for the auto-detected worker count (``num_workers: -1``).
# DataLoader shared-memory pressure grows with the worker count, while the
# CPU-derived count above knows nothing about shm: a containerised host can
# be CPU-rich but shm-poor (observed: 184 visible cores, no CFS quota, 8 GiB
# /dev/shm), where a purely CPU-derived count overruns shm and every worker
# dies with SIGBUS. 16 workers already saturate a single-GPU input pipeline.
AUTO_WORKER_CAP_DEFAULT = 16


def _auto_worker_cap() -> int:
    """Upper bound applied to auto-detected worker counts.

    Reads ``TRJDL_AUTO_WORKER_CAP``; a value <= 0 (or unparsable input
    falling back to the default) disables clamping entirely, restoring the
    pure CPU-derived count.
    """
    raw = os.environ.get("TRJDL_AUTO_WORKER_CAP", "")
    try:
        cap = int(raw) if raw else AUTO_WORKER_CAP_DEFAULT
    except ValueError:
        cap = AUTO_WORKER_CAP_DEFAULT
    return cap if cap > 0 else 0


def get_num_cpus() -> int:
    """
    Number of DataLoader workers considered safe for this process.

    Takes the minimum of the affinity mask (cpuset-restricted containers),
    the cgroup CPU quota (CFS-quota containers), and an
    auto-worker cap. Neither ``multiprocessing.cpu_count()`` nor the raw
    affinity mask alone is enough: both report the host's core count inside
    a quota-limited container, over-subscribing DataLoader workers by an
    order of magnitude (e.g. 96 workers on an 8-vCPU quota). The cap covers
    the opposite pathology: hosts with no CPU quota but a small /dev/shm,
    where CPU-derived worker counts exhaust shm and workers die with SIGBUS.
    Set ``TRJDL_AUTO_WORKER_CAP=0`` to disable the cap.
    """
    candidates = []
    if hasattr(os, "sched_getaffinity"):
        candidates.append(len(os.sched_getaffinity(0)))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        candidates.append(quota)
    if not candidates:
        candidates.append(os.cpu_count() or 1)
    count = max(1, min(candidates))
    cap = _auto_worker_cap()
    return count if cap == 0 else min(count, cap)


def dist2weight(
    dist: np.ndarray, tokenizer: T2VECTokenizer, dist_decay_speed: float = 0.008
) -> torch.Tensor:
    """
    dist: (vocab_size, k)
    """
    dist = np.exp(-dist * dist_decay_speed)
    s = dist.sum(axis=1, keepdims=True)
    dist = dist / s
    ## The PAD should not contribute to the decoding loss
    dist[tokenizer.pad] = 0.0
    return torch.Tensor(dist).float()


def try_gpu(i=0):
    """Return gpu(i) if available, else return cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def find_best_checkpoint(dir_path: str, is_maximizing: bool = False) -> str:
    """
    Find the best checkpoint file based on the metric encoded in its filename
    (e.g. ``val_loss`` or ``val_acc1``).

    Parameters
    ----------
    dir_path : str
        The path to the directory containing checkpoint files.

    is_maximizing : bool, optional
        Flag indicating whether to maximize or minimize the metric.
        If True, the function seeks the checkpoint with the highest value
        (e.g. val_acc1). If False, it seeks the checkpoint with the lowest
        value (e.g. val_loss). Default is False.

    Returns
    -------
    str
        The filename of the best checkpoint based on the specified criteria.

    Raises
    ------
    ValueError
        If no valid checkpoint files are found in the specified directory.

    Examples
    --------
    >>> best_checkpoint_name = find_best_checkpoint("path_to_your_directory", is_maximizing=False)
    >>> print(best_checkpoint_name)
    'model-epoch=004-val_loss=2.498780.ckpt'
    """

    # 获取目录下的所有文件
    files = os.listdir(dir_path)

    # 用于存储校验损失和对应的文件名
    checkpoints = []

    # 匹配文件名中的 epoch 和被监控指标（Lightning checkpoint 文件名模板
    # `{name}-epoch={epoch}-{metric}={value}.ckpt` 的指标段；name 前缀不限
    # （model-/tuler-/trajcl-…），last.ckpt 等无 `epoch=` 段的文件天然不匹配）
    pattern = r"[A-Za-z0-9_]+-epoch=(\d+)-([A-Za-z0-9_]+)=([0-9e\.-]+)\.ckpt"

    for file in files:
        match = re.match(pattern, file)
        if match:
            epoch = int(match.group(1))
            metric = float(match.group(3))
            checkpoints.append((epoch, metric, file))

    # 如果没有找到有效的 checkpoint，抛出错误
    if not checkpoints:
        raise ValueError("No valid checkpoint files found in the specified directory.")

    # 根据最大值或最小值标识选择最优的 checkpoint
    if is_maximizing:
        best_checkpoint = max(checkpoints, key=lambda x: x[1])
    else:
        best_checkpoint = min(checkpoints, key=lambda x: x[1])

    return best_checkpoint[2]  # 返回文件名


def load_tokenizer(tokenizer: Union[str, Path, AbstractTokenizer]) -> AbstractTokenizer:
    """
    load a tokenizer from a path of instance
    """
    if isinstance(tokenizer, str) or isinstance(tokenizer, Path):
        return AbstractTokenizer.load_pretrained(tokenizer)
    elif isinstance(tokenizer, AbstractTokenizer):
        return tokenizer
    else:
        raise ValueError(
            "`tokenizer` only support path of tokenizer or tokenizer instance"
        )


def load_bucketizer(bucketizer: Union[str, Bucketizer]) -> Bucketizer:
    if isinstance(bucketizer, str):
        return Bucketizer.load(bucketizer)
    elif isinstance(bucketizer, Bucketizer):
        return bucketizer
    else:
        raise ValueError(
            "`bucketizer` only support path of Bucketizer or Bucketizer instance"
        )


def valid_lengths_to_mask(valid_lengths: List[int]) -> torch.Tensor:
    """
    Generate a mask tensor based on the valid lengths of sequences.

    Parameters
    ----------
    valid_lengths : List[int]
        A list of integers representing the valid lengths of each sequence in the batch.

    Returns
    -------
    torch.Tensor
        A binary mask tensor of shape (N, max_length), where N is the number of sequences
        and max_length is the maximum valid length found in the input list.
        Each row corresponds to a sequence with 1s up to its valid length
        and 0s beyond that.

    Example
    -------
    >>> valid_lengths_to_mask([3, 2, 5])
    tensor([[1., 1., 1., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 1., 1.]])
    """
    if not valid_lengths:
        return torch.empty((0, 0))

    max_length = max(valid_lengths)
    mask = torch.arange(max_length)[None, :] < torch.tensor(valid_lengths)[:, None]
    return mask.float()  # Convert boolean tensor to float for binary mask


def export(func_or_class):
    """
    Automatically add the function or class to __all__
    """
    module = sys.modules[func_or_class.__module__]

    # 如果 __all__ 不存在，创建它
    if not hasattr(module, "__all__"):
        module.__all__ = []

    # 添加到 __all__
    module.__all__.append(func_or_class.__name__)

    return func_or_class
