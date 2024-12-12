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

import multiprocessing
import os
import re
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import yaml

from ..tokenizers.abstract import AbstractTokenizer
from ..tokenizers.slot import Bucketizer
from ..tokenizers.t2vec import T2VECTokenizer


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


def get_num_cpus() -> int:
    return multiprocessing.cpu_count()


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


def find_best_checkpoint(dir_path: str, is_maximizing: bool = True) -> str:
    """
    Find the best checkpoint file based on validation loss.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing checkpoint files.

    is_maximizing : bool, optional
        Flag indicating whether to maximize or minimize validation loss.
        If True, the function seeks the checkpoint with the highest validation loss.
        If False, it seeks the checkpoint with the lowest validation loss.
        Default is True.

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

    # 匹配文件名中的 epoch 和 val_loss
    pattern = r"model-epoch=(\d+)-val_loss=([0-9e\.-]+).ckpt"

    for file in files:
        match = re.match(pattern, file)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            checkpoints.append((epoch, val_loss, file))

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
