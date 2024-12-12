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
import pickle
import warnings
from pathlib import Path
from typing import Union

import torch


class Bucketizer:
    """
    A class to create buckets for numerical data, allowing for indexing of values into buckets.

    Parameters
    ----------
    lower_bound : float
        The lower bound of the range.
    upper_bound : float
        The upper bound of the range.
    num_buckets : int
        The number of buckets to create within the specified range.

    Attributes
    ----------
    bucket_size : float
        The size of each bucket.

    Methods
    -------
    get_bucket_index(value):
        Returns the index of the bucket that the given value falls into.
    """

    def __init__(self, lower_bound: float, upper_bound: float, num_buckets: int):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._num_buckets = num_buckets
        self.bucket_size = (upper_bound - lower_bound) / num_buckets

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def num_buckets(self) -> int:
        return self._num_buckets

    def get_bucket_index(self, value: float):
        """
        Get the index of the bucket that the given value belongs to.

        Parameters
        ----------
        value : float
            The value to be placed in a bucket.

        Returns
        -------
        int
            The index of the bucket that contains the value.
        """
        if value < self.lower_bound or value > self.upper_bound:
            warnings.warn(
                f"Value {value} is out of bounds ({self.lower_bound}, {self.upper_bound})",
                RuntimeWarning,
            )

        if value <= self.lower_bound:
            return 0
        if value >= self.upper_bound:
            return self.num_buckets - 1

        # 计算桶的索引
        idx = int((value - self.lower_bound) / self.bucket_size)

        # 确保idx不超过num_buckets - 1，这里会保证upper_bound属于最后一个桶
        return min(idx, self.num_buckets - 1)

    def get_bucket_indices(self, tensor: torch.Tensor) -> torch.LongTensor:
        """
        Get the indices of the buckets for each value in the input tensor.
        """
        indices = (
            ((tensor - self.lower_bound) / self.bucket_size)
            .clamp(0, self.num_buckets - 1)
            .floor()
        )

        return indices.long()

    def save(self, path: Union[str, Path]):
        p = Path(path)
        folder = p.parent
        os.makedirs(folder, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "Bucketizer":
        with open(path, "rb") as f:
            return pickle.load(f)
