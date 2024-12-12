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
from abc import ABC, abstractclassmethod, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pyarrow as pa
import torch

from ..common.enum import TokenEnum
from ..datasets.base import LocSeq, Trajectory


class AbstractTokenizer(ABC):
    _bos = None
    _eos = None
    _unk = None
    _pad = None
    _mask = None

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.check_vocab(vocab)
        self._init_special_token()

    @classmethod
    @abstractmethod
    def construct_vocab(cls, *args, **kwargs) -> Dict[str, int]:
        """静态方法，根据输入数据构造词汇表"""
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractclassmethod
    def build(cls, *args, **kwargs) -> "AbstractTokenizer":
        """类方法，用于构建Tokenizer实例，可以根据子类需求调整参数"""
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractmethod
    def loc2idx(self, loc: str) -> int:
        """将位置转换为下标"""
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def check_vocab(self, vocab: Dict[str, int]):
        """
        check the correctness of vocab
        """
        min_idx = min(vocab.values())
        if min_idx != 0:
            raise ValueError("Minimal index in vocab should be 0.")
        c = Counter(vocab.values())
        max_freq = max(c.values())
        if max_freq > 1:
            raise ValueError("Indices should be different in vocab.")
        for token in TokenEnum:
            if token.value not in vocab:
                raise ValueError(f"Token '{token}' should exist in vocab.")

    @property
    def bos(self) -> int:
        return self._bos

    @property
    def eos(self) -> int:
        return self._eos

    @property
    def unk(self) -> int:
        return self._unk

    @property
    def pad(self) -> int:
        return self._pad

    @property
    def mask(self) -> int:
        return self._mask

    def save_pretrained(self, path: str) -> None:
        """保存预训练模型到指定路径"""
        folder = os.path.split(path)[0]
        os.makedirs(folder, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pretrained(path: Union[str, Path]) -> "AbstractTokenizer":
        """静态方法，从指定路径加载预训练模型"""
        with open(Path(path), "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer

    def tokenize_loc_seq(
        self,
        loc_seq: Union[Iterable[str], LocSeq, pa.ListScalar],
        add_bos: bool = False,
        add_eos: bool = False,
        return_as: str = "py",
    ) -> Union[List[int], np.ndarray, torch.LongTensor]:
        """公共接口，用于将位置序列转换为数字序列"""

        token_result = self._tokenize_loc_seq_impl(loc_seq)
        r = []
        if add_bos:
            r.append(self.bos)
        r.extend(token_result)
        if add_eos:
            r.append(self.eos)

        if return_as == "py":
            return r
        elif return_as == "np":
            return np.array(r, dtype=np.int64)
        elif return_as == "pt":
            return torch.LongTensor(r)
        else:
            raise ValueError("`return_as` only supports {'py', 'np', 'pt'}")

    def _init_special_token(self):
        """
        initialize some special tokens
        """
        self._bos = self.loc2idx(TokenEnum.BOS_TOKEN.value)
        self._eos = self.loc2idx(TokenEnum.EOS_TOKEN.value)
        self._unk = self.loc2idx(TokenEnum.UNK_TOKEN.value)
        self._pad = self.loc2idx(TokenEnum.PAD_TOKEN.value)
        self._mask = self.loc2idx(TokenEnum.MASK_TOKEN.value)

    @abstractmethod
    def _tokenize_loc_seq_impl(
        self, loc_seq: Union[Iterable[str], LocSeq, pa.ListScalar]
    ) -> List[int]:
        """实现将位置序列转换为token序列的逻辑，具体实现由子类完成"""
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    def __len__(self) -> int:
        """size of vocab"""
        return len(self.vocab)


class AbstractLocSeqTokenizer(AbstractTokenizer):
    """
    tokenizer for location sequences
    """


class AbstractTrajTokenizer(AbstractTokenizer):
    """
    tokenizer for trajectories
    """

    @abstractmethod
    def tokenize_traj(
        self, traj: Trajectory, add_start_end_token: bool = False, return_as: str = "py"
    ) -> List[int]:
        """
        transform trajectory into location sequence
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover
