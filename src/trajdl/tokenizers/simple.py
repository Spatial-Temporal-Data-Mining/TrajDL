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

import warnings
from typing import Dict, Iterable, List, Union

import pyarrow as pa

from ..common.enum import TokenEnum
from ..datasets.base import LocSeq
from .abstract import AbstractLocSeqTokenizer


class SimpleTokenizer(AbstractLocSeqTokenizer):
    @classmethod
    def construct_vocab(cls, init_vocab: Dict[str, int]) -> Dict[str, int]:
        max_idx = max(init_vocab.values())
        for idx, token in enumerate(
            [
                TokenEnum.BOS_TOKEN.value,
                TokenEnum.EOS_TOKEN.value,
                TokenEnum.UNK_TOKEN.value,
                TokenEnum.MASK_TOKEN.value,
                TokenEnum.PAD_TOKEN.value,
            ]
        ):
            if token in init_vocab:
                warnings.warn(
                    f"Token '{token}' exist in vocab, tokenizer will not give this token a index automatically.",
                    RuntimeWarning,
                )
            init_vocab[token] = max_idx + idx + 1
        return init_vocab

    @classmethod
    def build(cls, init_vocab: Dict[str, int]) -> "SimpleTokenizer":
        vocab = cls.construct_vocab(init_vocab)
        return cls(vocab=vocab)

    def loc2idx(self, loc: str) -> int:
        return self.vocab.get(loc, self._unk)

    def _tokenize_loc_seq_impl(
        self, loc_seq: Union[Iterable[str], LocSeq, pa.ListScalar]
    ) -> List[int]:
        if isinstance(loc_seq, pa.ListScalar):
            loc_seq = loc_seq.as_py()
        return [self.loc2idx(loc) for loc in loc_seq]
