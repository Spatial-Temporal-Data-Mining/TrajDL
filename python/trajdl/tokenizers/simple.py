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
from typing import Dict

from ..common.enum import TokenEnum
from .abstract import AbstractLocSeqTokenizer


class SimpleTokenizer(AbstractLocSeqTokenizer):
    """A tokenizer built from a pre-defined, initial vocabulary.

    This tokenizer is useful when the set of all possible locations is already
    known and provided as an initial vocabulary. It simply enriches this
    vocabulary with the required special tokens.
    """

    def __init__(self, vocab: Dict[str, int]):
        super().__init__(vocab=vocab)

    @classmethod
    def from_vocab(cls, init_vocab: Dict[str, int]) -> "SimpleTokenizer":
        """Adds special tokens to an initial vocabulary.

        It finds the maximum index in the provided vocabulary and appends the
        standard special tokens (BOS, EOS, etc.) with subsequent indices.

        Parameters
        ----------
        init_vocab : Dict[str, int]
            The user-provided initial vocabulary.

        Returns
        -------
        Dict[str, int]
            The vocabulary enriched with special tokens.
        """
        vocab = dict(init_vocab)
        next_idx = max(vocab.values()) + 1 if vocab else 0

        for token in [
            TokenEnum.BOS_TOKEN.value,
            TokenEnum.EOS_TOKEN.value,
            TokenEnum.UNK_TOKEN.value,
            TokenEnum.MASK_TOKEN.value,
            TokenEnum.PAD_TOKEN.value,
        ]:
            if token in init_vocab:
                warnings.warn(
                    f"Token '{token}' exist in vocab, tokenizer will not give this token a index automatically.",
                    RuntimeWarning,
                )
                continue

            init_vocab[token] = next_idx
            next_idx += 1

        return cls(init_vocab)
