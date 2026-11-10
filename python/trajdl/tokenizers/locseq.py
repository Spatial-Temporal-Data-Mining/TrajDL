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

import itertools
from typing import Iterable

from tqdm.contrib import tenumerate

from .. import LocSeq, count_locations
from ..common.enum import TokenEnum
from .abstract import AbstractLocSeqTokenizer


class LocSeqTokenizer(AbstractLocSeqTokenizer):
    """A tokenizer that builds its vocabulary from a corpus of location sequences.

    This is a standard approach for NLP-style tasks where the vocabulary is
    derived from the frequency of tokens in a training dataset.
    """

    @classmethod
    def fit(
        cls,
        splits: Iterable[Iterable[LocSeq]],
        count_start_end_token: bool = False,
        min_count: int = 0,
    ) -> "LocSeqTokenizer":
        """Fits a LocSeqTokenizer instance from multiple splits of location sequences.

        This is a factory method that orchestrates the vocabulary construction
        and the instantiation of the tokenizer.

        Constructs a vocabulary from multiple iterables of location sequences
        (e.g., train, val, test splits). All splits are merged before counting.

        This method performs several steps:
        1. Merges all splits into a single sequence stream.
        2. Counts the frequency of each location token in the merged sequences.
        3. Filters out tokens with a frequency below `min_count`.
        4. Sorts the remaining tokens by frequency in descending order.
        5. Ensures that all special tokens (BOS, EOS, UNK, MASK, PAD) are included.
        6. Assigns a unique integer index to each token, creating the final vocabulary.

        Parameters
        ----------
        splits : Iterable[Iterable[LocSeq]]
            An iterable of splits, where each split is an iterable of location
            sequences. For example, pass ``[train_seqs]`` for a single split or
            ``[train_seqs, val_seqs, test_seqs]`` to build the vocabulary from
            multiple splits.
        count_start_end_token : bool, optional
            If True, counts the BOS and EOS tokens for each sequence.
            Defaults to False.
        min_count : int, optional
            The minimum frequency for a token to be included in the vocabulary.
            Tokens with a count lower than this will be discarded.
            Defaults to 0, which means no filtering.

        Returns
        -------
        LocSeqTokenizer
            A new instance of the LocSeqTokenizer.
        """
        token_counts = count_locations(
            list(itertools.chain.from_iterable(splits)),
            TokenEnum.BOS_TOKEN.value if count_start_end_token else None,
            TokenEnum.EOS_TOKEN.value if count_start_end_token else None,
        )
        print(f"num locs {len(token_counts)}")
        assert (
            TokenEnum.PAD_TOKEN.value not in token_counts
        ), "token [PAD] should not exist in loc_seqs"

        # Filter tokens based on their frequency (min_count)
        token_counts = (
            {
                token: count
                for token, count in token_counts.items()
                if count >= min_count
            }
            if min_count > 0
            else token_counts
        )

        # vocab_list is a list of (token, count) tuples.
        # Sort by frequency (descending) and then by token name (ascending) for tie-breaking.
        vocab_list = sorted(
            token_counts.items(), key=lambda x: (-x[1], x[0]), reverse=False
        )

        # Ensure all required special tokens are included in the vocabulary.
        # This is crucial for the tokenizer to function correctly.
        for token in (
            TokenEnum.BOS_TOKEN.value,
            TokenEnum.EOS_TOKEN.value,
            TokenEnum.UNK_TOKEN.value,
            TokenEnum.MASK_TOKEN.value,
            TokenEnum.PAD_TOKEN.value,
        ):
            if token not in token_counts:
                vocab_list.append((token, 0))

        # Assign a unique index to each token in the final vocabulary list.
        vocab = {
            token: idx
            for idx, (token, _) in tenumerate(vocab_list, desc="construct vocab")
        }
        return cls(vocab=vocab)
