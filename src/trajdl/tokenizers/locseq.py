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

from collections import defaultdict
from typing import Dict, Iterable, List, Union

import pyarrow as pa
from tqdm import tqdm
from tqdm.contrib import tenumerate

from ..common.enum import TokenEnum
from ..datasets.base import LocSeq
from .abstract import AbstractLocSeqTokenizer


class LocSeqTokenizer(AbstractLocSeqTokenizer):
    @classmethod
    def construct_vocab(
        cls,
        loc_seqs: Iterable[LocSeq],
        count_start_end_token: bool = True,
        min_count: int = 0,
        enable_progress_bar: bool = False,
    ) -> Dict[str, int]:
        token_counts = defaultdict(int)
        for loc_seq in (
            tqdm(loc_seqs, desc="counting tokens") if enable_progress_bar else loc_seqs
        ):
            for loc in loc_seq:
                token_counts[loc] += 1
            if count_start_end_token:
                token_counts[TokenEnum.BOS_TOKEN.value] += 1
                token_counts[TokenEnum.EOS_TOKEN.value] += 1
        assert (
            TokenEnum.PAD_TOKEN.value not in token_counts
        ), "token [PAD] should not exist in loc_seqs"

        # filter tokens
        token_counts = {
            token: count for token, count in token_counts.items() if count >= min_count
        }

        # 二元组，第一项是token，第二项是出现次数，按出现次数降序排列
        vocab_list = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        # 将special tokens添加到vocab list里面，确保其都被加入到了vocab list里面
        for token in (
            TokenEnum.BOS_TOKEN.value,
            TokenEnum.EOS_TOKEN.value,
            TokenEnum.UNK_TOKEN.value,
            TokenEnum.MASK_TOKEN.value,
        ):
            if token not in token_counts:
                vocab_list.append((token, 0))
        vocab_list.append((TokenEnum.PAD_TOKEN.value, 0))

        return {
            token: idx
            for idx, (token, _) in tenumerate(vocab_list, desc="construct vocab")
        }

    @classmethod
    def build(
        cls,
        loc_seqs: Iterable[LocSeq],
        count_start_end_token: bool = False,
        min_count: int = 0,
        enable_progress_bar: bool = False,
    ) -> "LocSeqTokenizer":
        vocab = cls.construct_vocab(
            loc_seqs=loc_seqs,
            count_start_end_token=count_start_end_token,
            min_count=min_count,
            enable_progress_bar=enable_progress_bar,
        )
        return LocSeqTokenizer(vocab=vocab)

    def loc2idx(self, loc: str) -> int:
        return self.vocab.get(loc, self.unk)

    def _tokenize_loc_seq_impl(
        self, loc_seq: Union[Iterable[str], LocSeq, pa.ListScalar]
    ) -> List[int]:
        if isinstance(loc_seq, pa.ListScalar):
            loc_seq = loc_seq.as_py()
        return [self.loc2idx(loc) for loc in loc_seq]
