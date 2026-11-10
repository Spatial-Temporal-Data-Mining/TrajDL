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
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pyarrow as pa
import torch

from .. import LocSeq, Trajectory, TrajectoryView
from ..common.enum import TokenEnum


class AbstractTokenizer(ABC):
    """Abstract base class for all tokenizers.

    This class defines the common interface and shared functionality for converting
    location-based sequences into numerical token IDs. It handles vocabulary management,
    special tokens, serialization, and provides a generic tokenization pipeline.

    Subclasses must implement `construct_vocab` and `fit` class methods to
    provide specific strategies for vocabulary creation.
    """

    _bos = -1
    _eos = -1
    _unk = -1
    _pad = -1
    _mask = -1
    vocab: Dict[str, int] = {}
    id2token: Dict[int, str] = {}

    def __init__(self, vocab: Dict[str, int]):
        """Initializes the tokenizer with a given vocabulary.

        Parameters
        ----------
        vocab : Dict[str, int]
            A dictionary mapping location strings to their unique integer IDs.
        """
        self.check_vocab(vocab)
        self.vocab = vocab
        self._init_special_token()
        self.id2token = {idx: token for token, idx in vocab.items()}

    def check_vocab(self, vocab: Dict[str, int]):
        """Validates the structural integrity of the vocabulary.

        Ensures that:
        1. The minimum index is 0.
        2. All indices are unique.
        3. All required special tokens (BOS, EOS, UNK, PAD, MASK) are present.

        Parameters
        ----------
        vocab : Dict[str, int]
            The vocabulary to check.

        Raises
        ------
        ValueError
            If any of the validation checks fail.
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
        for special_token in TokenEnum.values():
            if special_token not in vocab:
                raise ValueError(f"special token {special_token} not in vocab")

    def _init_special_token(self):
        """Initializes special token attributes (_bos, _eos, etc.) by looking
        up their string representations in the vocabulary."""
        self._bos = self.token_to_id(TokenEnum.BOS_TOKEN.value)
        self._eos = self.token_to_id(TokenEnum.EOS_TOKEN.value)
        self._unk = self.token_to_id(TokenEnum.UNK_TOKEN.value)
        self._pad = self.token_to_id(TokenEnum.PAD_TOKEN.value)
        self._mask = self.token_to_id(TokenEnum.MASK_TOKEN.value)

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

    def id_to_token(self, idx: int) -> str:
        """Converts a token ID back to its string representation.

        Parameters
        ----------
        idx : int
            The token ID to convert.

        Returns
        -------
        str
            The corresponding token string.

        Raises
        ------
        KeyError
            If the ID does not exist in the vocabulary.
        """
        return self.id2token[idx]

    def save_pretrained(self, path: str | Path) -> None:
        """Saves the tokenizer instance to a file using pickle.

        Parameters
        ----------
        path : str
            The file path where the tokenizer will be saved.
        """
        folder = os.path.split(path)[0]
        os.makedirs(folder, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pretrained(path: Union[str, Path]) -> "AbstractTokenizer":
        """Loads a tokenizer instance from a file.

        Parameters
        ----------
        path : Union[str, Path]
            The file path from which to load the tokenizer.

        Returns
        -------
        AbstractTokenizer
            The loaded tokenizer instance.
        """
        with open(Path(path), "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer

    def token_to_id(self, loc: str | None) -> int:
        """Converts a single location string to its corresponding index.

        If the token is not found in the vocabulary (out-of-vocabulary),
        the index for the unknown token (UNK) is returned.

        Parameters
        ----------
        loc : str | None
            The location string to convert.

        Returns
        -------
        int
            The corresponding integer ID.
        """
        return self.vocab.get(loc, self.unk) if loc else self.unk

    def encode_locseq(
        self,
        loc_seq: Union[Iterable[str], LocSeq, pa.ListScalar],
        add_bos: bool = False,
        add_eos: bool = False,
        return_as: str = "py",
    ) -> Union[List[int], np.ndarray, torch.LongTensor]:
        """Tokenizes a sequence of location strings into a sequence of IDs.

        This is the main public-facing method for tokenization. It adds
        special tokens (BOS/EOS) if requested and formats the output.

        Parameters
        ----------
        loc_seq : Union[Iterable[str], LocSeq, pa.ListScalar]
            An iterable of location strings to be tokenized.
        add_bos : bool, optional
            Whether to prepend the Beginning-Of-Sequence (BOS) token, by default False.
        add_eos : bool, optional
            Whether to append the End-Of-Sequence (EOS) token, by default False.
        return_as : str, optional
            The desired output format. Can be 'py' (Python list), 'np' (NumPy array),
            or 'pt' (PyTorch tensor), by default 'py'.

        Returns
        -------
        Union[List[int], np.ndarray, torch.LongTensor]
            The resulting sequence of token IDs in the specified format.
        """

        token_result = self._encode_locseq_impl(loc_seq)
        r = []
        if add_bos:
            r.append(self.bos)
        r.extend(token_result)
        if add_eos:
            r.append(self.eos)

        return self._format_output(r, return_as)

    def encode_locseq_batch(
        self,
        loc_col: Union["pa.ChunkedArray", "pa.Array", Iterable[Iterable[str]]],
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        """Batch-tokenizes a whole varlen location column in one pass.

        Single flat pass over all rows (one Arrow materialization + one
        vocab-lookup sweep) instead of per-row ``ListScalar`` iteration and
        per-row tensor construction. Token IDs are identical to per-row
        ``encode_locseq(..., return_as="np")``.

        Parameters
        ----------
        loc_col : Union[pa.ChunkedArray, pa.Array, Iterable[Iterable[str]]]
            A varlen string column (e.g. ``LocSeqDataset.seq``) or an
            iterable of per-row location sequences (e.g. ``LocSeqView.seq``).
        add_bos : bool, optional
            Whether to prepend BOS to every row, by default False.
        add_eos : bool, optional
            Whether to append EOS to every row, by default False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(flat_ids int64[T], offsets int64[N+1])``: row ``i``'s ids are
            ``flat_ids[offsets[i]:offsets[i+1]]``.
        """
        from ..datasets.arrow.convert import flat_values_offsets

        if isinstance(loc_col, (pa.ChunkedArray, pa.Array)):
            values, offsets = flat_values_offsets(loc_col)
            flat_list = values.tolist()
            offsets = offsets.astype(np.int64, copy=False)
        else:
            seqs = [s.as_py() if isinstance(s, pa.ListScalar) else s for s in loc_col]
            offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
            np.cumsum([len(s) for s in seqs], out=offsets[1:])
            flat_list = [loc for seq in seqs for loc in seq]

        ids = np.fromiter(
            (self.token_to_id(loc) for loc in flat_list),
            dtype=np.int64,
            count=len(flat_list),
        )

        lens = np.diff(offsets)
        if not add_bos and not add_eos:
            return ids, offsets
        out_offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens + int(add_bos) + int(add_eos), out=out_offsets[1:])

        out = np.empty(int(out_offsets[-1]), dtype=np.int64)
        if len(flat_list):
            shift = np.repeat(out_offsets[:-1] + int(add_bos) - offsets[:-1], lens)
            out[np.arange(len(ids), dtype=np.int64) + shift] = ids
        if add_bos:
            out[out_offsets[:-1]] = self.bos
        if add_eos:
            out[out_offsets[1:] - 1] = self.eos
        return out, out_offsets

    def _format_output(
        self, token_ids: List[int], return_as: str
    ) -> Union[List[int], np.ndarray, torch.LongTensor]:
        """Converts a list of token IDs to the specified format."""
        if return_as == "py":
            return token_ids
        elif return_as == "np":
            return np.array(token_ids, dtype=np.int64)
        elif return_as == "pt":
            return torch.LongTensor(token_ids)
        else:
            raise ValueError("`return_as` only supports {'py', 'np', 'pt'}")

    def _encode_locseq_impl(
        self, loc_seq: Union[Iterable[str], LocSeq, pa.ListScalar]
    ) -> List[int]:
        """Internal implementation for tokenizing a location sequence.

        This method handles different input collection types and performs the core
        token-to-ID conversion by calling `token_to_id` for each token.
        """
        if isinstance(loc_seq, pa.ListScalar):
            loc_seq = loc_seq.as_py()
        elif isinstance(loc_seq, LocSeq):
            loc_seq = loc_seq.seq
        return [self.token_to_id(loc) for loc in loc_seq]

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)


class AbstractLocSeqTokenizer(AbstractTokenizer):
    """Abstract base class for tokenizers that work directly with sequences of
    pre-defined location strings (e.g., `LocSeq`).

    This class serves as a semantic marker for a group of tokenizers that
    share the same input type. It inherits all its functionality from
    `AbstractTokenizer`.
    """


class AbstractTrajTokenizer(AbstractTokenizer):
    """Abstract base class for tokenizers that process raw GPS trajectories.

    These tokenizers are responsible for the entire pipeline, from converting
    continuous coordinate data (`Trajectory`) into a discrete sequence of
    location strings, and then to token IDs.
    """

    @abstractmethod
    def encode_traj(
        self,
        traj: Union[Trajectory, TrajectoryView],
        add_start_end_token: bool = False,
        return_as: str = "py",
    ) -> Union[List[int], np.ndarray, torch.LongTensor]:
        """Tokenizes a raw trajectory into a sequence of IDs.

        This is the primary interface for this type of tokenizer.

        Parameters
        ----------
        traj : Union[Trajectory, TrajectoryView]
            The trajectory object containing GPS points.
        add_start_end_token : bool, optional
            Whether to add BOS and EOS tokens to the sequence, by default False.
        return_as : str, optional
            The desired output format ('py', 'np', 'pt'), by default 'py'.

        Returns
        -------
        Union[List[int], np.ndarray, torch.LongTensor]
            The resulting sequence of token IDs.
        """
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover
