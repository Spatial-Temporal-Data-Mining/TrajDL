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

from abc import ABC, abstractstaticmethod
from typing import Any, List, Optional, Union


class BaseSeq(ABC):
    """Abstract base class for sequence data representation.

    This class provides a structure to initialize and validate sequences, entity IDs,
    and timestamps.

    Attributes
    ----------
    seq : Any
        The sequence data.
    entity_id : Optional[str]
        The identifier for the entity associated with the sequence.
    ts_seq : Optional[List[int]]
        The optional list of timestamps corresponding to the sequence data.
    ts_delta: Optional[List[float]]
        time interval.
    dis_delta: Optional[List[float]]
        distances.
    start_ts: Optional[int]
        表示序列的起始时间
    """

    def __init__(
        self,
        seq: Any,
        entity_id: Optional[str] = None,
        ts_seq: Optional[List[int]] = None,
        ts_delta: Optional[List[float]] = None,
        dis_delta: Optional[List[float]] = None,
        start_ts: Optional[int] = None,
    ):
        """Initialize a BaseSeq object.

        Parameters
        ----------
        seq : Any
            The sequence data to be checked and stored.
        entity_id : Optional[str], optional
            The ID for the sequence entity (default is None).
        ts_seq : Optional[List[int]], optional
            A list of timestamps corresponding to the sequence (default is None).
        ts_delta: Optional[List[float]], optional
            time interval.
        dis_delta: Optional[List[float]], optional
            distances.
        start_ts: Optional[Union[int]], optional
            The start timestamp of this sequence.

        Raises
        ------
        ValueError
            If entity_id is not a string or None, if ts_seq is not a list of integers,
            if ts_seq and seq have different lengths, or if ts_seq is not monotonically
            increasing.
            If ts_delta and seq have different lengths.
            If dis_delta and seq have different lengths.
        """
        self._seq = self.check_seq(seq)
        if entity_id is not None and not isinstance(entity_id, str):
            raise ValueError("`entity_id` can only be str or None.")

        self._entity_id = entity_id

        if ts_seq is not None and (
            not isinstance(ts_seq, list)
            or any(not isinstance(ts, int) for ts in ts_seq)
        ):
            raise ValueError("`ts_seq` should be a List[int] or None")

        if ts_seq:
            if len(ts_seq) != len(seq):
                raise ValueError("`seq` and `ts_seq` must have same length")
            for i in range(len(ts_seq) - 1):
                if ts_seq[i] > ts_seq[i + 1]:
                    raise ValueError(
                        "`ts_seq` should be a monotonically increasing List[int]."
                    )
        self._ts = ts_seq

        if ts_delta:
            if len(ts_delta) != len(seq):
                raise ValueError("`ts_delta` and `seq` must have same length")
        self._ts_delta = ts_delta

        if dis_delta:
            if len(dis_delta) != len(seq):
                raise ValueError("`dis_delta` and `seq` must have same length")
        self._dis_delta = dis_delta

        self._start_ts = start_ts

    @abstractstaticmethod
    def check_seq(seqs: Any) -> Any:
        """Check if the sequence type is valid.

        Parameters
        ----------
        seqs : Any
            The sequence to be checked.

        Returns
        -------
        Any
            The validated sequence.
        """
        raise ValueError("Subclasses should implement this method.")  # pragma: no cover

    @property
    def seq(self) -> Any:
        """Get the stored sequence data."""
        return self._seq

    @property
    def entity_id(self) -> Optional[str]:
        """Get the entity ID."""
        return self._entity_id

    @property
    def ts_seq(self) -> Optional[List[int]]:
        """Get the timestamps associated with the sequence."""
        return self._ts

    @property
    def ts_delta(self) -> Optional[List[float]]:
        return self._ts_delta

    @property
    def dis_delta(self) -> Optional[List[float]]:
        return self._dis_delta

    @property
    def start_ts(self) -> Optional[Union[int, float]]:
        return self._start_ts

    def __len__(self) -> int:
        """Return the length of the sequence."""
        return len(self._seq)

    def set_entity_id(self, entity_id: Optional[str]) -> None:
        """Set the entity ID.

        Parameters
        ----------
        entity_id : Optional[str]
            The identifier to be set for the entity.

        Raises
        ------
        ValueError
            If entity_id is not a string or None.
        """
        if entity_id is not None and not isinstance(entity_id, str):
            raise ValueError("`entity_id` should be str or None")
        self._entity_id = entity_id
