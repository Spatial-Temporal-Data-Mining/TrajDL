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

from typing import Any, Iterator, List, Union

from .abstract import BaseSeq


class LocSeq(BaseSeq):
    """Location sequence representation.

    Attributes
    ----------
    loc_seq : List[str]
        The list of locations in the sequence.
    entity_id : Union[str, None]
        An optional identifier, which can store the sequence ID or user ID.
    ts_seq : Union[List[int], None]
        A sequence of timestamps (in seconds or milliseconds), defined by the user without constraints.
    """

    @staticmethod
    def check_seq(seq: Any) -> List[str]:
        """Check if the sequence is valid.

        Parameters
        ----------
        seq : Any
            The sequence to be checked.

        Returns
        -------
        List[str]
            The validated location sequence.

        Raises
        ------
        ValueError
            If seq is not a List[str].
        """
        if not isinstance(seq, list) or any(not isinstance(loc, str) for loc in seq):
            raise ValueError("`seq` must be a List[str]")
        return seq

    def __getitem__(self, idx: int) -> str:
        """Get the location at the specified index.

        Parameters
        ----------
        idx : int
            The index of the location in the sequence.

        Returns
        -------
        str
            The location at the given index.
        """
        return self._seq[idx]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the location sequence.

        Yields
        ------
        str
            Each location in the sequence.
        """
        for loc in self._seq:
            yield loc

    @property
    def o(self) -> str:
        """Return the starting location."""
        return self.__getitem__(0)

    @property
    def d(self) -> str:
        """Return the destination location."""
        return self.__getitem__(-1)

    def _loc_expr(self, seq: Union[List[str], List[int]], length: int) -> str:
        """Create a string representation of the sequence for display.

        Parameters
        ----------
        seq : Union[List[str], List[int]]
            The sequence to create a representation for.
        length : int
            The length of the sequence.

        Returns
        -------
        str
            A string representation of the first three elements of the sequence,
            followed by an ellipsis if the length exceeds three.
        """
        seq_example = ", ".join(
            f"'{i}'" if isinstance(i, str) else str(i) for i in seq[:3]
        )
        if length > 3:
            seq_example += ", ..."
        return seq_example

    def __repr__(self) -> str:
        """Return a string representation of the LocSeq object.

        Returns
        -------
        str
            A string that represents the LocSeq object.
        """
        length = self.__len__()
        size_repr = f"size={length}"
        entity_id_expr = f"entity_id='{self.entity_id}'" if self.entity_id else ""
        loc_seq_expr = (
            f"loc_seq={self._loc_expr(self._seq, length)}" if self._seq else ""
        )
        ts_seq_repr = f"ts_seq={self._loc_expr(self._ts, length)}" if self._ts else ""

        expression = ", ".join(
            i for i in [size_repr, entity_id_expr, loc_seq_expr, ts_seq_repr] if i
        )
        return f"LocSeq({expression})"
