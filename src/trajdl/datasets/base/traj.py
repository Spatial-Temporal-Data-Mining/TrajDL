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

from typing import Any

import numpy as np

from .abstract import BaseSeq


class Trajectory(BaseSeq):
    """Trajectory representation.

    This class represents a trajectory as a two-dimensional NumPy array,
    where each row corresponds to a point in the trajectory (e.g., (x, y) coordinates).

    Attributes
    ----------
    seq : np.ndarray or List[List[float]]
        The trajectory data as a two-dimensional NumPy array of float64 or a List[List[float]].
    entity_id : Optional[str]
        An optional identifier for the trajectory.
    ts_seq : Optional[List[int]]
        Optional timestamps corresponding to the trajectory points.
    """

    @staticmethod
    def check_seq(seq: Any) -> np.ndarray:
        """Validate and check the trajectory sequence.

        Parameters
        ----------
        seq : Any
            The sequence to be validated, expected to be a NumPy array or a List[List[float]].

        Returns
        -------
        np.ndarray
            The validated trajectory sequence as a two-dimensional NumPy array.

        Raises
        ------
        ValueError
            If seq is not a numpy.ndarray or a list.
            If seq is 1D or does not have exactly 2 columns.
            If the data type of seq is not float64.
        """
        if not isinstance(seq, np.ndarray) and not isinstance(seq, list):
            raise ValueError("`seq` must be a numpy.ndarray or a list")

        if isinstance(seq, list):
            seq = np.array(seq)

        if seq.shape == (0,):
            seq = seq.reshape(0, 2)

        if seq.ndim != 2 or seq.shape[1] != 2:
            raise ValueError(
                f"`seq` must be a 2D ndarray with shape (n, 2), but current shape is {seq.shape}"
            )

        if seq.dtype != np.float64:
            raise ValueError("`seq` must be float64")

        return seq

    def __getitem__(self, idx: int) -> np.ndarray:
        """Retrieve a trajectory point at a specific index.

        Parameters
        ----------
        idx : int
            The index of the trajectory point to retrieve.

        Returns
        -------
        np.ndarray
            The trajectory point at the specified index.
        """
        return self._seq[idx]

    def __repr__(self) -> str:
        """Return a string representation of the Trajectory object.

        Returns
        -------
        str
            A string that represents the Trajectory object.
        """
        return f"Trajectory(entity_id={self.entity_id}, length={self.__len__()})"  # pragma: no cover
