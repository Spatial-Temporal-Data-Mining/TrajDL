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


import pyarrow as pa

from ... import CoordinateSystem
from ...common.enum import ArrowColName, SeqType
from .abstract import BaseArrowDataset, check_columns_exist
from .schema import TRAJ_ARROW_SCHEMA

# Type ids of Arrow floating-point types accepted as coordinate values
# (the canonical schema uses LargeList<FixedSizeList<Float64, 2>>).
_COORD_TYPE_IDS = {pa.float32().id, pa.float64().id}


class TrajectoryDataset(BaseArrowDataset):
    """
    Dataset for trajectories.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the TrajectoryDataset.

        Parameters
        ----------
        *args : tuple
            Variable length argument list for the superclass initializer.
        **kwargs : dict
            Arbitrary keyword arguments for the superclass initializer.
        """
        super().__init__(*args, **kwargs)

    def check_table(self, table: pa.Table) -> None:
        """
        Validate the input Arrow table.

        Enforces the table contract: the 4 core columns must exist and the
        geometry column (the ``seq`` column holding lists
        of ``(x, y)`` coordinate pairs) must be present with a coordinate
        layout. Which coordinate system the values are expressed in
        (WGS84 / Web Mercator) is carried by the dataset's ``coord_system``
        attribute, not by the Arrow type, so it is not checkable here.

        Parameters
        ----------
        table : pa.Table
            The Arrow table to validate.

        Raises
        ------
        ValueError
            If a required column is missing or the geometry column is absent
            (i.e. ``seq`` does not carry a list of 2D float coordinate
            pairs); the message names the offending column(s).
        """
        check_columns_exist(table, [col.value for col in ArrowColName])
        seq_type = table.schema.field(ArrowColName.SEQ.value).type
        has_geometry = (
            pa.types.is_list(seq_type) or pa.types.is_large_list(seq_type)
        ) and pa.types.is_fixed_size_list(seq_type.value_type)
        if has_geometry:
            coord_type = seq_type.value_type
            has_geometry = (
                coord_type.list_size == 2
                and coord_type.value_type.id in _COORD_TYPE_IDS
            )
        if not has_geometry:
            raise ValueError(
                f"Column '{ArrowColName.SEQ.value}' should be the geometry "
                "column: a list of 2D coordinate pairs "
                f"(FixedSizeList<float, 2>), got {seq_type}."
            )

    @classmethod
    def schema(cls) -> pa.lib.Schema:
        """
        Get the Arrow schema for the trajectory dataset.

        Returns
        -------
        pa.lib.Schema
            The schema definition for the dataset, defined in TRAJ_ARROW_SCHEMA.
        """
        return TRAJ_ARROW_SCHEMA

    @classmethod
    def seq_type(cls) -> SeqType:
        return SeqType.TRAJECTORY

    def __repr__(self) -> str:
        """
        Return a string representation of the TrajectoryDataset.

        Returns
        -------
        str
            A string indicating the size of the dataset.
        """
        return f"TrajectoryDataset(size={self.__len__()})"  # pragma: no cover

    def get_int_entitiy_id(self, idx: int) -> int:
        """
        Get a int entity id by row index.

        Parameters
        ----------
        idx: int
            Row idx

        Returns
        ----------
        int
            entity id

        """
        return int(self.entity_id[idx].as_py())

    def to_web_mercator(self) -> "TrajectoryDataset":
        """
        Convert all trajectories to Web Mercator coordinate system.

        Returns
        -------
        TrajectoryDataset
            A new TrajectoryDataset with all trajectories converted to Web Mercator.

        Notes
        -----
        This operation creates a copy of the entire core_table, which can be
        expensive for large datasets. If the dataset is already in Web Mercator
        coordinate system, a new dataset with the same data is returned.

        Warnings
        --------
        This method does NOT convert feature_table. If the feature_table contains
        coordinate-dependent features, they may become inconsistent after conversion.
        """
        # TODO, 优化性能
        if self._coord_system == CoordinateSystem.WebMercator:
            return TrajectoryDataset(
                table=self.core_table,
                unsafe=True,
                coord_system=CoordinateSystem.WebMercator,
            )

        converted_trajs = (view.to_web_mercator() for view in self.iter_as_seqs())
        return TrajectoryDataset.init_from_seqs(converted_trajs)

    def to_wgs84(self) -> "TrajectoryDataset":
        """
        Convert all trajectories to WGS84 coordinate system.

        Returns
        -------
        TrajectoryDataset
            A new TrajectoryDataset with all trajectories converted to WGS84.

        Notes
        -----
        This operation creates a copy of the entire core_table, which can be
        expensive for large datasets. If the dataset is already in WGS84
        coordinate system, a new dataset with the same data is returned.

        Warnings
        --------
        This method does NOT convert feature_table. If the feature_table contains
        coordinate-dependent features, they may become inconsistent after conversion.
        """
        # TODO, 优化性能
        if self._coord_system == CoordinateSystem.Wgs84:
            return TrajectoryDataset(
                table=self.core_table,
                unsafe=True,
                coord_system=CoordinateSystem.Wgs84,
            )

        converted_trajs = (view.to_wgs84() for view in self.iter_as_seqs())
        return TrajectoryDataset.init_from_seqs(converted_trajs)
