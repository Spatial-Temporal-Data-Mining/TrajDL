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
from typing import Iterable

import pyarrow as pa

from ...common.enum import ArrowColName
from ..base import LocSeq
from .abstract import BaseArrowDataset
from .schema import LOC_SEQ_ARROW_SCHEMA


class LocSeqDataset(BaseArrowDataset):
    """
    Dataset for location sequences.
    """

    def check_table(self, table: pa.Table) -> None:
        """
        Validate the input Arrow table.

        Parameters
        ----------
        table : pa.Table
            The Arrow table to validate.

        Notes
        -----
        This method is a placeholder and needs to be implemented
        to ensure the table complies with expected structure and data types.
        """
        # TODO: Implement validation logic

    @classmethod
    def schema(cls) -> pa.lib.Schema:
        """
        Get the Arrow schema for the location sequence dataset.

        Returns
        -------
        pa.lib.Schema
            The schema definition for the dataset, defined in LOC_SEQ_ARROW_SCHEMA.
        """
        return LOC_SEQ_ARROW_SCHEMA

    @classmethod
    def init_from_loc_seqs(cls, seqs: Iterable[LocSeq]) -> "LocSeqDataset":
        """
        Initialize the dataset from an iterable of location sequences.

        Parameters
        ----------
        seqs : Iterable[LocSeq]
            An iterable containing LocSeq objects.

        Returns
        -------
        LocSeqDataset
            An instance of LocSeqDataset initialized from the provided location sequences.

        Notes
        -----
        The method extracts the 'data', 'id', and 'ts' attributes from each LocSeq object
        to build an Arrow table.
        """

        func = {
            ArrowColName.SEQ.value: lambda seq: seq.seq,
            ArrowColName.ENTITY_ID.value: lambda seq: seq.entity_id,
            ArrowColName.TS_SEQ.value: lambda seq: seq.ts_seq,
            ArrowColName.TS_DELTA.value: lambda seq: seq.ts_delta,
            ArrowColName.DIS_DELTA.value: lambda seq: seq.dis_delta,
            ArrowColName.START_TS.value: lambda seq: seq.start_ts,
        }

        data = defaultdict(list)
        for seq in seqs:
            for col_name, transform_func in func.items():
                data[col_name].append(transform_func(seq))

        arrow_table = pa.Table.from_pydict(data, schema=cls.schema())
        return LocSeqDataset(table=arrow_table)

    def sub_classes_construction(self, *args, **kwargs) -> LocSeq:
        return LocSeq(*args, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the LocSeqDataset.

        Returns
        -------
        str
            A string indicating the size of the dataset.
        """
        return f"LocSeqDataset(size={len(self)})"  # pragma: no cover
