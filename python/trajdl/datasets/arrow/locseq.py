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

from ...common.enum import ArrowColName, SeqType
from .abstract import BaseArrowDataset, check_columns_exist
from .schema import LOC_SEQ_ARROW_SCHEMA

# Type ids of Arrow string types accepted as the inner type of `seq`
# (the canonical schema uses LargeList<LargeUtf8>; List<Utf8> is accepted too).
_STRING_TYPE_IDS = {pa.string().id, pa.large_string().id}


class LocSeqDataset(BaseArrowDataset):
    """
    Dataset for location sequences.
    """

    def check_table(self, table: pa.Table) -> None:
        """
        Validate the input Arrow table.

        Enforces the table contract: the 4 core columns must exist and the
        ``seq`` column must be a list of strings.

        Parameters
        ----------
        table : pa.Table
            The Arrow table to validate.

        Raises
        ------
        ValueError
            If a required column is missing or the ``seq`` column is not a
            ``list<string>``-shaped column; the message names the offending
            column(s).
        """
        check_columns_exist(table, [col.value for col in ArrowColName])
        seq_type = table.schema.field(ArrowColName.SEQ.value).type
        if not pa.types.is_list(seq_type) and not pa.types.is_large_list(seq_type):
            raise ValueError(
                f"Column '{ArrowColName.SEQ.value}' should be a list of strings, "
                f"got {seq_type}."
            )
        if seq_type.value_type.id not in _STRING_TYPE_IDS:
            raise ValueError(
                f"Column '{ArrowColName.SEQ.value}' should be a list of strings, "
                f"got {seq_type}."
            )

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
    def seq_type(cls) -> SeqType:
        return SeqType.LOC_SEQ

    def __repr__(self) -> str:
        """
        Return a string representation of the LocSeqDataset.

        Returns
        -------
        str
            A string indicating the size of the dataset.
        """
        return f"LocSeqDataset(size={len(self)})"  # pragma: no cover
