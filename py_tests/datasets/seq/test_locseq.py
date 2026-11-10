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

import re

import pytest
from trajdl import LocSeq


class TestLocSeq:
    @pytest.mark.parametrize("invalid_seq", [([1, 2, 3]), (["a", 2])])
    def test_from_list_rejects_non_string_locations(self, invalid_seq):
        """Ensure all elements in `seq` must be strings."""
        pattern = "'int' object cannot be converted to 'PyString'"
        with pytest.raises(TypeError, match=re.escape(pattern)):
            LocSeq.from_list(invalid_seq)

    def test_from_list_accepts_valid_strings(self):
        """Valid string list should create a proper sequence."""
        seq = LocSeq.from_list(["1", "b", "c"])
        assert len(seq) == 3
        assert seq[0] == "1"
        assert seq[-1] == "c"
        assert seq.o == "1"
        assert seq.d == "c"

    @pytest.mark.parametrize("loc_seq", [(["X", "Y"]), (["X", None])])
    def test_from_list_iterates_as_strings(self, loc_seq):
        """Sequence should support iteration and yield strings or None."""
        seq = LocSeq.from_list(loc_seq)
        values = list(seq)
        assert values == loc_seq
        for loc in values:
            assert isinstance(loc, str) or loc is None

    @pytest.mark.parametrize(
        "loc_seq, entity_id", [(["a", "b"], 1), ([None, None], -1)]
    )
    def test_from_list_with_entity_id_type_check(self, loc_seq, entity_id):
        """entity_id must be string or None."""
        pattern = re.escape(
            "argument 'entity_id': 'int' object cannot be converted to 'PyString'"
        )
        with pytest.raises(TypeError, match=pattern):
            LocSeq.from_list(loc_seq, entity_id=entity_id)

    def test_entity_id_setter_rejects_invalid_type(self):
        """Setting entity_id to non-str should raise TypeError."""
        seq = LocSeq.from_list(["P1"])
        pattern = re.escape(
            "argument 'entity_id': 'int' object cannot be converted to 'PyString'"
        )
        with pytest.raises(TypeError, match=pattern):
            seq.entity_id = 1

    def test_empty_sequence_is_allowed(self):
        """Empty list should produce empty sequence."""
        seq = LocSeq.from_list([])
        assert len(seq) == 0
        assert list(seq) == []
        assert seq.o is None
        assert seq.d is None

    def test_string_repr_non_empty(self):
        """Check readable representation of non-empty LocSeq."""
        seq = LocSeq.from_list(
            ["A", "B", "C", "D"], entity_id="test", ts_seq=[1, 2, 3, 4]
        )
        seq.entity_id = "test2"  # trigger change
        repr_str = repr(seq)
        assert "size=4" in repr_str
        assert "entity_id='test2'" in repr_str
        assert "loc_seq='A', 'B', 'C', ..." in repr_str
        assert "ts_seq=1, 2, 3, ..." in repr_str

    def test_string_repr_empty(self):
        """Representation of empty sequence should still be valid."""
        seq = LocSeq.from_list([])
        repr_str = repr(seq)
        assert "size=0" in repr_str


class TestLocSeqTsSeqValidation:

    def test_ts_seq_must_be_integers(self):
        """ts_seq elements must be integers (not floats)."""
        pattern = re.escape(
            "argument 'ts_seq': 'float' object cannot be interpreted as an integer"
        )
        with pytest.raises(TypeError, match=pattern):
            LocSeq.from_list(["1"], ts_seq=[1.2])

    def test_ts_seq_length_must_match_seq(self):
        """ts_seq must have same length as location sequence."""
        pattern = re.escape("InvalidParams: 'seq' and 'ts_seq' must have same length")
        with pytest.raises(ValueError, match=pattern):
            LocSeq.from_list(["1"], ts_seq=[1, 2])

    def test_ts_seq_must_be_monotonically_increasing(self):
        """ts_seq must be strictly increasing."""
        pattern = re.escape(
            "InvalidParams: 'ts_seq' should be a monotonically increasing List[int]."
        )
        with pytest.raises(ValueError, match=pattern):
            LocSeq.from_list(["1", "2"], ts_seq=[2, 1])
