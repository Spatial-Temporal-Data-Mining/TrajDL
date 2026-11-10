# Copyright 2026 All authors of TrajDL
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

"""Contract tests for ``BaseSeq`` after the Q3 dead-parameter removal.

Agenda-0.3.0 #5 (wave-5 audit Q3): ``ts_delta`` / ``dis_delta`` were dead
constructor configuration -- stored and exposed but consumed by nothing.
The concrete classes users actually hold (Rust ``LocSeq`` / ``Trajectory``)
never carried those attributes (``hasattr(trajdl.LocSeq.from_list([...]),
"ts_delta") is False``) and interval features ship through the Arrow
``feature_table`` instead, so the parameters were removed rather than
wired.  These tests pin the surviving 4-parameter surface and the loud
rejection of the removed kwargs.
"""

import inspect

import pytest
from trajdl.datasets import BaseSeq


class _Seq(BaseSeq):
    """Minimal concrete subclass: BaseSeq only requires ``check_seq``."""

    @staticmethod
    def check_seq(seqs):
        if not isinstance(seqs, list):
            raise ValueError("`seqs` must be a list")
        return seqs


def test_removed_params_absent_from_signature():
    """``ts_delta`` / ``dis_delta`` no longer exist on the constructor."""
    params = inspect.signature(BaseSeq.__init__).parameters
    assert "ts_delta" not in params
    assert "dis_delta" not in params
    # surviving positional/keyword contract (agenda #5 API-face note: the
    # removal shifts ``start_ts`` from slot 6 to slot 4)
    assert list(params) == ["self", "seq", "entity_id", "ts_seq", "start_ts"]


@pytest.mark.parametrize("dead_kwarg", ["ts_delta", "dis_delta"])
def test_removed_kwargs_rejected_with_type_error(dead_kwarg):
    """Callers still passing the removed kwargs fail loudly (TypeError)."""
    with pytest.raises(TypeError, match=dead_kwarg):
        _Seq(["a", "b"], **{dead_kwarg: [1.0, 2.0]})


def test_removed_properties_no_longer_exposed():
    obj = _Seq(["a", "b"])
    assert not hasattr(obj, "ts_delta")
    assert not hasattr(obj, "dis_delta")


def test_surviving_surface_behaviour_unchanged():
    """The 4 surviving parameters keep their store/validate semantics."""
    obj = _Seq(["a", "b"], entity_id="u1", ts_seq=[1, 2], start_ts=100)
    assert obj.seq == ["a", "b"]
    assert obj.entity_id == "u1"
    assert obj.ts_seq == [1, 2]
    assert obj.start_ts == 100
    assert len(obj) == 2
    obj.set_entity_id("u2")
    assert obj.entity_id == "u2"


def test_start_ts_is_fourth_positional():
    """Positional call shape after the removal: (seq, entity_id, ts_seq, start_ts)."""
    obj = _Seq(["a", "b"], "u1", [1, 2], 100)
    assert obj.start_ts == 100


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"entity_id": 1}, "entity_id"),
        ({"ts_seq": ["x", "y"]}, "ts_seq"),
        ({"ts_seq": [1]}, "same length"),
        ({"ts_seq": [2, 1]}, "monotonically increasing"),
    ],
)
def test_surviving_validation_still_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _Seq(["a", "b"], **kwargs)
