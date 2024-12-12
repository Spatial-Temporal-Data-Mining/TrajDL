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

import numpy as np
import pytest

from trajdl.datasets.base import LocSeq, Trajectory


def test_locseq():
    with pytest.raises(ValueError, match=re.escape("`seq` must be a List[str]")):
        LocSeq([1, 2, 3])

    with pytest.raises(ValueError, match=re.escape("`seq` must be a List[str]")):
        LocSeq(["1", 2])

    seq = LocSeq(["1", "b", "c"])
    assert len(seq) == 3
    assert seq[0] == "1"
    assert seq[-1] == "c"

    for loc in seq:
        assert isinstance(loc, str)

    assert seq.o == "1"
    assert seq.d == "c"

    with pytest.raises(
        ValueError, match=re.escape("`entity_id` can only be str or None.")
    ):
        LocSeq(["1"], 1)

    with pytest.raises(
        ValueError, match=re.escape("`ts_seq` should be a List[int] or None")
    ):
        LocSeq(["1"], "1", [1.2, 2.3])

    with pytest.raises(
        ValueError, match=re.escape("`seq` and `ts_seq` must have same length")
    ):
        LocSeq(["1"], "1", [1, 2])

    with pytest.raises(
        ValueError,
        match=re.escape("`ts_seq` should be a monotonically increasing List[int]."),
    ):
        LocSeq(["1", "2"], "1", [2, 1])

    with pytest.raises(
        ValueError, match=re.escape("`ts_delta` and `seq` must have same length")
    ):
        LocSeq(["1", "2"], "1", ts_delta=[1.0])

    with pytest.raises(
        ValueError, match=re.escape("`dis_delta` and `seq` must have same length")
    ):
        LocSeq(["1", "2"], "1", dis_delta=[1.0])

    seq2 = LocSeq(["A", "B", "C", "D"], entity_id="test", ts_seq=[1, 2, 3, 4])

    assert list(seq2) == ["A", "B", "C", "D"]
    seq2.set_entity_id("test2")
    assert seq2.entity_id == "test2"

    assert (
        str(seq2)
        == "LocSeq(size=4, entity_id='test2', loc_seq='A', 'B', 'C', ..., ts_seq=1, 2, 3, ...)"
    )

    seq3 = LocSeq([])
    assert list(seq3) == []

    with pytest.raises(
        ValueError, match=re.escape("`entity_id` should be str or None")
    ):
        seq3.set_entity_id(1)


def test_traj():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`seq` must be a 2D ndarray with shape (n, 2), but current shape is (3, 3)"
        ),
    ):
        Trajectory(np.random.random(size=(3, 3)))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`seq` must be a 2D ndarray with shape (n, 2), but current shape is (3,)"
        ),
    ):
        Trajectory([1.0, 2.0, 3.0])

    traj = Trajectory(np.random.random(size=(10, 2)), entity_id="a")
    assert len(traj) == 10
    assert isinstance(traj.seq, np.ndarray)
    assert traj[0].shape == (2,)
    assert traj.entity_id == "a"

    traj = Trajectory(np.random.random(size=(10, 2)).tolist(), entity_id="a")
    assert len(traj) == 10
    assert isinstance(traj.seq, np.ndarray)
    assert traj[0].shape == (2,)
    assert traj.entity_id == "a"

    with pytest.raises(
        ValueError, match=re.escape("`seq` must be a numpy.ndarray or a list")
    ):
        Trajectory(seq=(1, 2, 3))

    with pytest.raises(ValueError, match=re.escape("`seq` must be float64")):
        Trajectory(seq=np.random.random(size=(3, 2)).astype(np.float32))

    assert Trajectory(seq=[]).seq.shape == (0, 2)
