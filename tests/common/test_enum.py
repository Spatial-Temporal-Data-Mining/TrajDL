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

from trajdl.common.enum import ArrowColName, BaseEnum, LossEnum, Mode


def test_base_enum():
    invalid_value = "test"
    with pytest.raises(ValueError, match=f"{invalid_value} is not a valid BaseEnum"):
        BaseEnum.from_string(invalid_value)

    invalid_type = [1]
    with pytest.raises(
        ValueError,
        match=re.escape("`value` should be a str or an instance of BaseEnum"),
    ):
        BaseEnum.parse(value=invalid_type)


def test_mode_enum():
    keys = [Mode.PRETRAIN, Mode.TRAIN, Mode.EVAL]
    for key in keys:
        assert Mode.from_string(key.value) == key


def test_loss_enum():
    keys = [LossEnum.MEAN, LossEnum.SUM, LossEnum.NONE]
    for key in keys:
        assert LossEnum.from_string(key.value) == key


def test_arrow_col_enum():
    keys = [
        ArrowColName.SEQ,
        ArrowColName.ENTITY_ID,
        ArrowColName.TS_SEQ,
        ArrowColName.TS_DELTA,
        ArrowColName.DIS_DELTA,
        ArrowColName.START_TS,
    ]
    for key in keys:
        assert ArrowColName.from_string(key.value) == key
