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

from enum import Enum
from typing import Any, List


class BaseEnum(Enum):
    @classmethod
    def from_string(cls, value: str):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")

    @classmethod
    def parse(cls, value: Any) -> "BaseEnum":
        if isinstance(value, str):
            return cls.from_string(value)
        elif isinstance(value, cls):
            return value
        else:
            raise ValueError(
                f"`value` should be a str or an instance of {cls.__name__}"
            )

    @classmethod
    def values(cls) -> List[Any]:
        return {item.value for item in cls}


class Mode(BaseEnum):
    PRETRAIN = "pretrain"
    TRAIN = "train"
    EVAL = "eval"


class LossEnum(BaseEnum):
    SUM = "sum"
    MEAN = "mean"
    NONE = "none"


class ArrowColName(BaseEnum):
    SEQ = "seq"
    ENTITY_ID = "entity_id"
    TS_SEQ = "ts_seq"
    TS_DELTA = "ts_delta"
    DIS_DELTA = "dis_delta"
    START_TS = "start_ts"


class TokenEnum(BaseEnum):
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN = "[MASK]"


class OpenSourceDatasetEnum(BaseEnum):
    GOWALLA_URL = "GOWALLA_URL"
    PORTO_URL = "PORTO_URL"


class ReturnASEnum(BaseEnum):
    NP = "np"
    PT = "pt"
    PY = "py"
    TRAJ = "traj"
    LOCSEQ = "locseq"
