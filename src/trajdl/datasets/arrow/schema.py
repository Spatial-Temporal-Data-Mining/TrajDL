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

from ...common.enum import ArrowColName

LOC_SEQ_ARROW_SCHEMA = pa.schema(
    [
        pa.field(ArrowColName.SEQ.value, pa.large_list(pa.large_string())),
        pa.field(ArrowColName.ENTITY_ID.value, pa.large_string()),
        pa.field(ArrowColName.TS_SEQ.value, pa.large_list(pa.int64())),
        pa.field(ArrowColName.TS_DELTA.value, pa.large_list(pa.float32())),
        pa.field(ArrowColName.DIS_DELTA.value, pa.large_list(pa.float32())),
        pa.field(ArrowColName.START_TS.value, pa.int64()),
    ]
)

TRAJ_ARROW_SCHEMA = pa.schema(
    [
        pa.field(ArrowColName.SEQ.value, pa.large_list(pa.list_(pa.float64(), 2))),
        pa.field(ArrowColName.ENTITY_ID.value, pa.large_string()),
        pa.field(ArrowColName.TS_SEQ.value, pa.large_list(pa.int64())),
        pa.field(ArrowColName.TS_DELTA.value, pa.large_list(pa.float32())),
        pa.field(ArrowColName.DIS_DELTA.value, pa.large_list(pa.float32())),
        pa.field(ArrowColName.START_TS.value, pa.int64()),
    ]
)
