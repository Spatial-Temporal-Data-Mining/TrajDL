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

from trajdl.datasets.open_source.conf import GowallaDataset, PortoDataset


def test_original_dataset():
    ds = GowallaDataset().load(return_as="pl")
    assert ds.shape == (6442892, 5)
    assert ds.columns == ["user_id", "check_in_time", "lat", "lng", "loc_id"]

    ds = PortoDataset().load(return_as="pl")
    assert ds.shape == (1710670, 9)
    assert ds.columns == [
        "TRIP_ID",
        "CALL_TYPE",
        "ORIGIN_CALL",
        "ORIGIN_STAND",
        "TAXI_ID",
        "TIMESTAMP",
        "DAY_TYPE",
        "MISSING_DATA",
        "POLYLINE",
    ]
