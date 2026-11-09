// Copyright 2024 All authors of TrajDL
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use trajdl_rust::spatial::{coord::Coordinate, region::RectangleBoundary};

#[test]
fn test_to_arr() {
    let boundary = RectangleBoundary::new(
        Coordinate::Wgs84 { lng: 1.0, lat: 2.0 },
        Coordinate::Wgs84 { lng: 3.0, lat: 4.0 },
    );
    let x = boundary.to_arr();
    assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
}
