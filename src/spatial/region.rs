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

use serde::{Deserialize, Serialize};

use crate::{sequence::traits::PointSequence, spatial::coord::Coordinate};
use traj_dist_rs::traits::AsCoord;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RectangleBoundary {
    pub min_point: Coordinate,
    pub max_point: Coordinate,
}

impl Default for RectangleBoundary {
    fn default() -> Self {
        Self {
            min_point: Default::default(),
            max_point: Default::default(),
        }
    }
}

impl RectangleBoundary {
    pub fn new(min_point: Coordinate, max_point: Coordinate) -> Self {
        if min_point.x() > max_point.x() || min_point.y() > max_point.y() {
            panic!("min_point must not exceed max_point in any dimension")
        }

        RectangleBoundary {
            min_point,
            max_point,
        }
    }

    /// Convert the RectangleBoundary into an [f64; 4]
    ///
    /// Returns
    /// ----------
    /// [f64; 4]
    ///     length 4: min_x, min_y, max_x, max_y
    pub fn to_arr(&self) -> [f64; 4] {
        [
            self.min_point.x(),
            self.min_point.y(),
            self.max_point.x(),
            self.max_point.y(),
        ]
    }

    /// Check whether a point lies inside the region (half-open interval: [min, max))
    ///
    /// Note: points at max_point are treated as "outside"
    pub fn in_boundary<C: AsCoord>(&self, coord: &C) -> bool {
        let x = coord.x();
        let y = coord.y();
        self.min_point.x() <= x
            && x < self.max_point.x()
            && self.min_point.y() <= y
            && y < self.max_point.y()
    }

    /// Check whether a trajectory sequence lies inside the region
    /// For an empty sequence, returns an empty vector.
    pub fn in_boundary_seq<T: PointSequence>(&self, seq: &T) -> Vec<bool> {
        seq.iter_coords()
            .map(|coord| self.in_boundary(&coord))
            .collect()
    }

    pub fn to_web_mercator(&self) -> RectangleBoundary {
        RectangleBoundary {
            min_point: self.min_point.to_web_mercator(),
            max_point: self.max_point.to_web_mercator(),
        }
    }

    pub fn to_wgs84(&self) -> RectangleBoundary {
        RectangleBoundary {
            min_point: self.min_point.to_wgs84(),
            max_point: self.max_point.to_wgs84(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::point::container::Trajectory;
    use crate::spatial::coord::{CoordinateSystem, wgs84};
    use float_cmp::{F64Margin, approx_eq};

    use super::*;

    const fn margin() -> F64Margin {
        F64Margin {
            epsilon: 1e-8,
            ulps: 4,
        }
    }

    macro_rules! assert_coord_approx_eq {
        ($a:expr, $b:expr) => {
            assert!(approx_eq!(f64, $a.x(), $b.x(), margin()));
            assert!(approx_eq!(f64, $a.y(), $b.y(), margin()));
        };
    }

    // 测试构造方法中的异常情况
    #[test]
    #[should_panic(expected = "min_point must not exceed max_point in any dimension")]
    fn test_new_invalid_boundary() {
        let _ = RectangleBoundary::new(wgs84(3.0, 4.0), wgs84(1.0, 2.0));
    }

    // 测试boundary的默认情况
    #[test]
    fn test_default_boundary() {
        let boundary = RectangleBoundary::default();
        assert_eq!(boundary.to_arr(), [0.0, 0.0, 0.0, 0.0]);
    }

    // 测试to_arr方法符合预期
    #[test]
    fn test_to_arr_method() {
        let boundary = RectangleBoundary::new(wgs84(1.0, 2.0), wgs84(3.0, 4.0));
        let x = boundary.to_arr();
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
    }

    // 测试in_boundary方法
    #[test]
    fn test_in_boundary_method() {
        let boundary = RectangleBoundary::new(wgs84(1.0, 2.0), wgs84(3.0, 4.0));

        // 取中间的一个点
        let coord1 = [1.5, 2.5];
        assert!(boundary.in_boundary(&coord1));

        // 最小值的点是要包含在boundary内
        let coord2 = [1.0, 2.0];
        assert!(boundary.in_boundary(&coord2));

        // 最大值的点是不包含在boundary内的
        let coord3 = [3.0, 4.0];
        assert!(!boundary.in_boundary(&coord3));
    }

    // 测试in_boundary_seq方法
    #[test]
    fn test_in_boundary_seq_method() {
        let seq = Trajectory::from_vec(
            vec![[1.0, 2.0], [1.5, 2.5], [3.0, 4.0]],
            CoordinateSystem::Wgs84,
        );
        let boundary = RectangleBoundary::new(wgs84(1.0, 2.0), wgs84(3.0, 4.0));
        assert_eq!(boundary.in_boundary_seq(&seq), vec![true, true, false]);

        let seq = Trajectory::from_vec(
            vec![[0.5, 2.5], [1.5, 2.5], [3.5, 4.5]],
            CoordinateSystem::Wgs84,
        );
        let boundary = RectangleBoundary::new(wgs84(1.0, 2.0), wgs84(3.0, 4.0));
        assert_eq!(boundary.in_boundary_seq(&seq), vec![false, true, false]);
    }

    // 测试to_web_mercator和to_wgs84
    #[test]
    fn test_coordinate_system_conversion_roundtrip() {
        let boundary = RectangleBoundary::new(wgs84(0.0, 0.0), wgs84(90.0, 45.0));
        let wm = boundary.to_web_mercator();
        let expected_min = Coordinate::WebMercator { x: 0.0, y: 0.0 };
        let expected_max = Coordinate::WebMercator {
            x: 10018754.171394622,
            y: 5621521.486192066,
        };

        assert_coord_approx_eq!(wm.min_point, expected_min);
        assert_coord_approx_eq!(wm.max_point, expected_max);

        let reverse = wm.to_wgs84();
        assert_coord_approx_eq!(reverse.min_point, boundary.min_point);
        assert_coord_approx_eq!(reverse.max_point, boundary.max_point);
    }
}
