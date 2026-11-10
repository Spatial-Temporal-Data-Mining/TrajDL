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

//! TrajCL algorithm functions (pure Rust logic, no PyO3 dependency).

use std::f64::consts::PI;

/// Generate spatial features for each point in a trajectory.
///
/// For each point, 4 features are produced:
/// 1. **norm_x**: normalized x coordinate within `[x_min, x_max]`
/// 2. **norm_y**: normalized y coordinate within `[y_min, y_max]`
/// 3. **local_dist**: local average movement distance (mean of adjacent
///    segment lengths / `ref_length`)
/// 4. **heading_change**: heading change normalized to roughly `[-1, 0]`
///    - straight ahead: -1
///    - 90° turn: -0.5
///    - U-turn (180°): 0
///
/// The first and last points have `local_dist` and `heading_change` set to 0.
pub fn generate_spatial_features_logic(
    coords: &[[f64; 2]],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    ref_length: f64,
) -> Vec<Vec<f64>> {
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    // Pre-compute adjacent-point distances
    let mut distances: Vec<f64> = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = coords[i + 1][0] - coords[i][0];
        let dy = coords[i + 1][1] - coords[i][1];
        distances.push((dx * dx + dy * dy).sqrt());
    }

    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let x = coords[i][0];
        let y = coords[i][1];

        let norm_x = (x - x_min) / x_range;
        let norm_y = (y - y_min) / y_range;

        let (local_dist, heading_change) = if i == 0 || i == n - 1 {
            (0.0, 0.0)
        } else {
            let dist = (distances[i - 1] + distances[i]) / 2.0 / ref_length;

            // Heading change using the same formula as original TrajCL:
            // radian = π - atan2(dx1, dy1) + atan2(dx2, dy2)
            // Note: original code uses atan2(dx, dy) parameter order (non-standard)
            let angle1 = (coords[i - 1][0] - coords[i][0]).atan2(coords[i - 1][1] - coords[i][1]);
            let angle2 = (coords[i + 1][0] - coords[i][0]).atan2(coords[i + 1][1] - coords[i][1]);
            let radian = PI - angle1 + angle2;

            let heading = 1.0 - radian.abs() / PI;

            (dist, heading)
        };

        features.push(vec![norm_x, norm_y, local_dist, heading_change]);
    }

    features
}
