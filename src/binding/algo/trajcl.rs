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

//! Python bindings for TrajCL algorithm functions.

use pyo3::{Bound, PyAny, pyfunction};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::{
    binding::seq::types::PyTrajectoryType, err::TrajDLError, sequence::traits::PointSequence,
};

/// Generate per-point local spatial features along the trajectory
///
/// For each point, 4 features are generated:
/// 1. norm_x: normalized x coordinate
/// 2. norm_y: normalized y coordinate
/// 3. local_dist: local average movement distance (average of the distances to the neighboring points / reference length)
/// 4. heading_change: normalized heading change angle, typically in [-1, 0]
///    - straight ahead: -1
///    - 90-degree turn: -0.5
///    - 180-degree turn (U-turn): 0
///
/// The local_dist and heading_change of the first and last points are set to 0.
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Input trajectory (Web Mercator coordinate system)
/// x_min: float
///     Boundary x minimum
/// x_max: float
///     Boundary x maximum
/// y_min: float
///     Boundary y minimum
/// y_max: float
///     Boundary y maximum
/// ref_length: float
///     Reference length used for distance normalization (usually the grid diagonal length)
///
/// Returns
/// ----------
/// List[List[float]]
///     Feature list of shape [N, 4], 4 features per point
#[gen_stub_pyfunction]
#[pyfunction]
pub fn generate_spatial_features(
    seq: &Bound<'_, PyAny>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    ref_length: f64,
) -> Result<Vec<Vec<f64>>, TrajDLError> {
    let seq = PyTrajectoryType::try_from(seq)?;
    let coords: Vec<[f64; 2]> = seq.to_vec();

    Ok(crate::algo::trajcl::generate_spatial_features_logic(
        &coords, x_min, x_max, y_min, y_max, ref_length,
    ))
}
