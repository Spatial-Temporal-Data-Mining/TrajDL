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

use std::collections::HashMap;

use pyo3::{Bound, pyfunction, types::PyList};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::{
    binding::{
        seq::types::PyTrajectoryType,
        spatial::{grid::PySimpleGridSystem, region::PyRectangleBoundary},
    },
    err::TrajDLError,
    sequence::traits::PointSequence,
    spatial::coord::Coordinate,
};

/// Given a longitude/latitude sequence, a boundary, and a grid coordinate system, count the occurrences of each location
#[gen_stub_pyfunction]
#[pyfunction]
pub fn count_locations_for_t2vec(
    seqs: &Bound<'_, PyList>,
    web_grid: &PySimpleGridSystem,
    gps_boundary: &PyRectangleBoundary,
) -> Result<HashMap<String, i64>, TrajDLError> {
    // 构造一个闭包，用于将经纬度转换为grid_symbol
    let locate = |coord: Coordinate| {
        let web_point = coord.to_web_mercator();
        web_grid.coord_to_grid_symbol(&web_point)
    };

    // 统计每个轨迹点所在位置的出现频数
    let mut freq = HashMap::<String, i64>::new();

    for seq in seqs {
        let seq = PyTrajectoryType::try_from(&seq)?;
        for (loc, coord) in seq.get_locs(locate) {
            // 只有坐标在区域范围内，才统计它的出现频数
            if gps_boundary.in_boundary(&coord) {
                *freq.entry(loc).or_insert(0) += 1;
            }
        }
    }

    let result = freq
        .into_iter()
        .map(|(loc, count)| (loc.to_string(), count))
        .collect();

    Ok(result)
}
