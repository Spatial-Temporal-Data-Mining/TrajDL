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

use std::collections::HashSet;

use pyo3::{Bound, PyAny, pyfunction, types::PyList};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::{
    binding::{
        seq::{container::PyTrajectory, types::PyTrajectoryType},
        spatial::grid::PySimpleGridSystem,
    },
    err::TrajDLError,
    sequence::{point::container::Trajectory, traits::PointSequence},
    spatial::coord::{Coordinate, CoordinateSystem},
};

/// Collect the set of token symbols visited by all trajectories in splits (used by the TrajCL tokenizer's fit)
///
/// Unlike count_locations_for_t2vec, this only collects the symbols that occurred and does not count
/// frequencies, because TrajCL applies no frequency filtering and only drops grids never visited by any trajectory.
///
/// Note: this function assumes all points of the input trajectories lie inside gps_boundary (consistent with
/// the original TrajCL paper, where out-of-bound trajectories are filtered during preprocessing).
///
/// Parameters
/// ----------
/// seqs: List[Trajectory]
///     Trajectory list (WGS84 or Web Mercator coordinate system)
/// web_grid: SimpleGridSystem
///     Grid system in the Web Mercator coordinate system
///
/// Returns
/// ----------
/// List[str]
///     List of all token symbols that occurred (unordered, no duplicates)
#[gen_stub_pyfunction]
#[pyfunction]
pub fn collect_token_symbols_for_trajcl(
    seqs: &Bound<'_, PyList>,
    web_grid: &PySimpleGridSystem,
) -> Result<Vec<String>, TrajDLError> {
    let mut symbols = HashSet::<String>::new();

    for seq in seqs {
        let seq = PyTrajectoryType::try_from(&seq)?;
        let coord_system = seq.coord_system();

        for coord in seq.iter_coords() {
            let web_coord = match coord_system {
                CoordinateSystem::Wgs84 => coord.to_web_mercator(),
                CoordinateSystem::WebMercator => coord,
            };

            let symbol = web_grid.coord_to_grid_symbol(&web_coord);
            symbols.insert(symbol);
        }
    }

    Ok(symbols.into_iter().collect())
}

/// Convert a trajectory into a token-symbol sequence with adjacent duplicates removed, plus the corresponding web mercator coordinate trajectory
///
/// This function performs the following steps:
/// 1. Convert each coordinate point to Web Mercator, then map it to a grid token symbol
/// 2. Map points outside the boundary to unk_loc
/// 3. Filter consecutive duplicate token symbols (adjacent deduplication), keeping the coordinate of the first point of each run
/// 4. Return the deduplicated token-symbol sequence and the Trajectory built from the corresponding web mercator coordinates
///
/// Note: ideally all points of the input trajectory lie inside gps_boundary (they should have been
/// filtered during preprocessing). Any out-of-bound point is mapped to unk_loc.
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Input trajectory
/// web_grid: SimpleGridSystem
///     Grid system in the Web Mercator coordinate system
/// unk_loc: str
///     Token symbol used for points outside the boundary
///
/// Returns
/// ----------
/// Tuple[List[str], Trajectory]
///     - The deduplicated token symbol list
///     - The Trajectory built from the corresponding web mercator coordinates, of the same length as the token symbols
#[gen_stub_pyfunction]
#[pyfunction]
pub fn convert_traj_to_locseq_with_coords(
    seq: &Bound<'_, PyAny>,
    web_grid: &PySimpleGridSystem,
    unk_loc: &str,
) -> Result<(Vec<String>, PyTrajectory), TrajDLError> {
    let traj = PyTrajectoryType::try_from(seq)?;

    let locate = |coord: Coordinate| {
        let web_coord = coord.to_web_mercator();
        if web_grid.in_boundary(&web_coord) {
            web_grid.coord_to_grid_symbol(&web_coord)
        } else {
            unk_loc.to_string()
        }
    };

    let mut symbols: Vec<String> = Vec::with_capacity(traj.len());
    let mut coords: Vec<[f64; 2]> = Vec::with_capacity(traj.len());

    let mut last_loc: Option<String> = None;
    for coord in traj.iter_coords() {
        let loc = locate(coord);
        if last_loc.as_ref() != Some(&loc) {
            symbols.push(loc.clone());
            coords.push(coord.to_pair());
            last_loc = Some(loc);
        }
    }

    Ok((
        symbols,
        PyTrajectory(Trajectory::from_vec(coords, CoordinateSystem::WebMercator)),
    ))
}
