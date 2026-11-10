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

pub mod algo;
pub mod binding;
pub mod datasets;
pub mod err;
pub mod ops;
pub mod sequence;
pub mod spatial;
pub mod tokenizer;
use pyo3::prelude::*;

use crate::algo::t2vec::convert_coords_to_locseq;
use crate::binding::algo::tale_forest::build_tale_huffman_forest;
use crate::binding::algo::trajcl::generate_spatial_features;
use crate::spatial::coord::{Coordinate, CoordinateSystem, GridCoord};
use crate::tokenizer::locseq::count_locations;
use crate::tokenizer::t2vec::count_locations_for_t2vec;
use crate::tokenizer::trajcl::{
    collect_token_symbols_for_trajcl, convert_traj_to_locseq_with_coords,
};

// TODO(WIREUP): remaining algorithm-specific registrations are added by their
// slices:
// - binding::algo::trjsr::traj_to_image

#[pymodule]
fn trajdl_rust<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_locations_for_t2vec, m)?)?;
    m.add_function(wrap_pyfunction!(crate::binding::algo::t2vec::bucketize, m)?)?;

    m.add_function(wrap_pyfunction!(convert_coords_to_locseq, m)?)?;
    m.add_function(wrap_pyfunction!(count_locations, m)?)?;
    m.add_function(wrap_pyfunction!(collect_token_symbols_for_trajcl, m)?)?;
    m.add_function(wrap_pyfunction!(convert_traj_to_locseq_with_coords, m)?)?;
    m.add_function(wrap_pyfunction!(generate_spatial_features, m)?)?;
    m.add_function(wrap_pyfunction!(build_tale_huffman_forest, m)?)?;

    // Datasets: schema + RecordBatch builders (Rust fast path)
    m.add_function(wrap_pyfunction!(
        crate::binding::datasets::schema::get_traj_schema,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::datasets::schema::get_loc_seq_schema,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::datasets::schema::build_traj_record_batch_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::datasets::schema::build_loc_seq_record_batch_py,
        m
    )?)?;

    // Ops: general
    m.add_function(wrap_pyfunction!(
        crate::binding::ops::split_trajectory_by_parity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::binding::ops::subset_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(crate::binding::ops::mask_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::ops::seed_augmentation_rng,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::binding::ops::simplify::rdp, m)?)?;

    // Ops: distort
    m.add_function(wrap_pyfunction!(crate::binding::ops::distort::distort, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::ops::batch_downsample_and_distort,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::ops::distort_split_traj,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::ops::truncated_gaussian_distort,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::binding::ops::gaussian_distort, m)?)?;

    // Ops: downsample
    m.add_function(wrap_pyfunction!(crate::binding::ops::downsampling, m)?)?;
    m.add_function(wrap_pyfunction!(crate::binding::ops::random_subsample, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::binding::ops::downsample_split_traj,
        m
    )?)?;

    m.add_class::<Coordinate>()?;
    m.add_class::<CoordinateSystem>()?;
    m.add_class::<GridCoord>()?;
    m.add_class::<crate::binding::spatial::region::PyRectangleBoundary>()?;
    m.add_class::<crate::binding::spatial::grid::PySimpleGridSystem>()?;
    m.add_class::<crate::binding::spatial::grid::PyUniformDegreeGridSystem>()?;
    m.add_function(wrap_pyfunction!(crate::spatial::coord::wgs84, m)?)?;

    m.add_class::<crate::sequence::base::SeqType>()?;
    m.add_class::<crate::binding::seq::container::PyLocSeq>()?;
    m.add_class::<crate::binding::seq::container::PyTrajectory>()?;
    m.add_class::<crate::binding::seq::view::PyLocSeqView>()?;
    m.add_class::<crate::binding::seq::view::PyTrajectoryView>()?;

    Ok(())
}
