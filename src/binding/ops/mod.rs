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

pub mod distort;
pub mod downsample;
pub mod simplify;

use pyo3::{Bound, PyAny, PyResult, pyfunction};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::{
    binding::seq::{container::PyTrajectory, types::PyTrajectoryType},
    ops,
};

pub use self::distort::{
    batch_downsample_and_distort, distort as distort_fn, distort_split_traj, gaussian_distort,
    truncated_gaussian_distort,
};
pub use self::downsample::{downsample_split_traj, downsampling, random_subsample};

/// Given one trajectory, split the trajectory sequence by odd and even indices
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Trajectory data
///
/// Returns
/// ----------
/// Tuple[Trajectory, Trajectory]
///     The former is the new trajectory of the even-indexed points, the latter that of the odd-indexed points
#[gen_stub_pyfunction]
#[pyfunction]
pub fn split_trajectory_by_parity(
    seq: &Bound<'_, PyAny>,
) -> PyResult<(PyTrajectory, PyTrajectory)> {
    let seq = PyTrajectoryType::try_from(seq)?;
    let (traj_even, traj_odd) = ops::split_trajectory_by_parity(&seq);
    Ok((PyTrajectory(traj_even), PyTrajectory(traj_odd)))
}

// =============================================================================
// TrajCL Data Augmentation Functions
// =============================================================================

/// Set the seed of the global random number generator used by the trajectory
/// augmentation operations (mask_trajectory, subset_trajectory, random_subsample,
/// truncated_gaussian_distort)
///
/// After the call the augmentation sequence is fully deterministic, so multiple training runs can consume
/// exactly the same augmented views (replicating the original TrajCL experiment protocol that seeds the global
/// RNG), and evaluation test-set generation scripts can produce identical output across runs.
///
/// Parameters
/// ----------
/// seed: int
///     Random seed (in the unsigned 64-bit integer range)
///
/// Returns
/// ----------
/// None
#[gen_stub_pyfunction]
#[pyfunction]
pub fn seed_augmentation_rng(seed: u64) {
    crate::ops::seed_augmentation_rng(seed);
}

/// Extract a contiguous sub-trajectory from a trajectory
///
/// Randomly chooses a starting point and extracts a contiguous sub-trajectory of the given ratio.
/// Used for TrajCL data augmentation.
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Input trajectory
///
/// ratio: float
///     Ratio to keep (e.g. 0.7 means 70% of the points are kept)
///
/// Returns
/// ----------
/// Trajectory
///     The extracted contiguous sub-trajectory
#[gen_stub_pyfunction]
#[pyfunction]
pub fn subset_trajectory(seq: &Bound<'_, PyAny>, ratio: f64) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::subset_trajectory(&seq, ratio)?))
}

/// Randomly mask (remove) the given ratio of points in a trajectory
///
/// Unlike downsampling, this function removes exactly the specified ratio of points
/// rather than removing them probabilistically. Matches the mask augmentation behavior of TrajCL.
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Input trajectory
///
/// ratio: float
///     Proportion to remove (e.g. 0.3 means 30% of the points are removed)
///
/// min_length: int, optional
///     Minimum length of the resulting trajectory, defaults to 2
///
/// Returns
/// ----------
/// Trajectory
///     The masked trajectory
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (seq, ratio, min_length=2))]
pub fn mask_trajectory(
    seq: &Bound<'_, PyAny>,
    ratio: f64,
    min_length: usize,
) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::mask_trajectory(&seq, ratio, min_length)?))
}
