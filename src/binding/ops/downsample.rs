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

use pyo3::{Bound, PyAny, PyResult, pyfunction};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::{
    binding::seq::{container::PyTrajectory, types::PyTrajectoryType},
    ops,
};

/// Downsample a trajectory
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Trajectory data
///
/// rate: float
///     Downsample probability, i.e. the probability that each point is removed
///
/// keep_endpoints: bool, optional
///     Whether to keep the first and last points, default True
///
/// allow_empty: bool, optional
///     Whether the generated sequence is allowed to have length 0, default False
///
/// Returns
/// ----------
/// Trajectory
///     The downsampled trajectory data
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (seq, rate, keep_endpoints=true, allow_empty=false))]
pub fn downsampling(
    seq: &Bound<'_, PyAny>,
    rate: f64,
    keep_endpoints: bool,
    allow_empty: bool,
) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::downsampling(
        &seq,
        rate,
        keep_endpoints,
        allow_empty,
    )?))
}

/// Split the sequence by odd/even indices, then downsample each half separately
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Trajectory data
///
/// rate: float
///     Downsample probability, i.e. the probability that each point is removed
///
/// Returns
/// ----------
/// Tuple[Trajectory, Trajectory]
///     The two trajectories after splitting and downsampling
#[gen_stub_pyfunction]
#[pyfunction]
pub fn downsample_split_traj(
    seq: &Bound<'_, PyAny>,
    rate: f64,
) -> PyResult<(PyTrajectory, PyTrajectory)> {
    let seq = PyTrajectoryType::try_from(seq)?;
    let (traj1, traj2) = ops::downsample_split_traj(&seq, rate)?;
    Ok((PyTrajectory(traj1), PyTrajectory(traj2)))
}

/// Randomly downsample a trajectory to a fixed number of points
///
/// Equivalent to ``np.random.choice(N, ceil(N * (1 - rate)), replace=False)`` in the original TrajCL evaluation code:
/// keeps a fixed number of points and returns the indices in ascending order.
///
/// Unlike :func:`downsampling`, the number of kept points here is deterministic (``ceil(N * (1 - rate))``),
/// while :func:`downsampling` performs per-point independent Bernoulli sampling and the kept count is random.
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Trajectory data
///
/// rate: float
///     Proportion of points to remove, in [0, 1].
///
/// Returns
/// ----------
/// Trajectory
///     The trajectory obtained by fixed-size random downsampling
///
/// Note
/// ----
/// Uses the shared seedable RNG: call :func:`seed_augmentation_rng` first to make the output reproducible
/// (test-set generation for evaluation must be seeded first).
#[gen_stub_pyfunction]
#[pyfunction]
pub fn random_subsample(seq: &Bound<'_, PyAny>, rate: f64) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::random_subsample(&seq, rate)?))
}
