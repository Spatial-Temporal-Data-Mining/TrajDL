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

#[gen_stub_pyfunction]
#[pyfunction]
pub fn distort(seq: &Bound<'_, PyAny>, rate: f64, radius: f64) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::distort(&seq, rate, radius)?))
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn batch_downsample_and_distort(
    seq: &Bound<'_, PyAny>,
    downsample_rate_list: Vec<f64>,
    keep_endpoints: bool,
    allow_empty: bool,
    distort_rate_list: Vec<f64>,
    distort_radius: f64,
) -> PyResult<Vec<PyTrajectory>> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(ops::batch_downsample_and_distort(
        &seq,
        &downsample_rate_list,
        keep_endpoints,
        allow_empty,
        &distort_rate_list,
        distort_radius,
    )?
    .into_iter()
    .map(|traj| PyTrajectory(traj))
    .collect())
}

/// Split the sequence by odd/even indices, then perturb each half separately
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Trajectory data
///
/// rate: float
///     Probability of perturbation
///
/// radius: float
///     Radius of perturbation
///
/// Returns
/// ----------
/// Tuple[Trajectory, Trajectory]
///     The two trajectories after splitting and perturbation
#[gen_stub_pyfunction]
#[pyfunction]
pub fn distort_split_traj(
    seq: &Bound<'_, PyAny>,
    rate: f64,
    radius: f64,
) -> PyResult<(PyTrajectory, PyTrajectory)> {
    let seq = PyTrajectoryType::try_from(seq)?;
    let (traj1, traj2) = ops::distort_split_traj(&seq, rate, radius)?;
    Ok((PyTrajectory(traj1), PyTrajectory(traj2)))
}

/// Add truncated Gaussian noise to the points of a trajectory
///
/// Points are randomly perturbed using a truncated Gaussian distribution.
/// Matches the shift augmentation behavior of TrajCL and can also be used for test-data generation.
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Input trajectory
///
/// mu: float, optional
///     Mean of the Gaussian distribution, defaults to 0.0
///
/// sigma: float, optional
///     Standard deviation of the Gaussian distribution, defaults to 0.5
///
/// factor: float, optional
///     Scaling factor for the random values, defaults to 100.0
///
/// bound: float, optional
///     Absolute bound for the random values, defaults to 100.0
///
/// rate: float, optional
///     Probability that each point is perturbed, defaults to 1.0 (all points are perturbed)
///
/// Returns
/// ----------
/// Trajectory
///     The perturbed trajectory
///
/// Note
/// ----
/// Perturbation is performed in the Web Mercator coordinate system (meters).
/// The truncated Gaussian keeps the perturbation magnitude within [-bound, bound].
/// When rate=1.0 all points are perturbed; when rate<1.0 each point is perturbed with probability rate.
///
/// Uses the shared seedable RNG: call :func:`seed_augmentation_rng` first to make the output reproducible
/// (test-set generation for evaluation must be seeded first).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (seq, mu=0.0, sigma=0.5, factor=100.0, bound=100.0, rate=1.0))]
pub fn truncated_gaussian_distort(
    seq: &Bound<'_, PyAny>,
    mu: f64,
    sigma: f64,
    factor: f64,
    bound: f64,
    rate: f64,
) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::truncated_gaussian_distort(
        &seq, mu, sigma, factor, bound, rate,
    )?))
}

/// Add per-axis independent Gaussian noise (untruncated) to the points of a trajectory
///
/// Unlike `distort`, this function adds noise using per-axis independent Gaussian
/// distributions rather than a uniform direction with a fixed distance. Unlike
/// `truncated_gaussian_distort`, this function does not truncate the noise.
///
/// Matches the perturbation behavior of the TrjSR (Trajectory Super-Resolution) paper.
///
/// Parameters
/// ----------
/// seq: Trajectory
///     Input trajectory
///
/// rate: float
///     Probability that each point is perturbed, in [0, 1]
///
/// sigma: float
///     Standard deviation of the Gaussian noise (meters). The TrjSR paper uses 100.0
///
/// Returns
/// ----------
/// Trajectory
///     The perturbed trajectory
///
/// Note
/// ----
/// Perturbation is performed in the Web Mercator coordinate system (meters).
/// Each coordinate axis samples independently from N(0, sigma^2), without truncation.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn gaussian_distort(seq: &Bound<'_, PyAny>, rate: f64, sigma: f64) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::gaussian_distort(&seq, rate, sigma)?))
}
