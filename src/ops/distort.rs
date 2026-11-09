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

use rand::Rng;

use crate::{
    err::TrajDLError,
    sequence::{point::container::Trajectory, traits::PointSequence},
    spatial::coord::{Coordinate, CoordinateSystem},
};
use traj_dist_rs::traits::AsCoord;

use super::apply_to_split_traj_by_parity;
use super::downsample::downsampling;

// =============================================================================
// Uniform-disk distortion (t2vec)
// =============================================================================

/// Distort a sequence of coordinates using uniform-direction fixed-distance noise.
///
/// Each point is distorted with probability `rate`. When distorted, a random
/// direction is chosen uniformly and the point is shifted by exactly `radius`
/// meters in Web Mercator coordinates.
///
/// This matches the distortion behavior used in the t2vec paper.
///
/// Uses the shared seeded RNG: call `seed_augmentation_rng(seed)` first to
/// make the output reproducible.
pub fn distort<T: PointSequence>(
    seq: &T,
    rate: f64,
    radius: f64,
) -> Result<Trajectory, TrajDLError> {
    let mut rng = super::aug_rng().lock().unwrap();
    let mut gen_rand = || rng.random_range(0.0..1.0);

    let coord_system = seq.coord_system();

    let result = seq.iter_coords().map(|coord| {
        if gen_rand() < rate {
            let web_point = coord.to_web_mercator();
            let xnoise = 2.0 * gen_rand() - 1.0;
            let ynoise = 2.0 * gen_rand() - 1.0;
            let normz = xnoise.hypot(ynoise);
            let xnoise = xnoise * radius / normz;
            let ynoise = ynoise * radius / normz;

            let distorted = Coordinate::WebMercator {
                x: web_point.x() + xnoise,
                y: web_point.y() + ynoise,
            };

            // Convert back to the original coordinate system
            match coord_system {
                CoordinateSystem::Wgs84 => distorted.to_wgs84().to_pair(),
                CoordinateSystem::WebMercator => distorted.to_pair(),
            }
        } else {
            coord.to_pair()
        }
    });

    Ok(Trajectory::from_iter(result, coord_system))
}

// =============================================================================
// Composite: downsample + distort (t2vec)
// =============================================================================

pub fn downsample_and_distort<T: PointSequence>(
    seq: &T,
    downsampling_rate: f64,
    keep_endpoints: bool,
    allow_empty: bool,
    distort_rate: f64,
    distort_radius: f64,
) -> Result<Trajectory, TrajDLError> {
    Ok(distort(
        &downsampling(seq, downsampling_rate, keep_endpoints, allow_empty)?,
        distort_rate,
        distort_radius,
    )?)
}

pub fn batch_downsample_and_distort<T: PointSequence>(
    seq: &T,
    downsample_rate_list: &Vec<f64>,
    keep_endpoints: bool,
    allow_empty: bool,
    distort_rate_list: &Vec<f64>,
    distort_radius: f64,
) -> Result<Vec<Trajectory>, TrajDLError> {
    let mut result = Vec::new();
    for downsampling_rate in downsample_rate_list {
        for distort_rate in distort_rate_list {
            result.push(downsample_and_distort(
                seq,
                *downsampling_rate,
                keep_endpoints,
                allow_empty,
                *distort_rate,
                distort_radius,
            )?);
        }
    }
    Ok(result)
}

/// Split a trajectory by parity and then distort both sub-trajectories
///
/// Parameters
/// ----------
/// seq: &T
///     The input trajectory
/// rate: f64
///     The distortion rate (probability of distorting each point)
/// radius: f64
///     The distortion radius
///
/// Returns
/// -------
/// Result<(Trajectory, Trajectory), TrajDLError>
///     A tuple of two distorted sub-trajectories
pub fn distort_split_traj<T: PointSequence>(
    seq: &T,
    rate: f64,
    radius: f64,
) -> Result<(Trajectory, Trajectory), TrajDLError> {
    apply_to_split_traj_by_parity(seq, |traj| distort(&traj, rate, radius))
}

// =============================================================================
// Truncated Gaussian distortion (TrajCL)
// =============================================================================

/// Distort trajectory points by adding truncated Gaussian noise.
///
/// This function applies a random shift to points in the trajectory,
/// using a truncated Gaussian distribution. This matches the TrajCL `shift`
/// augmentation behavior and can be used for test data generation.
///
/// # Parameters
/// ----------
/// * `seq` - The input trajectory
/// * `mu` - Mean of the Gaussian distribution (default: 0.0)
/// * `sigma` - Standard deviation of the Gaussian distribution (default: 0.5)
/// * `factor` - Factor to multiply the random value by (default: 100.0)
/// * `bound` - Absolute bound for the random value (default: 100.0)
/// * `rate` - Probability of distorting each point, 1.0 means all points are distorted (default: 1.0)
///
/// # Returns
/// -------
/// * `Result<Trajectory, TrajDLError>` - The distorted trajectory
///
/// # Note
/// -----
/// The distortion is applied in Web Mercator coordinates (meters).
/// The truncated Gaussian ensures shifts are within [-bound, bound].
/// When rate=1.0, all points are distorted; when rate<1.0, each point has
/// a probability of being distorted.
///
/// Uses the shared seeded RNG: call `seed_augmentation_rng(seed)` first to
/// make the output reproducible (required for evaluation testset generation).
pub fn truncated_gaussian_distort<T: PointSequence>(
    seq: &T,
    mu: f64,
    sigma: f64,
    factor: f64,
    bound: f64,
    rate: f64,
) -> Result<Trajectory, TrajDLError> {
    use rand_distr::{Distribution, Normal};

    if !(0.0..=1.0).contains(&rate) {
        return Err(TrajDLError::InvalidParams(
            "rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Uses the shared seeded RNG so that seeding via `seed_augmentation_rng`
    // makes both training-time shift augmentation and evaluation testset
    // generation reproducible.
    let mut rng = super::aug_rng().lock().unwrap();
    let normal = Normal::new(mu, sigma).map_err(|_| {
        TrajDLError::InvalidParams("Invalid sigma for normal distribution".to_string())
    })?;

    // Generate a truncated Gaussian random value
    let truncated_rand = |rng: &mut rand::rngs::StdRng| -> f64 {
        loop {
            let n: f64 = normal.sample(rng);
            let scaled = n * factor;
            if (-bound..=bound).contains(&scaled) {
                return scaled;
            }
        }
    };

    let coord_system = seq.coord_system();

    let result = seq.iter_coords().map(|coord| {
        // Apply distortion with probability `rate`
        if rng.random_range(0.0..1.0) < rate {
            let web_point = coord.to_web_mercator();
            let xnoise = truncated_rand(&mut rng);
            let ynoise = truncated_rand(&mut rng);

            let distorted = Coordinate::WebMercator {
                x: web_point.x() + xnoise,
                y: web_point.y() + ynoise,
            };

            // Convert back to the original coordinate system
            match coord_system {
                CoordinateSystem::Wgs84 => distorted.to_wgs84().to_pair(),
                CoordinateSystem::WebMercator => distorted.to_pair(),
            }
        } else {
            coord.to_pair()
        }
    });

    Ok(Trajectory::from_iter(result, coord_system))
}

// =============================================================================
// Unbounded Gaussian distortion (TrjSR)
// =============================================================================

/// Distort trajectory points by adding axis-independent Gaussian noise (unbounded).
///
/// Unlike [`distort`], which applies uniform-direction fixed-distance noise,
/// and [`truncated_gaussian_distort`], which clips noise to `[-bound, bound]`,
/// this function adds **unclamped, axis-independent** Gaussian noise to each
/// coordinate independently. This matches the distortion behavior used in the
/// TrjSR (Trajectory Super-Resolution) paper.
///
/// The distortion is applied in Web Mercator coordinates (meters).
///
/// # Parameters
/// ----------
/// * `seq` - The input trajectory.
/// * `rate` - Probability of distorting each point (0.0..=1.0).
/// * `sigma` - Standard deviation of the Gaussian noise in meters.
///   The TrjSR paper uses 100.0.
///
/// # Returns
/// -------
/// * `Result<Trajectory, TrajDLError>` - The distorted trajectory.
///
/// Uses the shared seeded RNG: call `seed_augmentation_rng(seed)` first to
/// make the output reproducible.
pub fn gaussian_distort<T: PointSequence>(
    seq: &T,
    rate: f64,
    sigma: f64,
) -> Result<Trajectory, TrajDLError> {
    use rand_distr::{Distribution, Normal};

    if !(0.0..=1.0).contains(&rate) {
        return Err(TrajDLError::InvalidParams(
            "rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    if sigma < 0.0 {
        return Err(TrajDLError::InvalidParams(
            "sigma must be non-negative".to_string(),
        ));
    }

    let mut rng = super::aug_rng().lock().unwrap();
    let normal = Normal::new(0.0, sigma).map_err(|_| {
        TrajDLError::InvalidParams("Invalid sigma for normal distribution".to_string())
    })?;

    let coord_system = seq.coord_system();

    let result = seq.iter_coords().map(|coord| {
        if rng.random_range(0.0..1.0) < rate {
            let web_point = coord.to_web_mercator();
            let xnoise: f64 = normal.sample(&mut *rng);
            let ynoise: f64 = normal.sample(&mut *rng);

            let distorted = Coordinate::WebMercator {
                x: web_point.x() + xnoise,
                y: web_point.y() + ynoise,
            };

            match coord_system {
                CoordinateSystem::Wgs84 => distorted.to_wgs84().to_pair(),
                CoordinateSystem::WebMercator => distorted.to_pair(),
            }
        } else {
            coord.to_pair()
        }
    });

    Ok(Trajectory::from_iter(result, coord_system))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::seed_augmentation_rng;

    /// Deterministic diagonal trajectory near Porto (no external data).
    fn fixed_traj() -> Trajectory {
        let coords: Vec<[f64; 2]> = (0..50)
            .map(|i| {
                let t = i as f64 / 49.0;
                [113.544 + 0.02 * t, 22.184 + 0.02 * t]
            })
            .collect();
        Trajectory::from_vec(coords, CoordinateSystem::Wgs84)
    }

    /// Same seed → `distort` and `gaussian_distort` must be bitwise equal
    /// across two calls.
    #[test]
    fn same_seed_double_call_is_bitwise_equal() {
        let _serial = crate::ops::TEST_AUG_RNG_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let traj = fixed_traj();

        seed_augmentation_rng(42);
        let a = distort(&traj, 0.5, 50.0).unwrap();
        seed_augmentation_rng(42);
        let b = distort(&traj, 0.5, 50.0).unwrap();
        assert_eq!(
            a.to_vec(),
            b.to_vec(),
            "distort not bitwise equal under same seed"
        );

        seed_augmentation_rng(42);
        let g1 = gaussian_distort(&traj, 0.5, 100.0).unwrap();
        seed_augmentation_rng(42);
        let g2 = gaussian_distort(&traj, 0.5, 100.0).unwrap();
        assert_eq!(
            g1.to_vec(),
            g2.to_vec(),
            "gaussian_distort not bitwise equal under same seed"
        );
    }
}
