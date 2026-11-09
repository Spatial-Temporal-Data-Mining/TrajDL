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
};

use super::apply_to_split_traj_by_parity;

/// Down-samples a trajectory by randomly discarding points.
///
/// # Arguments
///
/// * `seq` - The input trajectory, which must implement `PointSequence`.
/// * `rate` - The probability of discarding each point (a value between 0.0 and 1.0).
/// * `keep_endpoints` - If `true`, the first and last points of the sequence are always preserved.
/// * `allow_empty` - If `true`, the function can return an empty trajectory.
///
/// # Returns
///
/// A `Result` containing the new `Trajectory` or a `TrajDLError`.
///
/// # Errors
///
/// Returns `TrajDLError::InvalidParams` if `allow_empty` is `false` and the input `seq` is empty.
///
/// # Note on Behavior
///
/// If `allow_empty` is `false` and the random sampling results in an empty trajectory,
/// a fallback mechanism is triggered to ensure a non-empty result. This fallback will
/// create a trajectory with the first and last points of the original sequence,
/// effectively ignoring the `keep_endpoints` parameter in this specific scenario.
///
/// # Note
///
/// Uses the shared seeded RNG: call `seed_augmentation_rng(seed)` first to
/// make the output reproducible.
pub fn downsampling<T: PointSequence>(
    seq: &T,
    rate: f64,
    keep_endpoints: bool,
    allow_empty: bool,
) -> Result<Trajectory, TrajDLError> {
    let num_points = seq.len();

    if !allow_empty && num_points == 0 {
        return Err(TrajDLError::InvalidParams(
            "length of seq should not be 0 when allow_empty is set to false".to_string(),
        ));
    }

    // Optimization: If discard rate is 0, or if we must keep endpoints on a tiny trajectory,
    // just return a copy.
    if rate == 0.0 || (keep_endpoints && num_points <= 2) {
        return seq.to_container();
    }

    let mut rng = super::aug_rng().lock().unwrap();
    let mut gen_rand = || rng.random_range(0.0..1.0);

    let iter = seq.iter_coords().enumerate().filter_map(|(idx, coord)| {
        if keep_endpoints && (idx == 0 || idx == num_points - 1) {
            Some(coord.to_pair())
        } else if gen_rand() > rate {
            // Keep the point with a probability of (1.0 - rate)
            Some(coord.to_pair())
        } else {
            // Discard the point
            None
        }
    });

    let result = Trajectory::from_iter(iter, seq.coord_system());

    // Fallback logic: if the result is empty but shouldn't be, create a minimal valid trajectory.
    if !allow_empty && result.len() == 0 && num_points > 0 {
        if num_points == 1 {
            // If the original had only one point, return it.
            return seq.to_container();
        } else {
            // Otherwise, return a trajectory with the start and end points.
            // This prioritizes non-emptiness over the `keep_endpoints` setting.
            let new_traj = vec![seq.get(0), seq.get(seq.len() - 1)];
            Ok(Trajectory::from_vec(new_traj, seq.coord_system()))
        }
    } else {
        Ok(result)
    }
}

/// Split a trajectory by parity and then downsample both sub-trajectories
///
/// Parameters
/// ----------
/// seq: &T
///     The input trajectory
/// rate: f64
///     The downsampling rate (probability of discarding each point)
///
/// Returns
/// -------
/// Result<(Trajectory, Trajectory), TrajDLError>
///     A tuple of two downsampled sub-trajectories
pub fn downsample_split_traj<T: PointSequence>(
    seq: &T,
    rate: f64,
) -> Result<(Trajectory, Trajectory), TrajDLError> {
    apply_to_split_traj_by_parity(seq, |traj: Trajectory| {
        downsampling(&traj, rate, true, false)
    })
}

/// Randomly subsample a fixed number of points from the trajectory.
///
/// Retains exactly `ceil(N * (1 - rate))` points, randomly selected without
/// replacement and returned in ascending index order. This matches the
/// original TrajCL evaluation-side downsampling behavior:
/// `np.random.choice(N, ceil(N * (1 - rate)), replace=False)`.
///
/// This differs from [`downsampling`], which uses independent Bernoulli
/// trials per point (variable retained count) and is used by other algorithms
/// (e.g. t2vec); we therefore introduce a separate function here rather than
/// changing the existing semantics.
///
/// # Parameters
/// ----------
/// * `seq` - The input trajectory.
/// * `rate` - Fraction of points to remove (0.0..=1.0). The number of
///   retained points is `ceil(N * (1 - rate))`, clamped to `[1, N]`.
///
/// # Returns
/// -------
/// * `Result<Trajectory, TrajDLError>` - The subsampled trajectory.
///
/// # Errors
/// ------
/// Returns `TrajDLError::InvalidParams` if `rate` is outside `[0.0, 1.0]`.
/// If the input trajectory has 0 or 1 point, it is returned as-is.
///
/// # Note
/// -----
/// Uses the shared seeded RNG: call `seed_augmentation_rng(seed)` first to
/// make the output reproducible (required for evaluation testset generation).
pub fn random_subsample<T: PointSequence>(seq: &T, rate: f64) -> Result<Trajectory, TrajDLError> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(TrajDLError::InvalidParams(
            "rate must be between 0.0 and 1.0".to_string(),
        ));
    }

    let num_points = seq.len();
    if num_points <= 1 {
        return seq.to_container();
    }

    let preserved_ratio = 1.0 - rate;
    let raw_keep = ((num_points as f64) * preserved_ratio).ceil() as usize;
    let num_keep = raw_keep.clamp(1, num_points);

    if num_keep == num_points {
        return seq.to_container();
    }

    // Partial Fisher-Yates: select `num_keep` distinct indices from [0, num_points).
    // Uses the shared seeded RNG so that seeding via `seed_augmentation_rng`
    // makes evaluation testset generation reproducible.
    let mut indices: Vec<usize> = (0..num_points).collect();
    {
        let mut rng = super::aug_rng().lock().unwrap();
        for i in 0..num_keep {
            let j = rng.random_range(i..num_points);
            indices.swap(i, j);
        }
    }
    let mut selected: Vec<usize> = indices.into_iter().take(num_keep).collect();
    selected.sort_unstable();

    let coords: Vec<[f64; 2]> = selected.into_iter().map(|idx| seq.get(idx)).collect();

    Ok(Trajectory::from_vec(coords, seq.coord_system()))
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
        Trajectory::from_vec(coords, crate::spatial::coord::CoordinateSystem::Wgs84)
    }

    /// Same seed → `downsampling` must be bitwise equal across two calls.
    #[test]
    fn same_seed_double_call_is_bitwise_equal() {
        let _serial = crate::ops::TEST_AUG_RNG_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let traj = fixed_traj();

        seed_augmentation_rng(42);
        let a = downsampling(&traj, 0.3, true, false).unwrap();
        seed_augmentation_rng(42);
        let b = downsampling(&traj, 0.3, true, false).unwrap();
        assert_eq!(
            a.to_vec(),
            b.to_vec(),
            "downsampling not bitwise equal under same seed"
        );
    }
}
