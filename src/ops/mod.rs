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

use itertools::{Either, Itertools};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::{Mutex, OnceLock};

use crate::{
    err::TrajDLError,
    sequence::{point::container::Trajectory, traits::PointSequence},
};

static AUG_RNG: OnceLock<Mutex<StdRng>> = OnceLock::new();

fn aug_rng() -> &'static Mutex<StdRng> {
    AUG_RNG.get_or_init(|| Mutex::new(StdRng::from_os_rng()))
}

/// Seed the shared RNG used by trajectory augmentation ops
/// (`distort`, `downsampling`, `gaussian_distort`, `subset_trajectory`,
/// `mask_trajectory`, `random_subsample`, `truncated_gaussian_distort`).
///
/// Calling this makes the augmentation sequence fully deterministic, so that
/// repeated training runs consume identical augmented views (mirroring the
/// original TrajCL protocol of seeding the global RNG) and evaluation
/// testset generation scripts produce identical outputs across runs.
pub fn seed_augmentation_rng(seed: u64) {
    *aug_rng().lock().unwrap() = StdRng::seed_from_u64(seed);
}

/// Serializes tests that reseed the process-global `AUG_RNG`, so that
/// `cargo test`'s default multithreaded runner cannot interleave another
/// test's op calls between a test's `seed_augmentation_rng` call and its
/// op calls.
#[cfg(test)]
pub(crate) static TEST_AUG_RNG_LOCK: Mutex<()> = Mutex::new(());

pub use distort::{
    batch_downsample_and_distort, distort, distort_split_traj, downsample_and_distort,
    gaussian_distort, truncated_gaussian_distort,
};
pub use downsample::{downsample_split_traj, downsampling, random_subsample};

/// Split the input trajectory into two sub-trajectories by the odd/even parity of the coordinate points' sequence indices.
pub fn split_trajectory_by_parity<T: PointSequence>(seq: &T) -> (Trajectory, Trajectory) {
    // 获取经纬度序列
    let vec = seq.to_vec();

    let (even_vec, odd_vec) = vec.into_iter().enumerate().partition_map(|(idx, point)| {
        if idx % 2 == 0 {
            Either::Left(point)
        } else {
            Either::Right(point)
        }
    });

    (
        Trajectory::from_vec(even_vec, seq.coord_system()),
        Trajectory::from_vec(odd_vec, seq.coord_system()),
    )
}

/// Apply a function on two sub trajectories split by parity
///
/// This function splits the input trajectory into two sub-trajectories based on
/// even and odd indices, then applies the given function to both.
pub fn apply_to_split_traj_by_parity<T, F>(
    seq: &T,
    op_func: F,
) -> Result<(Trajectory, Trajectory), TrajDLError>
where
    T: PointSequence,
    F: Fn(Trajectory) -> Result<Trajectory, TrajDLError>,
{
    // Split by even and odd indices
    let (even_traj, odd_traj) = split_trajectory_by_parity(seq);

    // Apply function to both sub-trajectories
    Ok((op_func(even_traj)?, op_func(odd_traj)?))
}

/// Extract a continuous sub-trajectory from the input trajectory.
///
/// This function randomly selects a starting point and extracts a continuous
/// sub-trajectory of the specified ratio of the original length.
/// This is used in TrajCL for data augmentation.
///
/// # Parameters
/// ----------
/// * `seq` - The input trajectory
/// * `ratio` - The ratio of points to preserve (e.g., 0.7 means keep 70% of points)
///
/// # Returns
/// -------
/// * `Result<Trajectory, TrajDLError>` - A continuous sub-trajectory
pub fn subset_trajectory<T: PointSequence>(seq: &T, ratio: f64) -> Result<Trajectory, TrajDLError> {
    let num_points = seq.len();

    if num_points == 0 {
        return Err(TrajDLError::InvalidParams(
            "Cannot subset an empty trajectory".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&ratio) {
        return Err(TrajDLError::InvalidParams(
            "ratio must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Calculate the number of points to keep
    let keep_len = ((num_points as f64) * ratio).floor() as usize;

    // If keep_len is 0, return at least 1 point
    let keep_len = keep_len.max(1);

    // If keep_len >= original length, return a copy
    if keep_len >= num_points {
        return seq.to_container();
    }

    // Randomly select a starting index
    let max_start_idx = num_points - keep_len;
    let start_idx = aug_rng().lock().unwrap().random_range(0..=max_start_idx);

    // Extract the sub-trajectory
    let result: Vec<[f64; 2]> = seq
        .iter_coords()
        .skip(start_idx)
        .take(keep_len)
        .map(|c| c.to_pair())
        .collect();

    Ok(Trajectory::from_vec(result, seq.coord_system()))
}

/// Mask (remove) a specific ratio of points from the trajectory.
///
/// Unlike `downsampling` which uses probability-based removal, this function
/// removes exactly `ratio` fraction of points randomly selected.
/// This matches the TrajCL `mask` augmentation behavior.
///
/// # Parameters
/// ----------
/// * `seq` - The input trajectory
/// * `ratio` - The ratio of points to remove (e.g., 0.3 means remove 30% of points)
/// * `min_length` - Minimum length of the resulting trajectory (default: 2)
///
/// # Returns
/// -------
/// * `Result<Trajectory, TrajDLError>` - The masked trajectory
///
/// # Note
/// -----
/// This function does NOT guarantee keeping endpoints.
/// If the trajectory is too short, it will return the original trajectory.
pub fn mask_trajectory<T: PointSequence>(
    seq: &T,
    ratio: f64,
    min_length: usize,
) -> Result<Trajectory, TrajDLError> {
    let num_points = seq.len();

    if num_points == 0 {
        return Err(TrajDLError::InvalidParams(
            "Cannot mask an empty trajectory".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&ratio) {
        return Err(TrajDLError::InvalidParams(
            "ratio must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Calculate number of points to remove
    let num_to_remove = ((num_points as f64) * ratio).floor() as usize;

    // Ensure minimum length
    let num_to_remove = num_to_remove.min(num_points.saturating_sub(min_length));

    // If nothing to remove, return a copy
    if num_to_remove == 0 {
        return seq.to_container();
    }

    // Randomly select indices to remove
    let mut indices: Vec<usize> = (0..num_points).collect();
    {
        let mut rng = aug_rng().lock().unwrap();

        // Fisher-Yates shuffle for the first num_to_remove positions
        for i in 0..num_to_remove {
            let j = rng.random_range(i..num_points);
            indices.swap(i, j);
        }
    }

    // The first num_to_remove indices are the ones to remove
    let remove_set: std::collections::HashSet<usize> =
        indices.into_iter().take(num_to_remove).collect();

    // Filter out the removed indices
    let result: Vec<[f64; 2]> = seq
        .iter_coords()
        .enumerate()
        .filter(|(idx, _)| !remove_set.contains(idx))
        .map(|(_, c)| c.to_pair())
        .collect();

    Ok(Trajectory::from_vec(result, seq.coord_system()))
}
