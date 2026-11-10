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

//! T2Vec algorithm functions.

use std::collections::HashMap;

use pyo3::{Bound, PyAny, pyfunction};
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

/// Convert a longitude/latitude sequence into a location sequence, filtering out consecutive duplicate locations
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (seq, web_grid, gps_boundary, unk_loc, bos_token = None, eos_token = None))]
pub fn convert_coords_to_locseq(
    seq: &Bound<'_, PyAny>,
    web_grid: &PySimpleGridSystem,
    gps_boundary: &PyRectangleBoundary,
    unk_loc: &str,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<Vec<String>, TrajDLError> {
    let seq = PyTrajectoryType::try_from(seq)?;

    // 定位闭包：对于不在区域内的经纬度点返回unk_loc，否则通过web mercator坐标定位
    let locate = |coord: Coordinate| {
        if gps_boundary.in_boundary(&coord) {
            let web_point = coord.to_web_mercator();
            web_grid.coord_to_grid_symbol(&web_point)
        } else {
            unk_loc.to_string()
        }
    };

    let mut result = Vec::with_capacity(seq.len() + 2);
    if let Some(bos) = bos_token {
        result.push(bos.to_string());
    }
    let mut last_loc: Option<String> = None;
    for (loc, _) in seq.get_locs(locate) {
        if last_loc.as_ref() != Some(&loc) {
            result.push(loc.clone());
            last_loc = Some(loc)
        }
    }

    if let Some(eos) = eos_token {
        result.push(eos.to_string());
    }

    Ok(result)
}

/// Efficient Rust implementation of `bisect_left`.
///
/// This function mimics the behavior of Python's `bisect.bisect_left`.
/// It finds the insertion point of `value` in the **sorted** slice `slice` and returns its index.
/// The returned index `i` splits the slice into two parts:
/// - every `slice[j]` with `j < i` satisfies `slice[j] < value`;
/// - every `slice[j]` with `j >= i` satisfies `slice[j] >= value`.
///
/// `#[inline]` is a compiler hint suggesting that this function's code be embedded
/// directly at each call site, which cuts call overhead for small, frequently called
/// functions and improves performance.
///
/// # Arguments
/// * `slice`: a sorted slice of i64 values representing the bucket boundaries.
/// * `value`: the value whose position is to be found in the slice.
///
/// # Returns
/// A `usize` index indicating where `value` should be inserted.
#[inline]
fn bisect_left(slice: &[i64], value: i64) -> usize {
    // 这里的闭包 `|&x| x < value` 正是 `bisect_left` 的定义：找到第一个不小于 `value` 的元素的位置。
    slice.partition_point(|&x| x < value)
}

/// Assign data indices to different buckets based on the source and target sequence lengths.
///
/// This is a two-dimensional bucketing strategy; the 2-D indices are then mapped to a 1-D index.
/// 1. **First dimension (primary bucket)**: determine the primary bucket index `trg_buc_idx` from `max(source length, target length)` and the `trg_bound` boundaries.
/// 2. **Second dimension (sub-bucket)**: once the primary bucket is fixed, determine the sub-bucket index `src_buc_idx` (0 or 1) from the `source length` and the `src_bound[trg_buc_idx]` boundary.
/// 3. **Final index**: map the 2-D index `(trg_buc_idx, src_buc_idx)` to the 1-D `bucket_key` via the formula `2 * trg_buc_idx + src_buc_idx`.
///
/// # Arguments
/// * `src_lengths`: array of source sequence lengths.
/// * `src_label_indices`: array mapping each source-sequence index to its paired target-sequence index.
/// * `trg_lengths`: array of target sequence lengths.
/// * `src_bound`: boundary values defining the sub-buckets inside each primary bucket. Its length must match `trg_bound`.
/// * `trg_bound`: boundary values defining the primary buckets; must be sorted.
///
/// # Returns
/// A `HashMap` where:
/// - `key (usize)`: the computed 1-D bucket id.
/// - `value (Vec<usize>)`: the list of original data indices that fall into that bucket.
pub fn bucketize_logic(
    src_lengths: &[i64],
    src_label_indices: &[i64],
    trg_lengths: &[i64],
    src_bound: &[i64],
    trg_bound: &[i64],
) -> HashMap<usize, Vec<usize>> {
    let length = src_lengths.len();
    let mut bucket_result: HashMap<usize, Vec<usize>> = HashMap::new();

    for idx in 0..length {
        // 获取当前源序列的长度
        let src_len = src_lengths[idx];

        // 获取当前源序列对应的目标序列的索引
        let src_label_idx = src_label_indices[idx] as usize;

        // 使用上面的索引，找到对应的目标序列的长度
        let trg_len = trg_lengths[src_label_idx];

        // --- 第一维分桶 ---
        // 计算主桶索引。分桶的依据是源长度和目标长度中的较大者。
        // 这种做法使得分桶逻辑对于 `src_len` 和 `trg_len` 的关系不敏感，更加健壮。
        let trg_buc_idx = bisect_left(trg_bound, src_len.max(trg_len));

        // --- 边界安全检查 ---
        // `bisect_left` 返回的 `trg_buc_idx` 范围是 `[0, trg_bound.len()]`。
        // 如果 `trg_buc_idx` 等于 `src_bound.len()` (假设 `trg_bound` 和 `src_bound` 长度相同),
        // 那么接下来的 `src_bound[trg_buc_idx]` 会导致数组越界 panic。
        // 因此，必须使用 `>=` 进行检查，以处理所有越界情况。
        if trg_buc_idx >= src_bound.len() {
            continue;
        }

        // --- 第二维分桶 ---
        // 在主桶内部进行二次划分，形成子桶。
        // `src_bound[trg_buc_idx]` 是与当前主桶 `trg_buc_idx` 对应的子桶边界。
        // 根据源序列的长度 `src_len` 与这个边界的比较，确定子桶索引为 0 或 1。
        let src_buc_idx = if src_len <= src_bound[trg_buc_idx] {
            0 // 小于等于边界，放入子桶 0
        } else {
            1 // 大于边界，放入子桶 1
        };

        // --- 最终桶索引计算 ---
        // 将二维的桶索引 (trg_buc_idx, src_buc_idx) 映射为一个一维的线性索引 `bucket_key`。
        // 例如: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3, ...
        let bucket_key = 2 * trg_buc_idx + src_buc_idx;

        // --- 存储结果 ---
        bucket_result.entry(bucket_key).or_default().push(idx);
    }

    bucket_result
}
