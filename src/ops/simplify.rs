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

use crate::{
    err::TrajDLError,
    sequence::{point::container::Trajectory, traits::PointSequence},
};
use traj_dist_rs::distance::euclidean::euclidean_distance;
use traj_dist_rs::traits::AsCoord;

/// Compute the perpendicular distance from a point to the line segment formed by start and end
fn point_line_distance<C: AsCoord, D: AsCoord, E: AsCoord>(point: &C, start: &D, end: &E) -> f64 {
    if start.x() == end.x() && start.y() == end.y() {
        return euclidean_distance(point, start);
    }

    let n = ((end.x() - start.x()) * (start.y() - point.y())
        - (start.x() - point.x()) * (end.y() - start.y()))
    .abs();

    let d = ((end.x() - start.x()).powi(2) + (end.y() - start.y()).powi(2)).sqrt();

    if d == 0.0 {
        return euclidean_distance(point, start);
    }

    n / d
}

// --- RDP 核心实现 ---

/// Public interface of the RDP algorithm
///
/// # Arguments
/// * `points`: a reference to a point sequence implementing the PointSequence trait
/// * `epsilon`: tolerance threshold determining the degree of simplification
///
/// # Returns
/// * `Vec<[f64; 2]>`: the simplified list of points
pub fn rdp<T: PointSequence + ?Sized>(points: &T, epsilon: f64) -> Result<Trajectory, TrajDLError> {
    // 如果点的数量不足以形成一条需要简化的线，则直接返回
    if points.len() < 2 {
        return points.to_container();
    }

    // 调用递归辅助函数
    rdp_impl(points, 0, points.len() - 1, epsilon)
}

/// Recursive helper of the RDP algorithm
/// It handles subsets of the point sequence via indices to avoid unnecessary memory allocation
fn rdp_impl<T: PointSequence + ?Sized>(
    points: &T,
    start_idx: usize,
    end_idx: usize,
    epsilon: f64,
) -> Result<Trajectory, TrajDLError> {
    if end_idx <= start_idx + 1 {
        return Ok(Trajectory::from_vec(
            vec![points.get(start_idx), points.get(end_idx)],
            points.coord_system(),
        ));
    }

    let start_point = points.get(start_idx);
    let end_point = points.get(end_idx);

    let mut dmax = 0.0;
    let mut index = start_idx;

    // 寻找距离基准线最远的点
    for i in (start_idx + 1)..end_idx {
        let d = point_line_distance(&points.get(i), &start_point, &end_point);
        if d > dmax {
            index = i;
            dmax = d;
        }
    }

    // 判断最远点的距离是否大于阈值
    if dmax > epsilon {
        // 递归处理两段子序列
        let results1 = rdp_impl(points, start_idx, index, epsilon)?;
        let results2 = rdp_impl(points, index, end_idx, epsilon)?;

        // 合并结果。移除 `results1` 的最后一个元素，因为这个元素（即分割点）会是 `results2` 的第一个元素。
        let mut vec = Vec::with_capacity(results1.len() + results2.len() - 1);
        for idx in 0..results1.len() - 1 {
            vec.push(results1.get(idx));
        }
        for idx in 0..results2.len() {
            vec.push(results2.get(idx));
        }

        Ok(Trajectory::from_vec(vec, points.coord_system()))
    } else {
        // Base case: 简化这段曲线为一条直线，只保留起点和终点
        Ok(Trajectory::from_vec(
            vec![start_point, end_point],
            points.coord_system(),
        ))
    }
}
