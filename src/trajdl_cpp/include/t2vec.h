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

// t2vec.h

#ifndef T2VEC_H
#define T2VEC_H

#include <vector>
#include <string>
#include <unordered_map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "grid/coord.h"
#include "grid/region.h"
#include "trajutils.h"

namespace py = pybind11;

std::unordered_map<std::string, size_t> count_locations(
    const std::vector<py::array_t<double>> &traj_vec, 
    RectangleBoundary gps_boundary, 
    RectangleBoundary web_boundary, 
    double step_x, 
    double step_y, 
    int64_t num_x_grids);

// convert trajectory into list of locations
std::vector<std::string> convert_points_to_seq(
    py::array_t<double> traj,
    RectangleBoundary gps_boundary,
    RectangleBoundary web_boundary,
    double step_x, double step_y, int64_t num_x_grids,
    const std::string &unknown_loc,
    bool add_start_end_token,
    const std::string &bos_token,
    const std::string &eos_token);

size_t bisect_left(std::vector<int> &arr, int64_t value);

std::unordered_map<int, std::vector<int>> bucketize(
    const py::array_t<int64_t> &src_lengths,
    const py::array_t<int64_t> &src_label_indices,
    const py::array_t<int64_t> &trg_lengths,
    std::vector<int> &src_bound,
    std::vector<int> &trg_bound);

py::tuple split_traj(const py::array_t<double> &traj);

py::tuple downsample_split_traj(const py::array_t<double> &traj, double rate);

py::tuple distort_split_traj(const py::array_t<double> &traj, double rate, double radius);

// 对序列进行下采样与扰动
py::array_t<double> downsample_and_distort(
    const py::array_t<double> &traj,
    double downsample_rate,
    double distort_rate,
    double distort_radius);

std::vector<py::array_t<double>> batch_downsample_and_distort(
    const py::array_t<double> &traj,
    std::vector<double> downsample_rate_list,
    std::vector<double> distort_rate_list,
    double distort_radius);

#endif // T2VEC_H