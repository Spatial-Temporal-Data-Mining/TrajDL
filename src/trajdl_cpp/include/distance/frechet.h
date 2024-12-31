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

// frechet.h

#ifndef FRECHET_H
#define FRECHET_H

#include <string>
#include <vector>
#include <optional>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct DiscreteFrechetResult
{
    double distance;
    std::optional<py::array_t<double>> dp_matrix;

    // 构造函数
    DiscreteFrechetResult(double d) : distance(d), dp_matrix(std::nullopt) {}
    DiscreteFrechetResult(double d, py::array_t<double> dp) : distance(d), dp_matrix(std::move(dp)) {}
};

class DistanceCalculator
{
    py::array_t<double> traj1;
    py::array_t<double> traj2;
    std::string metric;

    std::optional<py::array_t<double>> distance_matrix;

public:
    // 构造方法1
    DistanceCalculator(const py::array_t<double> &traj1_val, const py::array_t<double> &traj2_val, const std::string &metric_val) : traj1(traj1_val), traj2(traj2_val), metric(metric_val), distance_matrix(std::nullopt) {}

    // 构造方法2
    DistanceCalculator(const py::array_t<double> &distance_matrix_val) : distance_matrix(distance_matrix_val) {}

    // 距离计算方法
    double compute_dis(size_t idx1, size_t idx2);
};

// 给定两条轨迹序列，计算其discrete frechet distance
DiscreteFrechetResult discrete_frechet_traj1(const py::array_t<double> &traj1, const py::array_t<double> &traj2, const std::string &metric, bool return_dp);

// 给定两条轨迹序列，计算其discrete frechet distance
DiscreteFrechetResult discrete_frechet_traj2(const py::array_t<double> &traj1, const py::array_t<double> &traj2, const py::array_t<double> &distance_matrix, bool return_dp);

// 给定两条位置序列，计算其discrete frechet distance
DiscreteFrechetResult discrete_frechet_locseq(const std::vector<std::string> &locseq1, const std::vector<std::string> &locseq2, const py::array_t<double> &distance_matrix, bool return_dp);

// 通过一个提前计算好的距离矩阵计算discrete frechet distance
DiscreteFrechetResult discrete_frechet_via_distance_matrix(const py::array_t<double> &distance_matrix, bool return_dp, size_t size1, size_t size2);

#endif // FRECHET_H