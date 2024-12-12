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

// trajutils.h

#ifndef TRAJUTILS_H
#define TRAJUTILS_H

#include <vector>
#include <string>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "coord.h"
#include "region.h"

namespace py = pybind11;

std::string seq2str(const std::vector<int> &seq);

// 针对一条轨迹数据随机丢弃掉一些轨迹点，得到新的轨迹
py::array_t<double> downsampling(const py::array_t<double> &traj, double rate);

// 随机将一条轨迹序列里面的轨迹点进行扰动
py::array_t<double> distort(const py::array_t<double> &traj, double rate, double radius);

py::tuple split_array_by_first_dim_index(const py::array_t<double> &traj);

#endif // TRAJUTILS_H