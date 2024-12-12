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

// region.h
#ifndef REGION_H
#define REGION_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include "coord.h"

namespace py = pybind11;

class RectangleBoundary
{
public:
    double min_x;
    double min_y;
    double max_x;
    double max_y;

    RectangleBoundary(double min_x, double min_y, double max_x, double max_y);

    // 字符串表示
    std::string repr() const;

    // min_x, min_y, max_x, max_y按顺序组成python tuple
    py::tuple to_tuple() const;

    // 通过python的tuple构建Boundary
    static RectangleBoundary from_tuple(const py::tuple &tuple);

    // 判断一个点是否在区域内
    bool in_boundary(double x, double y) const;

    // 判断一个numpy.ndarray类型的轨迹是否在区域内
    py::array_t<bool> in_boundary_np(py::array_t<double> &coords) const;

    // 将当前区域的坐标系的4个角转换到Web Mercator坐标系
    RectangleBoundary to_web_mercator() const;
};

#endif // REGION_H