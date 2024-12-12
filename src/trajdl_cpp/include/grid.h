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

// grid.h
#ifndef GRID_H
#define GRID_H

#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <optional>
#include "coord.h"
#include "region.h"

namespace py = pybind11;

// 在网格内通过网格的idx定位位置
std::string locate_by_grid_coordinate(
    int64_t grid_x,
    int64_t grid_y,
    int64_t num_x_grids);

// 在网格内定位一个(x, y)的位置
std::string locate_in_grid(
    double x, double y,
    RectangleBoundary boundary,
    double step_x, double step_y,
    int64_t num_x_grids);

// 在网格内定位一组坐标的位置
std::vector<std::string> locate_in_grid_np(
    const py::array_t<double> &traj,
    RectangleBoundary boundary,
    double step_x, double step_y,
    int64_t num_x_grids,
    std::optional<std::string> unk_loc);

// 给定一个表示位置的字符串，在网格里面找到其原始网格坐标
GridCoord reverse_locate_in_grid(
    const std::string &loc,
    int64_t num_x_grids);

// 给定一个grid坐标，转换为对应网格的中心点坐标
Point grid_coord_to_centroid_point(const GridCoord &coord,
                                   double min_x, double min_y,
                                   double step_x, double step_y);

#endif // GRID_H
