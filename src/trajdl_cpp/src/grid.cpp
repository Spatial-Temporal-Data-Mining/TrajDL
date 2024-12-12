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

#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <optional>
#include "grid.h"
#include "region.h"

namespace py = pybind11;

std::string locate_by_grid_coordinate(
    int64_t grid_x,
    int64_t grid_y,
    int64_t num_x_grids)
{
    return std::to_string(grid_y * num_x_grids + grid_x);
};

std::string locate_in_grid(
    double x, double y,
    RectangleBoundary boundary,
    double step_x, double step_y,
    int64_t num_x_grids)
{
    double xoffset = (x - boundary.min_x) / step_x;
    double yoffset = (y - boundary.min_y) / step_y;

    xoffset = std::floor(xoffset);
    yoffset = std::floor(yoffset);

    return locate_by_grid_coordinate(static_cast<int64_t>(xoffset), static_cast<int64_t>(yoffset), num_x_grids);
}

std::vector<std::string> locate_in_grid_np(
    const py::array_t<double> &traj,
    RectangleBoundary boundary,
    double step_x, double step_y,
    int64_t num_x_grids,
    std::optional<std::string> unk_loc)
{
    // 确保传入的 ndarray 是二维的，并且列数为 2
    py::buffer_info buf_info = traj.request();
    if (buf_info.ndim != 2 || buf_info.shape[1] != 2)
    {
        throw std::invalid_argument("traj must be a 2D numpy array with shape (n, 2)");
    }

    // 获取指向数据的指针
    double *ptr = static_cast<double *>(buf_info.ptr);
    size_t n_points = buf_info.shape[0];

    std::vector<std::string> locations;

    for (size_t line_idx = 0; line_idx < n_points; ++line_idx)
    {
        double x = ptr[line_idx * 2];
        double y = ptr[line_idx * 2 + 1];
        std::string loc = locate_in_grid(x, y, boundary, step_x, step_y, num_x_grids);

        // 如果使用unk_tag并且当前坐标不在boundary里面的时候，使用unk_loc
        if (unk_loc.has_value() && !boundary.in_boundary(x, y))
        {
            loc = unk_loc.value();
        }
        locations.push_back(loc);
    }

    return locations;
}

GridCoord reverse_locate_in_grid(
    const std::string &loc,
    int64_t num_x_grids)
{
    int64_t cell_id = std::stoll(loc);
    int64_t yoffset = cell_id / num_x_grids;
    int64_t xoffset = cell_id % num_x_grids;
    return GridCoord(xoffset, yoffset);
}

Point grid_coord_to_centroid_point(const GridCoord &coord,
                                   double min_x, double min_y,
                                   double step_x, double step_y)
{
    int64_t xoffset = coord.grid_x;
    int64_t yoffset = coord.grid_y;

    double x = min_x + (xoffset + 0.5) * step_x;
    double y = min_y + (yoffset + 0.5) * step_y;
    return Point(x, y);
}