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

#include <cmath>
#include <vector>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "grid/coord.h"
#include "grid/region.h"

namespace py = pybind11;

RectangleBoundary::RectangleBoundary(double min_x, double min_y, double max_x, double max_y)
    : min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {}

std::string RectangleBoundary::repr() const
{
    return "RectangleBoundary(min_x=" + std::to_string(min_x) +
           ", min_y=" + std::to_string(min_y) +
           ", max_x=" + std::to_string(max_x) +
           ", max_y=" + std::to_string(max_y) + ")";
}

py::tuple RectangleBoundary::to_tuple() const
{
    return py::make_tuple(min_x, min_y, max_x, max_y);
}

RectangleBoundary RectangleBoundary::from_tuple(const py::tuple &tuple)
{
    if (tuple.size() != 4) {
        throw py::value_error("Length of tuple should be 4.");
    }
    return RectangleBoundary(py::cast<double>(tuple[0]), py::cast<double>(tuple[1]), py::cast<double>(tuple[2]), py::cast<double>(tuple[3]));
}

bool RectangleBoundary::in_boundary(double x, double y) const
{
    return min_x <= x && x < max_x && min_y <= y && y < max_y;
}

py::array_t<bool> RectangleBoundary::in_boundary_np(py::array_t<double> &coords) const
{
    // 确保传入的 ndarray 是二维的，并且列数为 2
    py::buffer_info buf_info = coords.request();
    if (buf_info.ndim != 2 || buf_info.shape[1] != 2)
    {
        throw std::invalid_argument("coords must be a 2D numpy array with shape (n, 2)");
    }

    size_t n = buf_info.shape[0];
    double *ptr = static_cast<double *>(buf_info.ptr);

    // Prepare the return array
    py::array_t<bool> result(n);
    auto r = result.mutable_unchecked<1>();

    for (size_t i = 0; i < n; i++)
    {
        double x = ptr[i * 2];
        double y = ptr[i * 2 + 1];
        r(i) = in_boundary(x, y);
    }
    return result;
}

RectangleBoundary RectangleBoundary::to_web_mercator() const
{
    WebMercatorCoord min_p = convert_gps_to_webmercator(min_x, min_y);
    WebMercatorCoord max_p = convert_gps_to_webmercator(max_x, max_y);
    return RectangleBoundary(min_p.x, min_p.y, max_p.x, max_p.y);
}