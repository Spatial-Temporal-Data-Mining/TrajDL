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
#include <sstream>
#include <iomanip>
#include <string>
#include <pybind11/pybind11.h>
#include "grid/coord.h"

namespace py = pybind11;

const double R = 6378137.0; // 地球半径 (WGS84)

std::string GridCoord::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(7);
    oss << "GridCoord(grid_x=" << grid_x << ", grid_y=" << grid_y << ")";
    return oss.str();
}

py::tuple GridCoord::to_tuple() const {
    return py::make_tuple(grid_x, grid_y);
}

std::string Point::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(7);
    oss << "Point(x=" << x << ", y=" << y << ")";
    return oss.str();
}

std::string Coord::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(7);
    oss << "Coord(lng=" << lng << ", lat=" << lat << ")";
    return oss.str();
}

std::string WebMercatorCoord::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(7);
    oss << "WebMercatorCoord(x=" << x << ", y=" << y << ")";
    return oss.str();
}

WebMercatorCoord convert_gps_to_webmercator(double lng, double lat)
{
    // 将经度和纬度从度转换为弧度
    double lngRad = lng * M_PI / 180.0;
    double latRad = lat * M_PI / 180.0;

    // 计算Web Mercator坐标
    double x = R * lngRad;
    double y = R * std::log(std::tan(M_PI / 4 + latRad / 2));
    return WebMercatorCoord(x, y);
}

Coord convert_webmercator_to_gps(double x, double y)
{
    // 计算经度
    double lng = x / R * 180.0 / M_PI;

    // 计算纬度
    double lat = (2 * atan(exp(y / R)) - M_PI / 2) * 180.0 / M_PI;
    return Coord(lng, lat);
}