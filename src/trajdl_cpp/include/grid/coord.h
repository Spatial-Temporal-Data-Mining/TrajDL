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

// coord.h
#ifndef COORD_H
#define COORD_H

#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct GridCoord
{
  int64_t grid_x;
  int64_t grid_y;
  GridCoord(int64_t grid_x_val, int64_t grid_y_val) : grid_x(grid_x_val), grid_y(grid_y_val) {}

  std::string to_string() const;

  py::tuple to_tuple() const;
};

struct Point
{
  double x;
  double y;
  Point(double x_val, double y_val) : x(x_val), y(y_val) {}

  std::string to_string() const;
};

// coordinate
struct Coord
{
  double lng;
  double lat;
  Coord(double longitude, double latitude) : lng(longitude), lat(latitude) {}

  std::string to_string() const;
};

// coordinate in web mercator
struct WebMercatorCoord
{
  double x;
  double y;
  WebMercatorCoord(double x_value, double y_value) : x(x_value), y(y_value) {}

  std::string to_string() const;
};

WebMercatorCoord convert_gps_to_webmercator(double lng, double lat);

Coord convert_webmercator_to_gps(double x, double y);

#endif // COORD_H
