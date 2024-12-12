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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "coord.h"
#include "grid.h"
#include "region.h"
#include "t2vec.h"
#include "trajutils.h"

namespace py = pybind11;

PYBIND11_MODULE(trajdl_cpp, m)
{
    m.doc() = "trajdl C++ extension";

    py::class_<GridCoord>(m, "GridCoord")
        .def(py::init<int64_t, int64_t>())
        .def(py::init<int64_t, int64_t>(), py::arg("grid_x"), py::arg("grid_y"))
        .def_readwrite("grid_x", &GridCoord::grid_x)
        .def_readwrite("grid_y", &GridCoord::grid_y)
        .def("to_tuple", &GridCoord::to_tuple)
        .def("__repr__", [](const GridCoord &self)
             { return self.to_string(); });
    
    py::class_<Point>(m, "Point")
        .def(py::init<double, double>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def("__repr__", [](const Point &self)
             { return self.to_string(); });

    py::class_<Coord>(m, "Coord")
        .def(py::init<double, double>())
        .def_readwrite("lng", &Coord::lng)
        .def_readwrite("lat", &Coord::lat)
        .def("__repr__", [](const Coord &self)
             { return self.to_string(); });

    py::class_<WebMercatorCoord>(m, "WebMercatorCoord")
        .def(py::init<double, double>())
        .def_readwrite("x", &WebMercatorCoord::x)
        .def_readwrite("y", &WebMercatorCoord::y)
        .def("__repr__", [](const WebMercatorCoord &self)
             { return self.to_string(); });

    m.def("convert_gps_to_webmercator", &convert_gps_to_webmercator, "Convert a gps coordinate into a webmercator coordinate.");

    m.def("convert_webmercator_to_gps", &convert_webmercator_to_gps, "Convert a webmercator coordinate into a gps coordinate.");

    m.def("downsampling", &downsampling, "downsampling a trajectory");

    m.def("distort", &distort, "distort a trajectory");

    m.def("count_locations", &count_locations, "count cells");

    m.def("convert_points_to_seq", &convert_points_to_seq, "Convert vector of Points to vector of cell IDs");

    m.def("bucketize", &bucketize, "bucketize in t2vec");

    m.def("split_array_by_first_dim_index", &split_array_by_first_dim_index, "split_array_by_first_dim_index");

    m.def("split_traj", &split_traj, "split_traj");

    m.def("downsample_split_traj", &downsample_split_traj, "downsample_split_traj");

    m.def("distort_split_traj", &distort_split_traj, "distort_split_traj");

    m.def("downsample_and_distort", &downsample_and_distort, "downsample_and_distort");

    m.def("batch_downsample_and_distort", &batch_downsample_and_distort, "batch_downsample_and_distort");

    py::class_<RectangleBoundary>(m, "RectangleBoundary")
        .def(py::init<double, double, double, double>())
        .def(py::init<double, double, double, double>(), py::arg("min_x"), py::arg("min_y"), py::arg("max_x"), py::arg("max_y"))
        .def("__repr__", &RectangleBoundary::repr)
        .def("to_tuple", &RectangleBoundary::to_tuple)
        .def_static("from_tuple", &RectangleBoundary::from_tuple)
        .def("in_boundary", &RectangleBoundary::in_boundary)
        .def("in_boundary_np", &RectangleBoundary::in_boundary_np)
        .def("to_web_mercator", &RectangleBoundary::to_web_mercator)
        .def_readonly("min_x", &RectangleBoundary::min_x)
        .def_readonly("min_y", &RectangleBoundary::min_y)
        .def_readonly("max_x", &RectangleBoundary::max_x)
        .def_readonly("max_y", &RectangleBoundary::max_y)
        .def(py::pickle(
            [](const RectangleBoundary &obj)
            {
                return obj.to_tuple();
            },
            [](py::tuple tuple)
            {
                return RectangleBoundary::from_tuple(tuple);
            }));

    m.def("locate_by_grid_coordinate", &locate_by_grid_coordinate, "locate_by_grid_coordinate");

    m.def("locate_in_grid", &locate_in_grid, "locate_in_grid");

    m.def("locate_in_grid_np", &locate_in_grid_np, "locate_in_grid_np");

    m.def("reverse_locate_in_grid", &reverse_locate_in_grid, "reverse_locate_in_grid");

    m.def("grid_coord_to_centroid_point", &grid_coord_to_centroid_point, "grid_coord_to_centroid_point");
}