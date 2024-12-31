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
#include "t2vec.h"
#include "trajutils.h"
#include "binding/distance_binding.h"
#include "binding/grid_binding.h"

namespace py = pybind11;

PYBIND11_MODULE(trajdl_cpp, m)
{
    m.doc() = "trajdl C++ extension";

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

    bind_grid_module(m);
    bind_distance_module(m);
}