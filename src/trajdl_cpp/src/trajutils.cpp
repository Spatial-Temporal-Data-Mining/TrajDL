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

#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <random>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "grid/coord.h"
#include "grid/region.h"

namespace py = pybind11;

std::string seq2str(const std::vector<int> &seq)
{
    std::ostringstream oss;
    for (size_t i = 0; i < seq.size(); ++i)
    {
        if (i != 0)
        {
            oss << " ";
        }
        oss << seq[i];
    }
    return oss.str();
}

// distort a trajectory
py::array_t<double> distort(const py::array_t<double> &traj, double rate, double radius)
{
    if (!py::isinstance<py::array_t<double>>(traj))
    {
        throw std::invalid_argument("traj must be a numpy array of type float64");
    }

    py::buffer_info buf_info = traj.request();
    auto num_points = buf_info.shape[0];
    if (buf_info.ndim != 2 || buf_info.shape[1] != 2)
    {
        throw std::invalid_argument("traj must be a 2D numpy array with shape (n, 2)");
    }

    py::array_t<double> result = py::array_t<double>({static_cast<int>(num_points), 2});
    auto ptr = result.mutable_unchecked<2>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    auto r = traj.unchecked<2>();
    for (int i = 0; i < num_points; ++i)
    {
        double lng = r(i, 0);
        double lat = r(i, 1);
        if (dis(gen) <= rate)
        {
            WebMercatorCoord web_point = convert_gps_to_webmercator(lng, lat);
            double xnoise = 2 * dis(gen) - 1;
            double ynoise = 2 * dis(gen) - 1;
            double normz = std::hypot(xnoise, ynoise);

            xnoise = xnoise * radius / normz;
            ynoise = ynoise * radius / normz;

            Coord coord = convert_webmercator_to_gps(web_point.x + xnoise, web_point.y + ynoise);
            ptr(i, 0) = coord.lng;
            ptr(i, 1) = coord.lat;
        }
        else
        {
            ptr(i, 0) = lng;
            ptr(i, 1) = lat;
        }
    }

    return result;
}

// down-sampling a trajectory, keep the first and the last point in trajectory
py::array_t<double> downsampling(const py::array_t<double> &traj, double rate)
{
    if (!py::isinstance<py::array_t<double>>(traj))
    {
        throw std::invalid_argument("traj must be a numpy array of type float64");
    }

    py::buffer_info buf_info = traj.request();
    auto num_points = buf_info.shape[0];

    if (buf_info.ndim != 2 || buf_info.shape[1] != 2)
    {
        throw std::invalid_argument("traj must be a 2D numpy array with shape (n, 2)");
    }

    if (rate == 0 || num_points <= 2)
    {
        return py::array_t<double>(traj);
    }

    std::vector<int> keep_idx = {0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // 随机选择保留的点
    for (int idx = 1; idx < num_points - 1; ++idx)
    {
        if (dis(gen) > rate)
        {
            keep_idx.push_back(idx);
        }
    }
    keep_idx.push_back(num_points - 1);

    int new_traj_length = keep_idx.size();

    py::array_t<double> result = py::array_t<double>({new_traj_length, 2});

    auto ptr = result.mutable_unchecked<2>();

    auto r = traj.unchecked<2>();

    for (int line_idx = 0; line_idx < new_traj_length; ++line_idx)
    {
        ptr(line_idx, 0) = r(keep_idx[line_idx], 0);
        ptr(line_idx, 1) = r(keep_idx[line_idx], 1);
    }

    return result;
}

py::tuple split_array_by_first_dim_index(const py::array_t<double> &traj)
{
    py::buffer_info buf_info = traj.request();
    int num_points = static_cast<int>(buf_info.shape[0]);
    if (buf_info.ndim != 2 || buf_info.shape[1] != 2)
    {
        throw std::invalid_argument("traj must be a 2D numpy array with shape (n, 2)");
    }
    int even_size = static_cast<int>(num_points / 2 + (num_points % 2 == 1 ? 1 : 0));
    int odd_size = static_cast<int>(num_points / 2);

    py::array_t<double> even_traj = py::array_t<double>({even_size, 2});
    py::array_t<double> odd_traj = py::array_t<double>({odd_size, 2});
    auto even_ptr = even_traj.mutable_unchecked<2>();
    auto odd_ptr = odd_traj.mutable_unchecked<2>();

    size_t even_idx = 0;
    size_t odd_idx = 0;

    auto ptr = traj.unchecked<2>();
    for (size_t i = 0; i < num_points; ++i)
    {
        double lng = ptr(i, 0);
        double lat = ptr(i, 1);
        if (i % 2 == 0) {
            even_ptr(even_idx, 0) = lng;
            even_ptr(even_idx, 1) = lat;
            even_idx++;
        } else {
            odd_ptr(odd_idx, 0) = lng;
            odd_ptr(odd_idx, 1) = lat;
            odd_idx++;
        }
    }

    return py::make_tuple(even_traj, odd_traj);
}