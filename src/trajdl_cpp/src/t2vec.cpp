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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include "grid/coord.h"
#include "grid/grid.h"
#include "grid/region.h"
#include "trajutils.h"
#include "t2vec.h"

namespace py = pybind11;

std::unordered_map<std::string, size_t> count_locations(const std::vector<py::array_t<double>> &traj_vec, RectangleBoundary gps_boundary, RectangleBoundary web_boundary, double step_x, double step_y, int64_t num_x_grids)
{
    std::unordered_map<std::string, size_t> frequency;

    for (py::array_t<double> traj : traj_vec)
    {
        py::buffer_info buf_info = traj.request();
        auto num_points = buf_info.shape[0];

        auto r = traj.unchecked<2>();

        for (int i = 0; i < num_points; i++)
        {
            double lng = r(i, 0);
            double lat = r(i, 1);
            if (gps_boundary.in_boundary(lng, lat)) {
                WebMercatorCoord web_point = convert_gps_to_webmercator(lng, lat);
                std::string loc = locate_in_grid(web_point.x, web_point.y, web_boundary, step_x, step_y, num_x_grids);
                frequency[loc]++;
            }
        }
    }

    return frequency;
}

// convert trajectory into list of locations
std::vector<std::string> convert_points_to_seq(
    py::array_t<double> traj,
    RectangleBoundary gps_boundary,
    RectangleBoundary web_boundary,
    double step_x, double step_y, int64_t num_x_grids,
    const std::string &unknown_loc,
    bool add_start_end_token,
    const std::string &bos_token,
    const std::string &eos_token)
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

    if (add_start_end_token)
    {
        locations.push_back(bos_token);
    }

    for (size_t line_idx = 0; line_idx < n_points; ++line_idx)
    {
        // set unknown_loc as default if the coord is not in region
        std::string loc = unknown_loc;
        double lng = ptr[line_idx * 2];
        double lat = ptr[line_idx * 2 + 1];

        if (gps_boundary.in_boundary(lng, lat))
        {
            WebMercatorCoord web_point = convert_gps_to_webmercator(lng, lat);
            loc = locate_in_grid(web_point.x, web_point.y, web_boundary, step_x, step_y, num_x_grids);
        }

        // Avoid duplicate consecutive locations
        if (locations.empty() || locations.back() != loc)
        {
            locations.push_back(loc);
        }
    }

    if (add_start_end_token)
    {
        locations.push_back(eos_token);
    }

    return locations;
}

size_t bisect_left(std::vector<int> &arr, int64_t value)
{
    size_t lo = 0, hi = arr.size();
    while (lo < hi)
    {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < value)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

std::unordered_map<int, std::vector<int>> bucketize(
    const py::array_t<int64_t> &src_lengths,
    const py::array_t<int64_t> &src_label_indices,
    const py::array_t<int64_t> &trg_lengths,
    std::vector<int> &src_bound,
    std::vector<int> &trg_bound)
{
    py::buffer_info buf_info_src = src_lengths.request();
    size_t n_points = buf_info_src.shape[0];

    py::buffer_info buf_info_src_label_indices = src_label_indices.request();

    py::buffer_info buf_info_trg = trg_lengths.request();

    // 获取指向数据的指针
    int64_t *ptr_src = static_cast<int64_t *>(buf_info_src.ptr);
    int64_t *ptr_src_label = static_cast<int64_t *>(buf_info_src_label_indices.ptr);
    int64_t *ptr_trg = static_cast<int64_t *>(buf_info_trg.ptr);

    std::unordered_map<int, std::vector<int>> bucket_result;

    for (size_t idx = 0; idx < n_points; ++idx)
    {
        int64_t src_len = ptr_src[idx];
        int64_t src_label_idx = ptr_src_label[idx];
        int64_t trg_len = ptr_trg[src_label_idx];
        size_t trg_idx = bisect_left(trg_bound, (src_len <= trg_len) ? trg_len : src_len);
        int src_idx = (src_len <= src_bound[trg_idx]) ? 0 : 1;
        bucket_result[trg_idx * 2 + src_idx].push_back(idx);
    }
    return bucket_result;
}

template <typename Func>
py::tuple apply_to_split_traj(const py::array_t<double> &traj, Func op_func)
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
        if (i % 2 == 0)
        {
            even_ptr(even_idx, 0) = lng;
            even_ptr(even_idx, 1) = lat;
            even_idx++;
        }
        else
        {
            odd_ptr(odd_idx, 0) = lng;
            odd_ptr(odd_idx, 1) = lat;
            odd_idx++;
        }
    }

    return py::make_tuple(op_func(even_traj), op_func(odd_traj));
}

py::tuple split_traj(const py::array_t<double> &traj)
{
    return apply_to_split_traj(traj, [](const py::array_t<double> &sub_traj)
                               { return sub_traj; });
}

py::tuple downsample_split_traj(const py::array_t<double> &traj, double rate)
{
    return apply_to_split_traj(traj, [rate](const py::array_t<double> &sub_traj)
                               { return downsampling(sub_traj, rate); });
}

py::tuple distort_split_traj(const py::array_t<double> &traj, double rate, double radius)
{
    return apply_to_split_traj(traj, [rate, radius](const py::array_t<double> &sub_traj)
                               { return distort(sub_traj, rate, radius); });
}

py::array_t<double> downsample_and_distort(
    const py::array_t<double> &traj,
    double downsample_rate,
    double distort_rate,
    double distort_radius)
{
    py::array_t<double> downsampled_array = downsampling(traj, downsample_rate);
    return distort(downsampled_array, distort_rate, distort_radius);
}

std::vector<py::array_t<double>> batch_downsample_and_distort(
    const py::array_t<double> &traj,
    std::vector<double> downsample_rate_list,
    std::vector<double> distort_rate_list,
    double distort_radius)
{
    std::vector<py::array_t<double>> result;
    for (auto downsample_rate : downsample_rate_list)
    {
        for (auto distort_rate : distort_rate_list)
        {
            result.push_back(downsample_and_distort(traj, downsample_rate, distort_rate, distort_radius));
        }
    }
    return result;
}