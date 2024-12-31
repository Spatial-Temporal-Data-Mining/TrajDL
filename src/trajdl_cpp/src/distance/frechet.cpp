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

#include <iostream>
#include <limits>
#include <algorithm>
#include <pybind11/embed.h>
#include "distance/base.h"
#include "distance/frechet.h"

/// @brief 检查numpy格式的轨迹序列是否符合要求
/// @param traj numpy格式的轨迹数据
void check_numpy_traj(
    const py::array_t<double> &traj)
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
}

double DistanceCalculator::compute_dis(size_t idx1, size_t idx2)
{
    // 优先使用距离矩阵计算
    if (distance_matrix.has_value())
    {
        auto ptr = distance_matrix.value().unchecked<2>();
        return ptr(idx1, idx2);
    }
    else
    {
        // 两条轨迹的指针
        auto traj1_ptr = traj1.unchecked<2>();
        auto traj2_ptr = traj2.unchecked<2>();

        double lng1 = traj1_ptr(idx1, 0);
        double lat1 = traj1_ptr(idx1, 1);
        double lng2 = traj2_ptr(idx2, 0);
        double lat2 = traj2_ptr(idx2, 1);

        if (metric == "euclidean")
        {
            return euclidean_distance(lng1, lat1, lng2, lat2);
        }
        else
        {
            // TODO
            return 0.0;
        }
    }
}

/// @brief 给定距离计算器的情况下计算两条序列的discrete frechet distance
/// @param distance_calculator 距离计算器
/// @param return_dp 是否返回dp矩阵
/// @param length1 序列1的长度
/// @param length2 序列2的长度
/// @return discrete frechet distance
DiscreteFrechetResult discrete_frechet(
    DistanceCalculator distance_calculator,
    bool return_dp,
    size_t length1,
    size_t length2)
{
    double inf = std::numeric_limits<double>::infinity();

    // 如果return_dp是true，创建一个二维的dp矩阵，然后返回
    if (return_dp)
    {
        // 定义一个DP矩阵，用来存储中间值
        py::array_t<double> dp_matrix = py::array_t<double>({length1 + 1, length2 + 1});

        // 读写dp_matrix的指针
        auto dp_ptr = dp_matrix.mutable_unchecked<2>();

        // 初始化
        dp_ptr(0, 0) = 0.0;
        for (size_t row_idx = 1; row_idx < length1 + 1; ++row_idx)
        {
            dp_ptr(row_idx, 0) = inf;
        }
        for (size_t col_idx = 1; col_idx < length2 + 1; ++col_idx)
        {
            dp_ptr(0, col_idx) = inf;
        }

        // 开始计算中间状态
        for (size_t row_idx = 1; row_idx < length1 + 1; ++row_idx)
        {
            for (size_t col_idx = 1; col_idx < length2 + 1; ++col_idx)
            {
                // 获取当前两个点之间的距离
                double current_dis = distance_calculator.compute_dis(row_idx - 1, col_idx - 1);

                dp_ptr(row_idx, col_idx) = std::max(
                    current_dis,
                    std::min({dp_ptr(row_idx - 1, col_idx),
                              dp_ptr(row_idx, col_idx - 1),
                              dp_ptr(row_idx - 1, col_idx - 1)}));
            }
        }

        return DiscreteFrechetResult(dp_ptr(length1, length2), dp_matrix);
    }
    else
    {
        // 如果return_dp是false，创建一个2行的dp矩阵，计算最终值
        double dp_matrix[2][length2 + 1] = {0};

        // 初始化第一行，除了第一个原始是0，其他都是inf
        for (size_t col_idx = 1; col_idx < length2 + 1; ++col_idx)
        {
            dp_matrix[0][col_idx] = inf;
        }

        for (size_t row_idx = 1; row_idx < length1 + 1; ++row_idx)
        {
            // 当前需要更新的行idx
            int current_row_idx = row_idx % 2;

            // 对于非第一行，第一个元素都要初始化为inf
            dp_matrix[current_row_idx][0] = inf;

            for (size_t col_idx = 1; col_idx < length2 + 1; ++col_idx)
            {
                // 获取当前两个点之间的距离
                double current_dis = distance_calculator.compute_dis(row_idx - 1, col_idx - 1);

                dp_matrix[current_row_idx][col_idx] = std::max(
                    current_dis,
                    std::min({dp_matrix[current_row_idx][col_idx - 1],
                              dp_matrix[1 - current_row_idx][col_idx],
                              dp_matrix[1 - current_row_idx][col_idx - 1]}));
            }
        }

        return DiscreteFrechetResult(dp_matrix[length1 % 2][length2]);
    }
}

/// @brief 给定metric的情况下计算两条轨迹的discrete frechet distance
/// @param traj1 轨迹1
/// @param traj2 轨迹2
/// @param metric metric
/// @param return_dp 是否返回dp矩阵
/// @return discrete frechet distance
DiscreteFrechetResult discrete_frechet_traj1(
    const py::array_t<double> &traj1,
    const py::array_t<double> &traj2,
    const std::string &metric,
    bool return_dp)
{
    check_numpy_traj(traj1);
    check_numpy_traj(traj2);
    DistanceCalculator distance_calculator(traj1, traj2, metric);
    return discrete_frechet(distance_calculator, return_dp, traj1.request().shape[0], traj2.request().shape[0]);
}

/// @brief 给定两条轨迹序列和一个预计算好的距离矩阵，计算discrete frechet distance
/// @param traj1 轨迹1
/// @param traj2 轨迹2
/// @param distance_matrix 预计算好的距离矩阵
/// @param return_dp 是否返回dp矩阵
/// @return discrete frechet distance
DiscreteFrechetResult discrete_frechet_traj2(
    const py::array_t<double> &traj1,
    const py::array_t<double> &traj2,
    const py::array_t<double> &distance_matrix,
    bool return_dp)
{
    // 检查两条轨迹序列是否合规
    check_numpy_traj(traj1);
    check_numpy_traj(traj2);

    size_t length1 = traj1.request().shape[0];
    size_t length2 = traj2.request().shape[0];

    py::buffer_info buf_info = distance_matrix.request();
    size_t size1 = buf_info.shape[0];
    size_t size2 = buf_info.shape[1];

    if (length1 != size1 || length2 != size2)
    {
        throw std::invalid_argument("lengths of traj1 and traj2 should be equal to size of distance_matrix");
    }

    return discrete_frechet_via_distance_matrix(distance_matrix, return_dp, size1, size2);
}

/// @brief 给定两条位置序列和预计算好的距离矩阵，计算这两条序列的discrete frechet distance
/// @param locseq1 位置序列1
/// @param locseq2 位置序列2
/// @param distance_matrix 预计算的距离矩阵
/// @param return_dp 是否返回dp矩阵
/// @return discrete frechet distance
DiscreteFrechetResult discrete_frechet_locseq(
    const std::vector<std::string> &locseq1,
    const std::vector<std::string> &locseq2,
    const py::array_t<double> &distance_matrix,
    bool return_dp)
{
    size_t length1 = locseq1.size();
    size_t length2 = locseq2.size();

    py::buffer_info buf_info = distance_matrix.request();
    size_t size1 = buf_info.shape[0];
    size_t size2 = buf_info.shape[1];

    if (length1 != size1 || length2 != size2)
    {
        throw std::invalid_argument("lengths of locseq1 and locseq2 should be equal to size of distance_matrix");
    }

    return discrete_frechet_via_distance_matrix(distance_matrix, return_dp, size1, size2);
}

/// @brief 给定距离矩阵情况下的discrete frechet distance计算
/// @param distance_matrix 距离矩阵
/// @param return_dp 是否返回dp矩阵
/// @param length1 序列1的length
/// @param length2 序列2的length
/// @return discrete frechet distance
DiscreteFrechetResult discrete_frechet_via_distance_matrix(
    const py::array_t<double> &distance_matrix,
    bool return_dp,
    size_t length1,
    size_t length2)
{
    DistanceCalculator distance_calculator(distance_matrix);
    return discrete_frechet(distance_calculator, return_dp, length1, length2);
}

int main()
{
    py::scoped_interpreter guard{};
    py::array_t<double> distance_matrix({3, 3});
    auto ptr = distance_matrix.mutable_unchecked<2>();
    ptr(0, 0) = 1;
    ptr(0, 1) = 2;
    ptr(0, 2) = 3;
    ptr(1, 0) = 4;
    ptr(1, 1) = 5;
    ptr(1, 2) = 6;
    ptr(2, 0) = 7;
    ptr(2, 1) = 8;
    ptr(2, 2) = 9;

    DiscreteFrechetResult r = discrete_frechet_via_distance_matrix(distance_matrix, false, 3, 3);
    std::cout << r.distance << std::endl;
}