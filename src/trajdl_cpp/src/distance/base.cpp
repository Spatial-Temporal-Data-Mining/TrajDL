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

#include "distance/base.h"

/// @brief 计算欧式距离
/// @param lng1 第一个点的经度
/// @param lat1 第一个点的纬度
/// @param lng2 第二个点的经度
/// @param lat2 第二个点的纬度
/// @return 两个点之间的欧氏距离
double euclidean_distance(double lng1, double lat1, double lng2, double lat2)
{
    return sqrt(pow(lng1 - lng2, 2) + pow(lat1 - lat2, 2));
}