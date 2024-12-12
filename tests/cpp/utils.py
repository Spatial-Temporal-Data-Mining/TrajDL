# Copyright 2024 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math


def gps_to_web_mercator(lng, lat):
    # 将经度和纬度从度转换为弧度
    lng_rad = math.radians(lng)
    lat_rad = math.radians(lat)

    # 地球半径 (WGS84)
    R = 6378137

    # 计算Web Mercator坐标
    x = R * lng_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))

    return x, y


def web_mercator_to_gps(x, y):
    # 计算经度
    lng = x / 6378137.0 * 180.0 / math.pi

    # 计算纬度
    lat = (2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2) * 180.0 / math.pi

    return lng, lat


def web_to_loc(
    x: float, y: float, minx: float, miny: float, numx: int, xstep: float, ystep: float
) -> str:
    return str(math.floor((y - miny) / ystep) * numx + math.floor((x - minx) / xstep))
