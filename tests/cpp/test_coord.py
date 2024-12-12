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

import random

import numpy as np

from trajdl import trajdl_cpp

from .utils import gps_to_web_mercator, web_mercator_to_gps


def test_coord():
    for _ in range(1000):
        lng, lat = random.uniform(-180, 180), random.uniform(-90, 90)
        coord = trajdl_cpp.Coord(lng, lat)
        assert np.allclose(lng, coord.lng)
        assert np.allclose(lat, coord.lat)


def test_web_coord():
    for _ in range(1000):
        x, y = random.uniform(-20037508.342789244, 20037508.342789244), random.uniform(
            103672149.75803147, -103672149.75439242
        )
        coord = trajdl_cpp.WebMercatorCoord(x, y)
        assert np.allclose(x, coord.x)
        assert np.allclose(y, coord.y)


def test_convert_gps_to_webmercator():
    for _ in range(1000):
        lng, lat = random.uniform(-180, 180), random.uniform(-90, 90)
        web_coord = trajdl_cpp.convert_gps_to_webmercator(lng, lat)
        web_coord_py = gps_to_web_mercator(lng, lat)
        assert np.allclose(web_coord_py[0], web_coord.x)
        assert np.allclose(web_coord_py[1], web_coord.y)


def test_convert_webmercator_to_gps():
    for _ in range(1000):
        x, y = random.uniform(-20037508.342789244, 20037508.342789244), random.uniform(
            103672149.75803147, -103672149.75439242
        )
        coord = trajdl_cpp.convert_webmercator_to_gps(x, y)
        coord_py = web_mercator_to_gps(x, y)
        assert np.allclose(coord_py[0], coord.lng)
        assert np.allclose(coord_py[1], coord.lat)
