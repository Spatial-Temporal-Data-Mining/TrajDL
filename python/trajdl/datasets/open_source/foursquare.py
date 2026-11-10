# Copyright 2024 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Raw readers for the public Foursquare TSMC2014 / TIST2015 check-in dumps.

This module is the hub layer's faithful-format-conversion boundary: it only
normalizes dtypes and (for TIST2015, whose check-in file carries no coordinates)
joins the published POI table onto the check-in rows. It applies no
dataset-specific filtering or rule of any kind; algorithm-level preprocessing
(e.g. the TALE paper rules) lives in each algorithm's benchmark layer.
"""

import pandas as pd

RAW_TSMC2014_COLUMNS = [
    "user_id",
    "poi_id",
    "category_id",
    "category_name",
    "lat",
    "lng",
    "tz_offset_min",
    "utc_time",
]

RAW_TIST2015_CHECKIN_COLUMNS = ["user_id", "poi_id", "utc_time", "tz_offset_min"]

RAW_TIST2015_POI_COLUMNS = ["poi_id", "lat", "lng", "category_name", "country"]

UTC_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def read_tsmc2014_txt(txt_path: str) -> pd.DataFrame:
    """Read one TSMC2014 city dump, preserving every row verbatim.

    ``utc_time`` stays a raw string: the hub layer performs no interpretation;
    timestamp parsing (strict vs coerce-and-drop) is algorithm-level semantics.
    """
    return pd.read_csv(
        txt_path,
        sep="\t",
        header=None,
        names=RAW_TSMC2014_COLUMNS,
        encoding="latin-1",
        dtype={
            "user_id": "int64",
            "poi_id": str,
            "category_id": str,
            "category_name": str,
            "lat": "float64",
            "lng": "float64",
            "tz_offset_min": "int32",
            "utc_time": str,
        },
    )


def read_tist2015(pois_path: str, checkins_path: str) -> pd.DataFrame:
    """Read the TIST2015 dumps chunk-wise and join POI attributes onto check-ins.

    All check-in rows are preserved verbatim (``utc_time`` stays a raw string;
    check-ins whose poi_id is absent from the POI table keep NaN attributes).
    """
    pois = pd.read_csv(
        pois_path,
        sep="\t",
        header=None,
        names=RAW_TIST2015_POI_COLUMNS,
        encoding="latin-1",
        dtype={"poi_id": str},
    ).drop_duplicates("poi_id")

    chunks = pd.read_csv(
        checkins_path,
        sep="\t",
        header=None,
        names=RAW_TIST2015_CHECKIN_COLUMNS,
        encoding="utf-8",
        encoding_errors="replace",
        dtype={
            "user_id": "int64",
            "poi_id": str,
            "utc_time": str,
            "tz_offset_min": "int64",
        },
        chunksize=5_000_000,
    )
    return pd.concat(
        [chunk.merge(pois, on="poi_id", how="left") for chunk in chunks],
        ignore_index=True,
    )
