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

import os

import pandas as pd
import polars as pl
import pytest

from trajdl.datasets.open_source.hasher import Hasher

from ...conftest import FOLDER


def test_hasher():
    df = pd.DataFrame([[0, 3, 2], [5, 1, 6], [2, 1, 1]], columns=["a", "b", "c"])
    pd_parquet_path = os.path.join(FOLDER, "test_pd.parquet")
    pl_parquet_path = os.path.join(FOLDER, "test_pl.parquet")
    df.to_parquet(pd_parquet_path, index=False)

    df = pl.read_parquet(pd_parquet_path)
    df.write_parquet(pl_parquet_path)

    digest_result_for_arrow = [
        "5d598adab1717dd557afe7b9d09c368b",
        "ecc644b2155d895faea21dbdc8f9cb42898501d931aba974f731e3fe3ac41115",
    ]

    for hasher_type, digest_result in zip(["mmh3", "sha256"], digest_result_for_arrow):
        hasher = Hasher(hasher_type=hasher_type)
        assert hasher.digest_file(pd_parquet_path) != hasher.digest_file(
            pl_parquet_path
        )
        assert hasher.digest_parquet(pd_parquet_path) == hasher.digest_parquet(
            pl_parquet_path
        )
        assert hasher.digest_arrow(df.to_arrow()) == digest_result

    with pytest.raises(ValueError):
        Hasher(hasher_type="MD5")
