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

import argparse
import math
import os
from pathlib import Path
from typing import List

import polars as pl
from tqdm import tqdm

from trajdl.datasets import LocSeq, LocSeqDataset
from trajdl.datasets.open_source import GowallaDataset
from trajdl.tokenizers import LocSeqTokenizer
from trajdl.tokenizers.slot import Bucketizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("output") / "gowalla",
        help="存储目录",
    )
    args = parser.parse_args()

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # 读取gowalla数据集，按日期切分
    # 计算任意两点之间的时间差和位移差，每日第一个点的时间差和位移差设定为0
    detail_df = (
        GowallaDataset()
        .load(return_as="pl")
        .sort(["user_id", "check_in_time"])
        .with_columns(ds=pl.col("check_in_time").dt.strftime("%Y%m%d"))
        .with_columns(tmp_id=pl.int_range(pl.len()).over("user_id", "ds"))
        .with_columns(
            ts_delta=(
                pl.col("check_in_time") - pl.col("check_in_time").shift(1)
            ).dt.total_seconds()
        )
        .with_columns(
            ts_delta=pl.when(pl.col("tmp_id") == 0)
            .then(pl.lit(0))
            .otherwise(pl.col("ts_delta"))
        )
        .with_columns(
            lat1=pl.col("lat") * math.pi / 180, lng1=pl.col("lng") * math.pi / 180
        )
        .with_columns(lat2=pl.col("lat1").shift(1), lng2=pl.col("lng1").shift(1))
        .with_columns(
            dlat=(pl.col("lat2") - pl.col("lat1")),
            dlng=(pl.col("lng2") - pl.col("lng1")),
        )
        .with_columns(
            a=((pl.col("dlat") / 2).sin() ** 2)
            + pl.col("lat1").cos()
            * pl.col("lat2").cos()
            * ((pl.col("dlng") / 2).sin() ** 2)
        )
        .with_columns(dis_delta=2 * (pl.col("a") ** 0.5).arcsin() * 6371)
        .with_columns(
            dis_delta=pl.when(pl.col("tmp_id") == 0)
            .then(pl.lit(0))
            .otherwise(pl.col("dis_delta"))
        )
        .drop("lat1", "lat2", "lng1", "lng2", "dlat", "dlng", "a", "tmp_id")
    )

    # 按日进行序列切分，并且只保留序列长度大于3小于等于10的序列，这些序列称为session
    sessions = (
        detail_df.group_by("user_id", "ds")
        .agg(
            pl.len().alias("seq_len"),
            pl.col("loc_id").sort_by("check_in_time").alias("loc_seq"),
            pl.col("ts_delta").sort_by("check_in_time").alias("ts_delta"),
            pl.col("dis_delta").sort_by("check_in_time").alias("dis_delta"),
            (pl.col("check_in_time").min().dt.timestamp() / 1e6).alias("start_ts"),
        )
        .filter(pl.col("seq_len") > 3)
        .filter(pl.col("seq_len") <= 10)
        .select("user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts")
    )

    # 统计每个用户的session数
    num_sessions = (
        sessions.group_by("user_id").agg(pl.len().alias("num_sessions"))
        # .filter(pl.col("num_sessions") <= 10)
    )

    tmp_df = sessions.join(num_sessions, on=["user_id"], how="inner")

    # 对每个用户，按start_ts切分，训练集、验证集、测试集6:2:2
    tmp_df = (
        tmp_df.sort(["user_id", "start_ts"])
        .with_columns(
            tmp_id=pl.int_range(pl.len()).over("user_id", order_by="start_ts")
        )
        .join(
            tmp_df.group_by("user_id").agg(pl.len().alias("num_sessions")),
            how="left",
            on=["user_id"],
        )
    )

    train_df = tmp_df.filter(pl.col("tmp_id") <= pl.col("num_sessions") * 0.6).select(
        "user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts"
    )
    val_df = tmp_df.filter(
        (pl.col("tmp_id") > pl.col("num_sessions") * 0.6)
        & (pl.col("tmp_id") <= pl.col("num_sessions") * 0.8)
    ).select("user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts")
    test_df = tmp_df.filter(pl.col("tmp_id") > pl.col("num_sessions") * 0.8).select(
        "user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts"
    )

    tsd_stats = train_df.select(ts_delta=pl.col("ts_delta").explode()).select(
        ts_delta_max=pl.col("ts_delta").max(), ts_delta_min=pl.col("ts_delta").min()
    )
    disd_stats = train_df.select(dis_delta=pl.col("dis_delta").explode()).select(
        dis_delta_max=pl.col("dis_delta").max(), dis_delta_min=pl.col("dis_delta").min()
    )
    ts_delta_lower, ts_delta_upper = math.floor(
        tsd_stats["ts_delta_min"][0]
    ), math.ceil(tsd_stats["ts_delta_max"][0])
    dis_delta_lower, dis_delta_upper = math.floor(
        disd_stats["dis_delta_min"][0]
    ), math.ceil(disd_stats["dis_delta_max"][0])

    def generate_locseqs(df: pl.DataFrame) -> List[LocSeq]:
        return [
            LocSeq(
                seq=loc_seq,
                entity_id=user_id,
                ts_delta=ts_delta,
                dis_delta=dis_delta,
                start_ts=start_ts,
            )
            for user_id, loc_seq, ts_delta, dis_delta, start_ts in tqdm(
                df.iter_rows(), total=df.height
            )
        ]

    train_locseqs = generate_locseqs(train_df)
    LocSeqDataset.init_from_loc_seqs(train_locseqs).save(
        output_folder / "train_ds.parquet"
    )
    LocSeqDataset.init_from_loc_seqs(generate_locseqs(val_df)).save(
        output_folder / "val_ds.parquet"
    )
    LocSeqDataset.init_from_loc_seqs(generate_locseqs(test_df)).save(
        output_folder / "test_ds.parquet"
    )

    LocSeqTokenizer.build(loc_seqs=train_locseqs).save_pretrained(
        output_folder / "tokenizer.pkl"
    )
    Bucketizer(
        lower_bound=ts_delta_lower, upper_bound=ts_delta_upper, num_buckets=10
    ).save(output_folder / "ts_bucketizer.pkl")
    Bucketizer(
        lower_bound=dis_delta_lower, upper_bound=dis_delta_upper, num_buckets=10
    ).save(output_folder / "loc_bucketizer.pkl")
