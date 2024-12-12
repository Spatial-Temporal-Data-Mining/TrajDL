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

import polars as pl

from trajdl.datasets.open_source.utils import load_dataset

val_ratio = 0.1
test_ratio = 0.1
output_folder = "output/hier/gowalla"
os.makedirs(output_folder, exist_ok=True)

df = load_dataset("gowalla", return_as="pl")

# split trajectory
# 按天切分，每个用户在一天的轨迹至少覆盖3个位置
r = (
    df.with_columns(pl.col("check_in_ts").dt.strftime("%Y%m%d").alias("ds"))
    .with_columns((pl.col("check_in_ts").dt.timestamp() // 1000000).alias("ts"))
    .group_by("user_id", "ds")
    .agg(
        pl.concat_list("lng", "lat").sort_by(pl.col("check_in_ts")).alias("traj"),
        pl.col("ts").sort_by(pl.col("check_in_ts")),
    )
    .filter(pl.col("traj").list.len() >= 3)
)

train_ratio = 1 - val_ratio - test_ratio
train_val_ratio = 1 - test_ratio

# split dataset into train, val, test datasets
# 将每个用户的日期从小到大排序，前中后分别作为训练、验证、测试集
result_df = (
    r.with_columns(pl.col("ds").count().over("user_id").alias("num_trajs"))
    .filter(pl.col("num_trajs") >= 100)
    .with_columns(pl.col("ds").rank().over("user_id").alias("rank"))
    .with_columns(
        pl.when(pl.col("rank") <= (pl.col("num_trajs") * train_ratio).ceil())
        .then(pl.lit(0))
        .when(pl.col("rank") <= (pl.col("num_trajs") * train_val_ratio).ceil())
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias("type")
    )
    .select(
        pl.col("traj"),
        pl.col("ts").alias("ts_seq"),
        pl.col("type"),
    )
)

# 统计一下数据集的平均长度
avg_length = (
    result_df.with_columns(pl.col("traj").list.len().alias("seq_len"))
    .select("seq_len")
    .mean()
    .item()
)
print(f"avg length: {avg_length}")

train_df = result_df.filter(pl.col("type") == 0).select("traj", "ts_seq")
val_df = result_df.filter(pl.col("type") == 1).select("traj", "ts_seq")
test_df = result_df.filter(pl.col("type") == 2).select("traj", "ts_seq")

train_df.write_parquet(os.path.join(output_folder, "train_ds.parquet"))
val_df.write_parquet(os.path.join(output_folder, "val_ds.parquet"))
test_df.write_parquet(os.path.join(output_folder, "test_ds.parquet"))
