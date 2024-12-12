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
import os
from typing import Union

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--exp", choices=["exp1", "exp2", "exp3"])
parser.add_argument("--rate", nargs="*", type=float)
args = parser.parse_args()

folder = args.folder
exp_prefix = args.exp

rate_list = args.rate
if rate_list is None:
    rate_list = [None]


def load_vec(exp_prefix: str, type_: str, rate: Union[int, float]):
    vec = np.load(os.path.join(folder, "eval", f"{exp_prefix}-{type_}-{rate}-vec.npy"))
    labels = np.load(
        os.path.join(folder, "eval", f"{exp_prefix}-{type_}-{rate}-label.npy")
    )
    return vec, labels


for rate in rate_list:
    vec_in_query, labels_in_query = load_vec(exp_prefix, "query", rate)
    vec_in_db, labels_in_db = load_vec(exp_prefix, "db", rate)

    db_sizes = range(20000, 110000, 20000) if exp_prefix == "exp1" else [100000]
    for db_size in db_sizes:
        # shape is (num_query, num_db)，算距离
        dis = euclidean_distances(vec_in_query, vec_in_db[:db_size])

        # (num_query, num_db), 看一下query在db里面的位置
        query_in_db_loc = labels_in_query.reshape(-1, 1) == labels_in_db[
            :db_size
        ].reshape(1, -1)

        # 这里要保证每个query一定在db里面都出现了
        assert query_in_db_loc.any(axis=1).all()

        # (num_query,), 看一下query在db里面测算的距离是多少
        query_dis_in_db = dis[query_in_db_loc]

        # (num_query,), 找当前的距离在db里面是第几小的，也就是看db里面有多少数字比当前dis小，以此得到rank
        rank = (query_dis_in_db.reshape(-1, 1) > dis).sum(axis=1)

        print(
            f"exp: {exp_prefix}, rate: {rate}, mean rank: {np.mean(rank)} with dbsize: {db_size}"
        )
