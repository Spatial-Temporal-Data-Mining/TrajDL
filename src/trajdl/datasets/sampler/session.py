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

"""
这个文件主要做一些session维度的sampler
"""

from typing import Dict, List

import numpy as np
import polars as pl
from torch.utils.data import Sampler

from ...common.enum import ArrowColName
from ..arrow.abstract import BaseArrowDataset
from .bucket import SeqInfo, generate_buckets_by_stats


class SessionSampler(Sampler):
    """
    这个sampler是为了ST-LSTM等算法设计的，需要在基础的数据集上根据用户的id进行聚合
    sampler在构建的时候，要拿到数据集，比如LocSeqDataset或者TrajectoryDataset

    """

    def __init__(
        self,
        ds: BaseArrowDataset,
        num_batches: int,
        batch_size: int,
        num_buckets: int = 10,
    ):
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._num_buckets = num_buckets

        # key是user_id，values是List[int]，里面是样本的下标，并且按照他们的start_ts升序排列了
        self.user_session_map = self.construct_user_samples_dict(ds=ds)

        self.all_users = list(self.user_session_map)
        user_maps = {user_id: idx for idx, user_id in enumerate(self.all_users)}
        # sample_idx是用户的id，而size就是这个用户的session个数
        stats = [
            SeqInfo(sample_idx=user_maps[user_id], size=len(value))
            for user_id, value in self.user_session_map.items()
        ]

        # 通过这些buckets，可以取得session个数相似的user
        self.buckets = generate_buckets_by_stats(stats=stats, num_buckets=num_buckets)

    @property
    def num_buckets(self):
        return self._num_buckets

    @property
    def batch_size(self):
        return self._batch_size

    def construct_user_samples_dict(self, ds: BaseArrowDataset) -> Dict[str, List[int]]:
        """
        这个方法是用来统计每个用户的序列的
        """
        df = ds.to_polars()

        user_map_dict = (
            df.select(ArrowColName.ENTITY_ID.value, ArrowColName.START_TS.value)
            .with_columns(pl.int_range(df.height).alias("sample_idx"))
            .group_by(ArrowColName.ENTITY_ID.value)
            .agg(
                pl.col("sample_idx")
                .sort_by(ArrowColName.START_TS.value)
                .alias("sample_indices")
            )
        )

        return dict(
            zip(
                user_map_dict[ArrowColName.ENTITY_ID.value].to_list(),
                user_map_dict["sample_indices"].to_list(),
            )
        )

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        """
        每次随机挑选一个bucket，然后从bucket里面挑选用户的idx，通过self.all_users映射到实际的用户id
        然后通过user_session_map，可以取到这个用户的对应的样本的list，然后从list里面再随机挑选一些下标，这个下标就是序列的下标
        """
        for _ in range(self._num_batches):
            random_bucket = self.buckets[np.random.choice(self.num_buckets)]
            user_indices = np.random.choice(random_bucket, self.batch_size)
            user_ids = (self.all_users[idx] for idx in user_indices)

            # 这里面每一项是一个用户的完整序列，里面的每个子项是用户的一个session，session里面的子项是ds的下标
            indices = []
            for user_id in user_ids:
                indices.extend(self.user_session_map[user_id])
            yield indices
