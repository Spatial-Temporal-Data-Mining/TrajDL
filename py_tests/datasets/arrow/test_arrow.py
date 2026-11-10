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
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import pytest
from tqdm import trange
from trajdl import CoordinateSystem, LocSeq, LocSeqView, Trajectory, TrajectoryView
from trajdl.common.enum import ArrowColName
from trajdl.datasets.arrow import BaseArrowDataset
from trajdl.datasets.arrow.locseq import LocSeqDataset
from trajdl.datasets.arrow.schema import GET_ARROW_FROM_RUST_SEQUENCE
from trajdl.datasets.arrow.traj import TrajectoryDataset

from ...conftest import FOLDER


def construct_ds() -> List[LocSeqDataset]:
    test_ds = []
    for add_entity_id in [True, False]:
        for add_ts in [True, False]:
            seq_length = random.randint(0, 10)
            now_ts = int(datetime.now().timestamp())
            loc_seqs = [
                LocSeq.from_list(
                    [str(loc_idx) for loc_idx in range(seq_length)],
                    entity_id=str(i) if add_entity_id else None,
                    ts_seq=(
                        [now_ts + ts_idx for ts_idx in range(seq_length)]
                        if add_ts
                        else None
                    ),
                )
                for i in range(10)
            ]
            test_ds.append(LocSeqDataset.init_from_seqs(seqs=loc_seqs))

    return test_ds


def check_col_names(ds: BaseArrowDataset):
    column_names = ds.core_table.schema.names
    for col in ArrowColName:
        assert col.value in column_names


def test_arrow_datasets(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    """
    测试arrow数据集
    """
    test_ds = construct_ds()
    test_ds.append(test_locseq_dataset)
    test_ds.append(test_trajectory_dataset)

    for ds in [test_locseq_dataset, test_trajectory_dataset]:
        tmp_path = FOLDER / "test_ds"
        ds.save(tmp_path, force=True)

        assert (tmp_path / "core_table.parquet").is_file()
        test_ds.append(ds.__class__.load(tmp_path))

    for ds in test_ds:
        check_col_names(ds)

        indices = [[0, 1], [1, 2]]
        indices2 = [np.array(i, dtype=np.int64) for i in indices]
        indices3 = [0, 1, 2]
        indices4 = [np.int64(0), np.int32(1), 2]

        assert len(ds[0]) == 1
        assert len(ds.__getitems__([np.array([0, 1], dtype=np.int64)])) == 2
        assert len(ds.__getitems__(indices)) == 4
        assert len(ds.__getitems__(indices2)) == 4
        assert len(ds.__getitems__(indices3)) == 3
        assert len(ds.__getitems__(indices4)) == 3

        for new_ds in [
            ds[0],
            ds.__getitems__(indices),
            ds.__getitems__(indices2),
            ds.__getitems__(indices3),
            ds.__getitems__(indices4),
        ]:
            check_col_names(ds=new_ds)

        for t in ds.iter_as_seqs():
            if isinstance(ds, LocSeqDataset):
                assert isinstance(t, LocSeqView)
            elif isinstance(ds, TrajectoryDataset):
                assert isinstance(t, TrajectoryView)
            else:
                raise RuntimeError("Test cases do not cover.")


def test_init_from_different_types(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    """
    测试数据集从不同类型的数据进行初始化
    """
    for ds in [test_locseq_dataset, test_trajectory_dataset]:
        df = ds.to_polars()[0]
        assert isinstance(df, pl.DataFrame)
        assert df.height == len(ds)

        for col_enum in ArrowColName:
            arrow_table = df.drop(col_enum.value).to_arrow()
            with pytest.warns(
                RuntimeWarning, match=r"Field .* does not exist in the input table."
            ):
                tmp_ds = ds.__class__.init_from_arrow(
                    arrow_table, coord_system=ds.coord_system
                )
                tmp_table_col = tmp_ds.core_table.column(col_enum.value)
                for t in tmp_table_col:
                    assert t.as_py() is None

        ds.__class__.init_from_table(df, coord_system=ds.coord_system)
        ds.__class__.init_from_table(df.to_pandas(), coord_system=ds.coord_system)
        ds.__class__.init_from_table(df.to_arrow(), coord_system=ds.coord_system)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "`table` should be an instance of {'pyarrow.Table', 'polars.DataFrame', 'pandas.DataFrame'}"
            ),
        ):
            ds.__class__.init_from_table(df.to_dicts(), coord_system=ds.coord_system)


def test_arrow_datasets_properties(
    test_locseq_dataset: LocSeqDataset, test_trajectory_dataset: TrajectoryDataset
):
    def check_func(
        ds: BaseArrowDataset,
        seqs: List[LocSeq | Trajectory],
        attr_name: str,
        idx: int,
        all_close: bool,
    ) -> bool:
        property_from_arrow = getattr(ds, attr_name)[idx].as_py()
        property_from_rust = GET_ARROW_FROM_RUST_SEQUENCE[attr_name](seqs[idx])
        print(attr_name, type(property_from_arrow), type(property_from_rust))
        print(property_from_rust)

        if property_from_rust is None:
            return property_from_arrow is None

        if all_close:
            return np.allclose(property_from_arrow, property_from_rust)
        else:
            return property_from_arrow == property_from_rust

    for ds in [test_locseq_dataset, test_trajectory_dataset]:
        ds = ds.__getitems__(range(100))
        seqs = list(ds.iter_as_seqs())

        for idx in trange(len(seqs), desc="iter ds and assert equal"):
            if isinstance(ds, TrajectoryDataset):
                assert check_func(ds, seqs, ArrowColName.SEQ.value, idx, all_close=True)
            elif isinstance(ds, LocSeqDataset):
                assert check_func(
                    ds, seqs, ArrowColName.SEQ.value, idx, all_close=False
                )
            else:
                raise RuntimeError("Test cases do not cover.")

            assert check_func(
                ds, seqs, ArrowColName.ENTITY_ID.value, idx, all_close=False
            )
            assert check_func(
                ds, seqs, ArrowColName.START_TS.value, idx, all_close=False
            )
            assert check_func(ds, seqs, ArrowColName.TS_SEQ.value, idx, all_close=True)


class TestTrajectoryViewCoordinateSystem:
    """测试 TrajectoryView 的坐标系功能"""

    @pytest.fixture
    def trajectory_dataset(self):
        """创建测试数据集"""
        trajs = [
            Trajectory.from_list(
                [[116.0 + i * 0.1, 39.0 + i * 0.1]], CoordinateSystem.Wgs84
            )
            for i in range(3)
        ]
        return TrajectoryDataset.init_from_seqs(trajs)

    def test_view_default_coord_system(self, trajectory_dataset):
        """测试视图默认坐标系"""
        view = trajectory_dataset.get_view(0)
        assert view.coord_system == CoordinateSystem.Wgs84

    def test_view_with_coord_system(self, trajectory_dataset):
        """测试带坐标系参数的视图构造"""
        view_merc = TrajectoryView(
            trajectory_dataset.core_table, 0, coord_system=CoordinateSystem.WebMercator
        )
        assert view_merc.coord_system == CoordinateSystem.WebMercator

    def test_view_to_web_mercator(self, trajectory_dataset):
        """测试视图转 WebMercator"""
        view = trajectory_dataset.get_view(0)
        traj_merc = view.to_web_mercator()

        assert traj_merc.coord_system == CoordinateSystem.WebMercator
        assert view.coord_system == CoordinateSystem.Wgs84  # 原视图不变

    def test_view_to_wgs84(self, trajectory_dataset):
        """测试视图转 Wgs84"""
        view_merc = TrajectoryView(
            trajectory_dataset.core_table, 0, coord_system=CoordinateSystem.WebMercator
        )
        traj_wgs = view_merc.to_wgs84()

        assert traj_wgs.coord_system == CoordinateSystem.Wgs84


class TestTrajectoryDatasetCoordinateSystem:
    """测试 TrajectoryDataset 的坐标系功能"""

    def test_init_from_seqs_default_coord_system(self):
        """测试默认坐标系"""
        trajs = [Trajectory.from_list([[116.0, 39.0]], CoordinateSystem.Wgs84)]
        ds = TrajectoryDataset.init_from_seqs(trajs)
        assert ds.coord_system == CoordinateSystem.Wgs84

    def test_init_from_seqs_infers_coord_system_from_inputs(self):
        """坐标系完全由输入轨迹推断，无需用户传参"""
        trajs = [
            Trajectory.from_list(
                [[12913060.0, 4721671.0]], CoordinateSystem.WebMercator
            )
        ]
        ds = TrajectoryDataset.init_from_seqs(trajs)
        assert ds.coord_system == CoordinateSystem.WebMercator

    def test_save_load_coord_system(self):
        """测试保存和加载坐标系"""
        trajs = [
            Trajectory.from_list([[116.0 + i, 39.0 + i]], CoordinateSystem.WebMercator)
            for i in range(3)
        ]
        ds = TrajectoryDataset.init_from_seqs(trajs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ds"
            ds.save(path)

            # 检查 meta.json 存在且包含版本信息
            assert (path / "meta.json").exists()
            import json

            with open(path / "meta.json") as f:
                meta = json.load(f)
            assert "version" in meta
            assert meta["version"] is not None

            # 加载并验证坐标系
            loaded = TrajectoryDataset.load(path)
            assert loaded.coord_system == CoordinateSystem.WebMercator

    def test_save_load_backward_compatibility(self):
        """测试向后兼容（没有 meta.json 的情况）应该报错"""
        trajs = [Trajectory.from_list([[116.0, 39.0]], CoordinateSystem.Wgs84)]
        ds = TrajectoryDataset.init_from_seqs(trajs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ds"
            ds.save(path)

            # 删除 meta.json 模拟旧版本数据，应该报错
            import os

            os.remove(path / "meta.json")

            with pytest.raises(FileNotFoundError):
                TrajectoryDataset.load(path)

    def test_dataset_to_web_mercator(self):
        """测试数据集转 WebMercator"""
        trajs = [
            Trajectory.from_list([[116.0 + i, 39.0 + i]], CoordinateSystem.Wgs84)
            for i in range(3)
        ]
        ds = TrajectoryDataset.init_from_seqs(trajs)

        ds_merc = ds.to_web_mercator()
        assert ds_merc.coord_system == CoordinateSystem.WebMercator
        assert ds.coord_system == CoordinateSystem.Wgs84  # 原数据集不变

    def test_dataset_to_web_mercator_already_mercator(self):
        """测试已经是 WebMercator 时调用 to_web_mercator"""
        trajs = [
            Trajectory.from_list(
                [[12913060.0, 4721671.0]], CoordinateSystem.WebMercator
            )
        ]
        ds = TrajectoryDataset.init_from_seqs(trajs)

        ds_converted = ds.to_web_mercator()
        assert ds_converted.coord_system == CoordinateSystem.WebMercator

    def test_dataset_to_wgs84(self):
        """测试数据集转 Wgs84"""
        trajs = [
            Trajectory.from_list(
                [[12913060.0, 4721671.0]], CoordinateSystem.WebMercator
            )
        ]
        ds = TrajectoryDataset.init_from_seqs(trajs)

        ds_wgs = ds.to_wgs84()
        assert ds_wgs.coord_system == CoordinateSystem.Wgs84

    def test_dataset_round_trip_conversion(self):
        """测试数据集往返转换"""
        trajs = [
            Trajectory.from_list(
                [[116.0 + i * 0.1, 39.0 + i * 0.1]], CoordinateSystem.Wgs84
            )
            for i in range(5)
        ]
        ds = TrajectoryDataset.init_from_seqs(trajs)

        # Wgs84 -> WebMercator -> Wgs84
        ds_round_trip = ds.to_web_mercator().to_wgs84()

        assert ds_round_trip.coord_system == CoordinateSystem.Wgs84
        assert len(ds_round_trip) == len(ds)

        # 验证坐标精度
        for i in range(len(ds)):
            orig_view = ds.get_view(i)
            rt_view = ds_round_trip.get_view(i)
            orig_seq = orig_view.seq
            rt_seq = rt_view.seq
            np.testing.assert_array_almost_equal(orig_seq[0], rt_seq[0], decimal=10)

    def test_get_view_inherits_coord_system(self):
        """测试 get_view 继承数据集的坐标系"""
        trajs = [
            Trajectory.from_list(
                [[12913060.0, 4721671.0]], CoordinateSystem.WebMercator
            )
        ]
        ds = TrajectoryDataset.init_from_seqs(trajs)

        view = ds.get_view(0)
        assert view.coord_system == CoordinateSystem.WebMercator


class TestInitFromSeqsChunkSize:
    """测试 init_from_seqs 的 chunk_size 参数在不同 seqs 长度下的正确性。

    关键边界：
    - len(seqs) < chunk_size：仅有一次尾部 flush
    - len(seqs) == chunk_size：仅有一次满 chunk flush，无尾部
    - len(seqs) > chunk_size：多个满 chunk + 一次尾部 flush
    - 还需覆盖：len(seqs) 为 chunk_size 的整数倍（无尾部）、空 seqs、chunk_size=1
    """

    @staticmethod
    def _build_locseqs(num_seqs: int, base_ts: int) -> List[LocSeq]:
        """构造可重复、有区分度的 LocSeq 列表，便于逐条 assert。"""
        seqs: List[LocSeq] = []
        for i in range(num_seqs):
            seq_length = (i % 5) + 1  # 1 ~ 5，避免空序列
            seqs.append(
                LocSeq.from_list(
                    [f"loc_{i}_{j}" for j in range(seq_length)],
                    entity_id=f"entity_{i}",
                    ts_seq=[base_ts + i * 100 + j for j in range(seq_length)],
                )
            )
        return seqs

    @staticmethod
    def _build_trajectories(num_seqs: int, base_ts: int) -> List[Trajectory]:
        """构造可重复、有区分度的 Trajectory 列表。

        Note: ``Trajectory.from_list`` 不支持 ``ts_seq`` 参数，时间戳信息
        无法在构造时附带，因此 ``ts_seq`` 列在数据集中均为 None。这里仍
        保留 ``base_ts`` 形参以便未来 API 扩展时直接启用。
        """
        del base_ts  # 当前未使用，仅占位
        trajs: List[Trajectory] = []
        for i in range(num_seqs):
            seq_length = (i % 5) + 1
            coords = [
                [116.0 + i * 0.001 + j * 0.0001, 39.0 + i * 0.001 + j * 0.0001]
                for j in range(seq_length)
            ]
            trajs.append(
                Trajectory.from_list(
                    coords,
                    CoordinateSystem.Wgs84,
                    entity_id=f"entity_{i}",
                )
            )
        return trajs

    @staticmethod
    def _assert_locseq_dataset_equal(ds: LocSeqDataset, seqs: List[LocSeq]) -> None:
        assert len(ds) == len(seqs)
        for idx, seq in enumerate(seqs):
            assert ds.seq[idx].as_py() == GET_ARROW_FROM_RUST_SEQUENCE[
                ArrowColName.SEQ.value
            ](seq)
            assert ds.entity_id[idx].as_py() == GET_ARROW_FROM_RUST_SEQUENCE[
                ArrowColName.ENTITY_ID.value
            ](seq)
            assert ds.start_ts[idx].as_py() == GET_ARROW_FROM_RUST_SEQUENCE[
                ArrowColName.START_TS.value
            ](seq)
            ts_arrow = ds.ts_seq[idx].as_py()
            ts_rust = GET_ARROW_FROM_RUST_SEQUENCE[ArrowColName.TS_SEQ.value](seq)
            assert ts_arrow == ts_rust

    @staticmethod
    def _assert_trajectory_dataset_equal(
        ds: TrajectoryDataset, trajs: List[Trajectory]
    ) -> None:
        assert len(ds) == len(trajs)
        for idx, traj in enumerate(trajs):
            seq_arrow = np.array(ds.seq[idx].as_py())
            seq_rust = np.array(
                GET_ARROW_FROM_RUST_SEQUENCE[ArrowColName.SEQ.value](traj)
            )
            np.testing.assert_allclose(seq_arrow, seq_rust)
            assert ds.entity_id[idx].as_py() == GET_ARROW_FROM_RUST_SEQUENCE[
                ArrowColName.ENTITY_ID.value
            ](traj)
            assert ds.start_ts[idx].as_py() == GET_ARROW_FROM_RUST_SEQUENCE[
                ArrowColName.START_TS.value
            ](traj)
            assert ds.ts_seq[idx].as_py() == GET_ARROW_FROM_RUST_SEQUENCE[
                ArrowColName.TS_SEQ.value
            ](traj)

    # 使用一个相对小但有意义的 chunk_size 以加速测试
    CHUNK_SIZE = 16

    @pytest.mark.parametrize(
        "num_seqs,expected_num_batches",
        [
            (5, 1),  # < chunk_size：仅尾部 flush
            (CHUNK_SIZE, 1),  # == chunk_size：仅一次满 chunk
            (CHUNK_SIZE * 2, 2),  # 整数倍：两个满 chunk，无尾部
            (CHUNK_SIZE + 3, 2),  # > chunk_size：1 个满 chunk + 1 个尾部
            (CHUNK_SIZE * 3 + 7, 4),  # 多个满 chunk + 尾部
        ],
    )
    def test_locseq_chunk_size_boundaries(
        self, num_seqs: int, expected_num_batches: int
    ):
        """LocSeqDataset.init_from_seqs 在不同 seqs 长度下应构建出准确的数据集。"""
        base_ts = int(datetime(2024, 1, 1).timestamp())
        seqs = self._build_locseqs(num_seqs=num_seqs, base_ts=base_ts)

        ds = LocSeqDataset.init_from_seqs(seqs=seqs, chunk_size=self.CHUNK_SIZE)

        # 验证内部 RecordBatch 切分符合预期（验证 chunk_size 真实生效）
        assert ds.core_table.column(ArrowColName.SEQ.value).num_chunks == (
            expected_num_batches
        )
        # 验证数据完整性
        self._assert_locseq_dataset_equal(ds, seqs)

    @pytest.mark.parametrize(
        "num_seqs,expected_num_batches",
        [
            (5, 1),
            (CHUNK_SIZE, 1),
            (CHUNK_SIZE * 2, 2),
            (CHUNK_SIZE + 3, 2),
            (CHUNK_SIZE * 3 + 7, 4),
        ],
    )
    def test_trajectory_chunk_size_boundaries(
        self, num_seqs: int, expected_num_batches: int
    ):
        """TrajectoryDataset.init_from_seqs 在不同 seqs 长度下应构建出准确的数据集。"""
        base_ts = int(datetime(2024, 1, 1).timestamp())
        trajs = self._build_trajectories(num_seqs=num_seqs, base_ts=base_ts)

        ds = TrajectoryDataset.init_from_seqs(
            seqs=trajs,
            chunk_size=self.CHUNK_SIZE,
        )

        assert ds.core_table.column(ArrowColName.SEQ.value).num_chunks == (
            expected_num_batches
        )
        self._assert_trajectory_dataset_equal(ds, trajs)

    def test_chunk_size_one(self):
        """chunk_size=1 时每条 seq 一个 RecordBatch，结果仍然正确。"""
        base_ts = int(datetime(2024, 1, 1).timestamp())
        seqs = self._build_locseqs(num_seqs=7, base_ts=base_ts)

        ds = LocSeqDataset.init_from_seqs(seqs=seqs, chunk_size=1)

        assert ds.core_table.column(ArrowColName.SEQ.value).num_chunks == 7
        self._assert_locseq_dataset_equal(ds, seqs)

    def test_chunk_size_larger_than_default(self):
        """chunk_size 远大于 seqs 长度时，等价于一次性构建。"""
        base_ts = int(datetime(2024, 1, 1).timestamp())
        seqs = self._build_locseqs(num_seqs=10, base_ts=base_ts)

        ds = LocSeqDataset.init_from_seqs(seqs=seqs, chunk_size=10_000)

        assert ds.core_table.column(ArrowColName.SEQ.value).num_chunks == 1
        self._assert_locseq_dataset_equal(ds, seqs)

    def test_generator_input_with_chunk_size(self):
        """seqs 为 generator 时（无法预知长度），chunk_size 仍能正确工作。"""
        base_ts = int(datetime(2024, 1, 1).timestamp())
        num_seqs = self.CHUNK_SIZE + 5  # 大于 chunk_size，存在尾部 flush
        materialized = self._build_locseqs(num_seqs=num_seqs, base_ts=base_ts)

        def seq_gen():
            for s in materialized:
                yield s

        ds = LocSeqDataset.init_from_seqs(seqs=seq_gen(), chunk_size=self.CHUNK_SIZE)

        assert ds.core_table.column(ArrowColName.SEQ.value).num_chunks == 2
        self._assert_locseq_dataset_equal(ds, materialized)

    def test_chunk_size_consistency_across_values(self):
        """同一份 seqs 在不同 chunk_size 下应构建出语义上完全一致的数据集。"""
        base_ts = int(datetime(2024, 1, 1).timestamp())
        seqs = self._build_locseqs(num_seqs=37, base_ts=base_ts)

        datasets = [
            LocSeqDataset.init_from_seqs(seqs=seqs, chunk_size=cs)
            for cs in (1, 8, 16, 37, 64, 8192)
        ]

        # 以第一个为基准，combine_chunks 后整张 table 应完全一致
        baseline = datasets[0].core_table.combine_chunks()
        for other in datasets[1:]:
            assert other.core_table.combine_chunks().equals(baseline)

    def test_invalid_chunk_size_raises(self):
        """chunk_size <= 0 应该抛出 ValueError。"""
        base_ts = int(datetime(2024, 1, 1).timestamp())
        seqs = self._build_locseqs(num_seqs=3, base_ts=base_ts)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            LocSeqDataset.init_from_seqs(seqs=seqs, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            LocSeqDataset.init_from_seqs(seqs=seqs, chunk_size=-1)
