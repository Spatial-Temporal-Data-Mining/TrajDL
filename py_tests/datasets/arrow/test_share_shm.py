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

"""共享内存路径（share / load_from_shm）回归测试。

0.2.0 blocker（audit-questions Q1）：coord_system 化重构时 ``load_from_shm``
漏传 ``coord_system``，任何调用都必抛 TypeError。修复方案是 share() 把
save() 同款的 meta.json 侧车一并写入 /dev/shm，load_from_shm 按 load()
的同一解析函数读回 coord_system。本文件锁死该行为。
"""

import os

import pyarrow as pa
import pytest
from trajdl import CoordinateSystem, LocSeq, Trajectory
from trajdl.datasets.arrow.abstract import (
    FEATURE_TABLE_FILENAME,
    META_FILENAME,
)
from trajdl.datasets.arrow.locseq import LocSeqDataset
from trajdl.datasets.arrow.traj import TrajectoryDataset


def _make_locseq_dataset() -> LocSeqDataset:
    return LocSeqDataset.init_from_seqs(
        [
            LocSeq.from_list([f"loc_{i}_{j}" for j in range(3)], entity_id=str(i))
            for i in range(3)
        ]
    )


def _make_trajectory_dataset(coord_system) -> TrajectoryDataset:
    if coord_system == CoordinateSystem.WebMercator:
        # Web Mercator 是米制投影坐标，用有量级区分度的坐标值
        coords_at = lambda i, j: [  # noqa: E731
            12913060.0 + i * 100.0 + j * 10.0,
            4721671.0 + i * 100.0 + j * 10.0,
        ]
    else:
        coords_at = lambda i, j: [  # noqa: E731
            116.0 + i * 0.01 + j * 0.001,
            39.0 + i * 0.01 + j * 0.001,
        ]
    return TrajectoryDataset.init_from_seqs(
        [
            Trajectory.from_list(
                [coords_at(i, j) for j in range(4)], coord_system, entity_id=str(i)
            )
            for i in range(3)
        ]
    )


class TestShareLoadFromShm:
    """share → load_from_shm 往返：坐标系统必须经 meta 侧车随 handle 传递。"""

    def test_handle_contains_meta_sidecar(self):
        """shm_handle 必须包含 meta.json 侧车条目（与 save() 侧车同名）。"""
        ds = _make_trajectory_dataset(CoordinateSystem.Wgs84)
        with ds.share() as shm_handle:
            assert META_FILENAME in shm_handle

    def test_trajectory_roundtrip_wgs84(self):
        """Wgs84 轨迹数据集经共享内存往返后坐标系统与数据一致。"""
        ds = _make_trajectory_dataset(CoordinateSystem.Wgs84)
        with ds.share() as shm_handle:
            loaded = TrajectoryDataset.load_from_shm(shm_handle)
        assert loaded.coord_system == CoordinateSystem.Wgs84
        assert len(loaded) == len(ds)
        assert loaded.core_table.combine_chunks().equals(ds.core_table.combine_chunks())
        assert loaded.get_view(0).coord_system == CoordinateSystem.Wgs84

    def test_trajectory_roundtrip_web_mercator(self):
        """WebMercator 必须原样回读——证明 coord_system 真取自侧车而非写死。"""
        ds = _make_trajectory_dataset(CoordinateSystem.WebMercator)
        with ds.share() as shm_handle:
            loaded = TrajectoryDataset.load_from_shm(shm_handle)
        assert loaded.coord_system == CoordinateSystem.WebMercator
        assert len(loaded) == len(ds)

    def test_locseq_roundtrip_coord_system_none(self):
        """LocSeqDataset 无坐标系，往返后 coord_system 保持 None。"""
        ds = _make_locseq_dataset()
        with ds.share() as shm_handle:
            loaded = LocSeqDataset.load_from_shm(shm_handle)
        assert loaded.coord_system is None
        assert len(loaded) == len(ds)
        assert loaded.core_table.combine_chunks().equals(ds.core_table.combine_chunks())

    def test_roundtrip_with_feature_table(self):
        """feature_table 与 meta 侧车共存时，往返后 core/feature/coord_system 齐备。"""
        ds = _make_trajectory_dataset(CoordinateSystem.Wgs84)
        feature_table = pa.table(
            {"speed": [1.0, 2.5, 4.0]}, schema=pa.schema([("speed", pa.float64())])
        )
        ds.set_feature_table(feature_table)
        with ds.share() as shm_handle:
            assert FEATURE_TABLE_FILENAME in shm_handle
            loaded = TrajectoryDataset.load_from_shm(shm_handle)
        assert loaded.coord_system == CoordinateSystem.Wgs84
        assert loaded.feature_table is not None
        assert loaded.feature_table.column("speed").to_pylist() == [1.0, 2.5, 4.0]

    def test_handle_without_meta_is_rejected(self):
        """缺 meta 侧车的旧 handle 必须显式报错，禁止静默按无坐标系处理。"""
        ds = _make_trajectory_dataset(CoordinateSystem.Wgs84)
        with ds.share() as shm_handle:
            stale_handle = {k: v for k, v in shm_handle.items() if k != META_FILENAME}
            with pytest.raises(ValueError, match="meta.json"):
                TrajectoryDataset.load_from_shm(stale_handle)

    def test_context_exit_cleans_all_files(self):
        """上下文退出后共享内存文件（含 meta 侧车）必须全部清理。"""
        ds = _make_trajectory_dataset(CoordinateSystem.Wgs84)
        with ds.share() as shm_handle:
            paths = list(shm_handle.values())
            assert all(os.path.exists(p) for p in paths)
        assert not any(os.path.exists(p) for p in paths)

    def test_exception_in_share_context_still_cleans_all_files(self):
        """share 上下文内业务异常外传时，core/feature/meta 共享内存文件仍全部清理。"""

        class BoomOnPurpose(RuntimeError):
            pass

        ds = _make_trajectory_dataset(CoordinateSystem.Wgs84)
        ds.set_feature_table(
            pa.table(
                {"speed": [1.0, 2.5, 4.0]},
                schema=pa.schema([("speed", pa.float64())]),
            )
        )
        with pytest.raises(BoomOnPurpose):
            with ds.share() as shm_handle:
                # 前置：core + feature + meta 侧车三类文件都已登记且存在
                assert len(shm_handle) == 3
                paths = list(shm_handle.values())
                assert all(os.path.exists(p) for p in paths)
                raise BoomOnPurpose("boom")
        assert not any(os.path.exists(p) for p in paths)
