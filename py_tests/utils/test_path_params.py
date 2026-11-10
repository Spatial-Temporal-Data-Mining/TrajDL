# Copyright 2026 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0, with the same licensing as
# the TrajDL project.

"""path_params 类自描述路径参数约定的单测."""

from pathlib import Path
from typing import ClassVar

from trajdl.utils.path_params import (
    class_path_params,
    load_class,
    resolve_declared_paths,
    resolve_path,
)


class _FakeDataModule:
    path_params: ClassVar = ("data_path", "cache_dir")

    def __init__(self, data_path: str | None = None, cache_dir: str | None = None,
                 batch_size: int = 8):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size


def test_resolve_path_relative_vs_absolute():
    run_dir = Path("/tmp/run")
    assert resolve_path(run_dir, "data/x.pkl") == "/tmp/run/data/x.pkl"
    assert resolve_path(run_dir, "/abs/x.pkl") == "/abs/x.pkl"
    assert resolve_path(run_dir, 42) == 42


def test_class_path_params_via_mro():
    cls = load_class("trajdl.datasets.modules.unitraj.UnitrajDataModule")
    assert cls is not None
    assert class_path_params(cls) == ("data_path",)


def test_class_path_params_merges_across_mro():
    from trajdl.datasets.modules import TULERDataModule

    params = class_path_params(TULERDataModule)
    assert params[0] == "user_map"
    assert set(params) == {
        "user_map",
        "tokenizer",
        "train_ds_path",
        "val_ds_path",
        "test_ds_path",
    }


def test_benchmark_classes_declare_paths():
    from trajdl.algorithms.tale import TaleTrainingFramework
    from trajdl.algorithms.transfertraj.lightning import TransferTrajLightning
    from trajdl.datasets.modules import TrjSRDataModule

    assert set(class_path_params(TaleTrainingFramework)) == {
        "tokenizer",
        "forest_path",
        "pretrained_ckpt_folder",
    }
    assert set(class_path_params(TransferTrajLightning)) == {
        "poi_embed_path",
        "poi_coors_path",
        "road_embed_path",
        "road_coors_path",
    }
    assert set(class_path_params(TrjSRDataModule)) == {
        "train_hr_path",
        "train_lr_path",
        "val_hr_path",
        "val_lr_path",
    }


def test_resolve_declared_paths_base_class_declaration(tmp_path):
    cls_path = "trajdl.datasets.modules.ctle.CTLEDataModule"
    cfg = {
        "class_path": cls_path,
        "init_args": {
            "train_ds_path": "output/porto/train",
            "tokenizer": None,
            "val_ds_path": "/abs/val",
        },
    }
    resolve_declared_paths(cfg, tmp_path)
    assert cfg["init_args"]["train_ds_path"] == str(tmp_path / "output/porto/train")
    assert cfg["init_args"]["val_ds_path"] == "/abs/val"
    assert cfg["init_args"]["tokenizer"] is None


def test_resolve_declared_paths_resolves_only_declared(tmp_path):
    cfg = {
        "class_path": f"{_FakeDataModule.__module__}.{_FakeDataModule.__qualname__}",
        "init_args": {
            "data_path": "data/x.pkl",
            "batch_size": 16,
            "undeclared": "keep/relative.pkl",
        },
    }
    resolve_declared_paths(cfg, tmp_path)
    assert cfg["init_args"]["data_path"] == str(tmp_path / "data/x.pkl")
    assert cfg["init_args"]["batch_size"] == 16
    assert cfg["init_args"]["undeclared"] == "keep/relative.pkl"


def test_resolve_declared_paths_abs_untouched_and_no_classpath(tmp_path):
    cfg = {
        "class_path": f"{_FakeDataModule.__module__}.{_FakeDataModule.__qualname__}",
        "init_args": {"data_path": "/abs/y.pkl", "cache_dir": None},
    }
    resolve_declared_paths(cfg, tmp_path)
    assert cfg["init_args"]["data_path"] == "/abs/y.pkl"
    assert cfg["init_args"]["cache_dir"] is None

    # 无 class_path：安全跳过
    cfg2 = {"init_args": {"data_path": "rel.pkl"}}
    resolve_declared_paths(cfg2, tmp_path)
    assert cfg2["init_args"]["data_path"] == "rel.pkl"


def test_resolve_declared_paths_jsonargparse_namespace(tmp_path):
    # 云上冒烟暴露的回归：LightningCLI self.config 子节点是 Namespace 而非
    # dict，isinstance 守卫曾使解析整体静默跳过；__dict__ 原地改写须传导。
    from jsonargparse import Namespace

    init_args = Namespace(data_path="data/x.pkl", batch_size=16)
    cfg = Namespace(
        class_path=f"{_FakeDataModule.__module__}.{_FakeDataModule.__qualname__}",
        init_args=init_args,
    )
    resolve_declared_paths(cfg, tmp_path)
    assert cfg.init_args.data_path == str(tmp_path / "data/x.pkl")
    assert cfg.init_args.batch_size == 16


def test_resolve_declared_paths_namespace_flat_no_init_args(tmp_path):
    from jsonargparse import Namespace

    cfg = Namespace(
        class_path=f"{_FakeDataModule.__module__}.{_FakeDataModule.__qualname__}",
        data_path="data/y.pkl",
    )
    resolve_declared_paths(cfg, tmp_path)
    assert cfg.data_path == str(tmp_path / "data/y.pkl")


def test_resolve_declared_paths_non_dict_non_namespace_noop(tmp_path):
    resolve_declared_paths("not-a-mapping", tmp_path)
    resolve_declared_paths(None, tmp_path)
