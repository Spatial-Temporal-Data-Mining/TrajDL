# Copyright 2026 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0, with the same licensing as
# the TrajDL project.

"""LightningCLI relative-path resolution (the path_params convention).

Algorithm classes self-describe which init_args are path files/directories
relative to run_dir via ``path_params: ClassVar[tuple[str, ...]]``;
:func:`resolve_declared_paths` applies prefix resolution according to that
declaration.  main.py no longer maintains a per-algorithm if-else list of key
names (the legacy whitelist is kept as a fallback during the gradual
migration).
"""

import importlib
from pathlib import Path
from typing import Any


def resolve_path(run_dir: Path, value: Any) -> Any:
    """Resolve relative paths against the run_dir prefix; absolute paths and other types are returned as-is."""
    if not isinstance(value, str):
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    return str(run_dir / path)


def load_class(class_path: str) -> type | None:
    """Import a class from its jsonargparse class_path; return None on failure (resolution is skipped)."""
    try:
        module_name, _, cls_name = class_path.rpartition(".")
        return getattr(importlib.import_module(module_name), cls_name)
    except Exception:
        return None


def class_path_params(cls: type) -> tuple[str, ...]:
    """Collect the path_params declared by a class (MRO included), merging with subclass declarations first and de-duplicating; empty when nothing is declared."""
    declared: list[str] = []
    for klass in cls.__mro__:
        for key in klass.__dict__.get("path_params", ()):
            if key not in declared:
                declared.append(key)
    return tuple(declared)


def _as_dict(obj: Any) -> dict | None:
    """Return a dict as-is; for attribute objects such as a jsonargparse Namespace return its ``__dict__``; None otherwise.

    The child nodes of LightningCLI's self.config are Namespaces (not dicts)
    whose attributes live in ``__dict__``; rewriting them in place propagates
    back to the very same object used for the later instantiation.
    """
    if isinstance(obj, dict):
        return obj
    d = getattr(obj, "__dict__", None)
    return d if isinstance(d, dict) else None


def resolve_declared_paths(cfg: Any, run_dir: Path) -> None:
    """Resolve relative paths inside init_args in place, according to the path_params declaration of the class_path class.

    cfg has the shape of a jsonargparse sub-config (with class_path /
    init_args) and accepts a dict or a Namespace; when there is no init_args,
    resolution happens directly on cfg.  Non-str values and absolute paths are
    left alone; undeclared keys are left alone too (the legacy whitelist
    fallback).
    """
    cfg_dict = _as_dict(cfg)
    if cfg_dict is None:
        return
    class_path = cfg_dict.get("class_path")
    if not class_path:
        return
    cls = load_class(class_path)
    if cls is None:
        return
    declared = class_path_params(cls)
    if not declared:
        return
    init_args = _as_dict(cfg_dict.get("init_args"))
    targets = [init_args] if init_args is not None else [cfg_dict]
    for target in targets:
        for key in declared:
            value = target.get(key)
            if isinstance(value, str):
                target[key] = resolve_path(run_dir, value)
