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

"""Download open-source datasets from dataset hubs (hub-only distribution).

Contract: sources are HuggingFace and ModelScope-CN only, selected by
``TRAJDL_DATASET_SOURCE`` (default ``huggingface``) with no silent fallback.
"""

import os
from dataclasses import dataclass
from typing import Optional

from ...config import HUB_CACHE_DIR

HUGGINGFACE = "huggingface"
MODELSCOPE = "modelscope"
VALID_SOURCES = (HUGGINGFACE, MODELSCOPE)
SOURCE_ENV_VAR = "TRAJDL_DATASET_SOURCE"

# conf.py 中的 hub repo 均为 HF datasets 仓库，固定 repo_type
_HF_REPO_TYPE = "dataset"


@dataclass(frozen=True)
class HubFile:
    """Position of a single dataset file on a hub.

    The same original file must be uploaded to both hubs with identical bytes;
    ``revision`` must be pinned to a commit hash (never a moving branch).
    """

    repo_id: str
    filename: str
    revision: str


def resolve_source(source: Optional[str] = None) -> str:
    """Resolve the download source.

    Priority: explicit argument > ``TRAJDL_DATASET_SOURCE`` env var > ``huggingface``.
    """
    resolved = source if source is not None else os.environ.get(SOURCE_ENV_VAR, HUGGINGFACE)
    if resolved not in VALID_SOURCES:
        raise ValueError(
            f"Invalid dataset source: {resolved!r}. Valid sources: {', '.join(VALID_SOURCES)}"
        )
    return resolved


def _import_hf_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download datasets from HuggingFace. "
            "Install it with: pip install 'trajdl[hf]'"
        ) from e
    return hf_hub_download


def _import_ms_download():
    try:
        from modelscope.hub.file_download import dataset_file_download
    except ImportError as e:
        raise ImportError(
            "modelscope is required to download datasets from ModelScope. "
            "Install it with: pip install 'trajdl[ms]'"
        ) from e
    return dataset_file_download


def _link_into_place(local_path: str, target_path: str) -> None:
    """Place the downloaded file at ``target_path`` by symlinking into it.

    The hub SDK cache lives under ``~/.cache/trajdl/hub`` and the dataset cache
    under ``~/.cache/trajdl/dataset``, so both are always on the same filesystem.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.lexists(target_path):
        os.remove(target_path)
    os.symlink(local_path, target_path)


def download_hub_file(hub_file: HubFile, source: str, target_path: str) -> str:
    """Download the file referenced by ``hub_file`` from ``source`` and place it at ``target_path``.

    The hub SDK cache is redirected to ``~/.cache/trajdl/hub`` (instead of the
    SDK defaults) to avoid duplicating large files, and the file is then
    symlinked to ``target_path`` where SHA-256 validation happens.
    """
    if source == HUGGINGFACE:
        hf_hub_download = _import_hf_download()
        local_path = hf_hub_download(
            repo_id=hub_file.repo_id,
            filename=hub_file.filename,
            repo_type=_HF_REPO_TYPE,
            revision=hub_file.revision,
            cache_dir=HUB_CACHE_DIR,
        )
    elif source == MODELSCOPE:
        dataset_file_download = _import_ms_download()
        local_path = dataset_file_download(
            dataset_id=hub_file.repo_id,
            file_path=hub_file.filename,
            revision=hub_file.revision,
            cache_dir=HUB_CACHE_DIR,
        )
    else:
        raise ValueError(
            f"Invalid dataset source: {source!r}. Valid sources: {', '.join(VALID_SOURCES)}"
        )

    _link_into_place(local_path, target_path)
    return target_path


def build_download_error_message(
    source: str,
    hub_file: HubFile,
    expected_sha256: str,
    actual_sha256: Optional[str] = None,
    error: Optional[Exception] = None,
) -> str:
    """Build the standardized dataset download-failure message (source, repo,
    file, expected vs actual sha256)."""
    if actual_sha256 is not None:
        got = actual_sha256
    elif error is not None:
        got = f"unavailable ({error})"
    else:
        got = "unavailable"
    return (
        f"Dataset download failed from {source}.\n"
        f"- repo: {hub_file.repo_id}@{hub_file.revision}\n"
        f"- file: {hub_file.filename}\n"
        f"- expected sha256: {expected_sha256}, got: {got}\n"
        "Hint: behind a network firewall export TRAJDL_DATASET_SOURCE=modelscope "
        "(reachable from the domestic ModelScope mirror), "
        "or set HF_ENDPOINT=https://hf-mirror.com to use the HuggingFace mirror, "
        "or load offline via dataset.load(original_dataset_path=...). "
    )
