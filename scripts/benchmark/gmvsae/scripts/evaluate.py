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
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm
from tqdm.contrib import tenumerate

from trajdl.algorithms.gmvsae import GMVSAE
from trajdl.common.enum import Mode
from trajdl.datasets import LocSeqDataset
from trajdl.datasets.modules.gmvsae import GMVSAEDataModule
from trajdl.utils import find_best_checkpoint


def auc_score(y_true, y_score):
    # shape of precision and recall is (N - 1,)
    # 这里用1减去，说明0是正常序列，1是异常序列，precision_recall_curve默认的pos_label是1
    precision, recall, _ = precision_recall_curve(1 - y_true, 1 - y_score)
    # float
    return auc(recall, precision)


def eval(
    trainer: L.Trainer,
    model: GMVSAE,
    folder: str,
    test_dataset_path: str,
    outlier_idx_path: str,
    tokenizer_path: str = "tokenizer.pkl",
):
    module = GMVSAEDataModule(
        tokenizer=os.path.join(folder, tokenizer_path),
        test_parquet_path=os.path.join(folder, test_dataset_path),
        val_batch_size=128,
        num_cpus=-1,
    )

    module.setup("predict")
    test_loader = module.test_dataloader()

    with torch.inference_mode():
        predictions = trainer.predict(model, test_loader)
    predictions = np.concatenate([i.detach().cpu().numpy() for i in predictions])

    # 异常样本的idx
    train_outlier_idx = np.load(os.path.join(folder, outlier_idx_path))
    ds: LocSeqDataset = LocSeqDataset.init_from_parquet(
        os.path.join(folder, test_dataset_path)
    )

    y_true = np.ones(shape=(len(ds)))
    for idx in train_outlier_idx:
        y_true[idx] = 0

    od_agg = defaultdict(list)
    for idx, loc_seq in tenumerate(ds.iter_as_seqs(), total=len(ds)):
        od_agg[(loc_seq.o, loc_seq.d)].append(idx)

    od_auc = {}
    for od, traj_indices in tqdm(od_agg.items()):
        od_true, od_pred = y_true[traj_indices], predictions[traj_indices]
        if od_true.sum() < od_true.shape[0]:
            od_auc[od] = auc_score(od_true, od_pred)

    return np.mean(list(od_auc.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", type=Path, default=Path("output") / "porto", help="数据集目录"
    )
    parser.add_argument(
        "--ckpt_folder", type=Path, required=True, help="train阶段的checkpoint folder"
    )
    args = parser.parse_args()

    folder = args.data_folder
    ckpt_path = args.ckpt_folder / find_best_checkpoint(
        args.ckpt_folder, is_maximizing=False
    )
    print(f"loading checkpoint from {ckpt_path}")

    model = GMVSAE.load_from_checkpoint(
        ckpt_path, map_location=torch.device("cpu")
    ).eval()
    model.set_mode(Mode.EVAL)

    trainer = L.Trainer(logger=False, enable_checkpointing=False)

    outlier_idx_path = "train_outlier_idx.npy"
    for test_dataset_path in [
        "train_outliers_perturb.parquet",
    ]:
        gmvsae_auc = eval(trainer, model, folder, test_dataset_path, outlier_idx_path)
        print(f"{test_dataset_path.split('.')[0]} auc score: {gmvsae_auc}")
