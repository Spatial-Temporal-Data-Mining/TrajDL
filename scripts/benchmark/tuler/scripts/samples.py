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
import pickle
from pathlib import Path
from typing import Dict, List

from tqdm import trange

from trajdl.datasets import LocSeq, LocSeqDataset
from trajdl.tokenizers.locseq import LocSeqTokenizer


def build_tokenizer(path: str) -> LocSeqTokenizer:
    def loc_seq_iter():
        with open(path, "r") as f:
            for line in f:
                yield line.strip().split()[1:]

    return LocSeqTokenizer.build(loc_seqs=list(loc_seq_iter()))


def read_data(path: str):
    loc_seqs: List[LocSeq] = []
    user_id: List[str] = []

    # 读取作者处理的数据集
    with open(path, "r") as f:
        num_samples = 0
        for line in f:
            sample = line.strip().split()
            user_id.append(sample[0])
            loc_seqs.append(LocSeq(sample[1:], entity_id=sample[0]))
            num_samples += 1
        print(f"num samples: {num_samples}")

    # Test 98481
    training_sample_size = 20000
    loc_seqs = loc_seqs[:training_sample_size]
    user_id = user_id[:training_sample_size]
    user_list = list(set(user_id))

    avg_len = sum(len(loc_seq) for loc_seq in loc_seqs) / len(loc_seqs)
    print(f"Average Length of loc seq: {avg_len}")
    print(f"num users: {len(user_list)}")

    # 这段的原理是遍历每个用户的轨迹，直到最后一条轨迹。然后将最后10%的轨迹序列取出作为测试集，前面的90%的轨迹序列作为训练集
    train_loc_seqs: List[LocSeq] = []
    test_loc_seqs: List[LocSeq] = []

    flag = 0
    count = 0

    test_set_ratio = 0.1  # 10% for test
    for idx in trange(len(loc_seqs)):
        if user_id[idx] != flag or idx == len(loc_seqs) - 1:
            # split data
            if count > 1:
                split_line = idx - math.ceil(count * test_set_ratio)
                test_loc_seqs.extend(loc_seqs[split_line:idx])
                train_loc_seqs.extend(loc_seqs[int(idx - count) : split_line])
            else:
                train_loc_seqs.extend(loc_seqs[int(idx - count) : int(idx)])
            count = 1
            flag = user_id[idx]
        else:
            count += 1

    print(f"Train Size: {len(train_loc_seqs)} Test Size: {len(test_loc_seqs)}")

    user_map: Dict[str, int] = {i: idx for idx, i in enumerate(user_list)}

    return (
        train_loc_seqs,
        test_loc_seqs,
        user_map,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["gowalla", "brightkite"], required=True
    )
    parser.add_argument("--data_path", type=Path, required=True, help="数据的路径")
    parser.add_argument("--output_folder", type=Path, default="output")
    args = parser.parse_args()

    output_folder = args.output_folder / args.dataset
    print(output_folder)

    source_file = args.data_path
    tokenizer = build_tokenizer(source_file)
    tokenizer.save_pretrained(output_folder / "tokenizer.pkl")

    train_loc_seqs, test_loc_seqs, user_map = read_data(source_file)
    with open(output_folder / "user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    print(f"num users: {len(user_map)}")

    # save dataset
    train_ds = LocSeqDataset.init_from_loc_seqs(train_loc_seqs)
    train_ds.save(output_folder / "train_ds.parquet")
    test_ds = LocSeqDataset.init_from_loc_seqs(test_loc_seqs)
    test_ds.save(output_folder / "test_ds.parquet")

    all_loc_list: List[LocSeq] = []
    with open(source_file, "r") as f:
        for line in f:
            all_loc_list.append(LocSeq(seq=line.split()[1:]))
    LocSeqDataset.init_from_loc_seqs(all_loc_list).save(
        output_folder / "full_ds.parquet"
    )
