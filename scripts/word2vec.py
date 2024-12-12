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
import logging
from typing import List

from gensim.models import Word2Vec
from tqdm import tqdm

from trajdl.datasets.arrow import LocSeqDataset
from trajdl.utils import get_num_cpus

# 设置日志以便我们可以看到训练过程的输出
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def load_sentences(ds_path: str) -> List[List[str]]:
    ds = LocSeqDataset.init_from_parquet(ds_path)
    return [seq.as_py() for seq in tqdm(ds.seq, desc="loading dataset")]


def train_word2vec(
    sentences,
    args,
):
    workers = get_num_cpus() if args.workers < 0 else args.workers

    # 训练Word2Vec模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        alpha=args.alpha,
        window=args.window,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
        sample=args.sample,
        seed=args.seed,
        workers=workers,
        min_alpha=args.min_alpha,
        sg=args.sg,
        hs=args.hs,
        negative=args.negative,
        ns_exponent=args.ns_exponent,
        cbow_mean=args.cbow_mean,
        epochs=args.epochs,
        null_word=args.null_word,
        compute_loss=True,
    )
    print("Final loss:", model.get_latest_training_loss())
    return model


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Train a Word2Vec model.")
    parser.add_argument(
        "--ds",
        type=str,
        required=True,
        help="train a word2vec model on given arrow dataset",
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=100,
        help="The dimensionality of the feature vectors",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.025, help="The initial learning rate"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="The maximum distance between the current and predicted word within a sentence",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Discard words with total frequency lower than this",
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=None,
        help="Limit the number of words in the vocabulary",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1e-3,
        help="Subsample all words with frequency greater than this.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for initializing weights"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Use these many worker threads to train the model",
    )
    parser.add_argument(
        "--min_alpha",
        type=float,
        default=0.0001,
        help="Learning rate will linearly drop to this value",
    )
    parser.add_argument(
        "--sg", type=int, choices=[0, 1], default=0, help="Skip-gram (1) or CBOW (0)"
    )
    parser.add_argument(
        "--hs",
        type=int,
        choices=[0, 1],
        default=0,
        help="Hierarchical softmax (1) or negative sampling (0)",
    )
    parser.add_argument(
        "--negative", type=int, default=5, help="Number of negative samples"
    )
    parser.add_argument(
        "--ns_exponent",
        type=float,
        default=0.75,
        help="Exponent for importance sampling",
    )
    parser.add_argument(
        "--cbow_mean",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, use mean of context word vectors for CBOW; default is 1",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of iterations (epochs) over the corpus",
    )
    parser.add_argument(
        "--null_word",
        type=int,
        default=0,
        help="If 1, include a special word representing null in the model",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file to save the trained model",
    )

    args = parser.parse_args()

    # 加载句子
    sentences = load_sentences(args.ds)

    # 训练模型
    model = train_word2vec(
        sentences,
        args,
    )

    # 保存模型
    model.save(args.output)
    print(f"Model trained and saved to {args.output}")


if __name__ == "__main__":
    main()
