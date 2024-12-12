# Benchmarking

## TULER

### Configuration

Running on 8 vCPU, 32 GB memory, 1 x V100-16 GB

Hyperparameters configuration: Same as the code released by the authors, 1-layer GRU
The original code is built on TensorFlow 1.x

Steps of preprocessing dataset in original code:
1. Train Word2Vec

Steps of preprocessing dataset in TrajDL:
1. Train tokenizer
2. Train Word2Vec

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Preprocessing dataset time | 270s     | 367s    |
| Max CPU usage              | 100%     | 58.40%  |
| Max memory usage           | 4.70%    | 2.97%   |
| Max GPU utility            | 0%       | 0%      |
| Max GPU memory             | 0%       | 0%      |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Training time              | 2520s    | 701s    |
| Max CPU usage              | 27.10%   | 31.00%  |
| Max memory usage           | 6.35%    | 15.20%  |
| Max GPU utility            | 34.00%   | 7.00%   |
| Max GPU memory             | 95.40%   | 3.16%   |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Acc1                       | 27.68%   | 47.85%  |
| Acc5                       | 43.40%   | 64.91%  |
| Macro-F1                   | 23.39%   | 36.75%  |

### Some tips
- The original code uses `word2vec` to train Word2Vec models, while TrajDL uses Gensim. `word2vec` tool is more efficient than Gensim.
- When attempting to use Word2Vec to iterate 100 epochs in the original code and then train the TULER_GRU_1 model, the evaluation results were worse than those from TrajDL and the results published in the paper. 原因是：
作者的代码是在一个采样后的数据集，大约161万条序列里面抽取2万条做实验，2万条序列按用户分组，每个用户的前90%做训练集，后10%做测试集。然而使用的word2vec模型是在161万条序列上训练的，并不是2万里面的训练集，也不是2万的训练+测试集。TrajDL在161万条数据上训练的word2vec，效果可以达到一个非常好的效果，超越论文效果。

## T2VEC

### Configuration

Running on 8 vCPU, 32 GB memory, 1 x V100-16 GB

Hyperparameters configuration: Same as the code released by the authors

Steps of preprocessing dataset in original code:
1. Transform CSV dataset into H5
2. Generate Train, Val, Test dataset

Steps of preprocessing dataset in TrajDL:
1. Download Porto Taxi dataset (this test did not use cache)
2. Build tokenizers
3. Generate Train, Val, Test dataset
4. Train Word2Vec model

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Preprocessing dataset time | 4683s    | 325s    |
| Max CPU usage              | 25.10%   | 100.00% |
| Max memory usage           | 14.30%   | 63.8%   |
| Max GPU utility            | 0%       | 0%      |
| Max GPU memory             | 0%       | 0%      |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Training time              | 17040s   | 69161s  |
| Max CPU usage              | 12.50%   | 28.80%  |
| Max memory usage           | 41.40%   | 56.40%  |
| Max GPU utility            | 44.00%   | 68.00%  |
| Max GPU memory             | 39.30%   | 94.90%  |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| DB size 20k                | 2.437    | 0.845   |
| DB size 40k                | 3.669    | 1.556   |
| DB size 60k                | 5.030    | 2.255   |
| DB size 80k                | 6.725    | 3.215   |
| DB size 100k               | 8.153    | 4.068   |

| Code                       | Original | TrajDL        |
|----------------------------|----------|---------------|
| r1 0.2                     | 8.945    | 4.320±0.242   |
| r1 0.3                     | 8.859    | 4.766±0.236   |
| r1 0.4                     | 10.199   | 5.783±0.470   |
| r1 0.5                     | 12.226   | 7.657±0.590   |
| r1 0.6                     | 29.227   | 20.106±24.710 |

| Code                       | Original | TrajDL        |
|----------------------------|----------|---------------|
| r2 0.2                     | 8.703    | 4.673±0.720   |
| r2 0.3                     | 8.769    | 4.739±0.360   |
| r2 0.4                     | 7.836    | 4.559±0.777   |
| r2 0.5                     | 9.000    | 4.818±0.552   |
| r2 0.6                     | 6.799    | 4.871±0.551   |

### Some tips

- In the paper, experiments 2 and 3 introduced downsampling and distortion, which are operations with randomness. Therefore, when evaluating these two experiments, multiple trials should be conducted, and the mean and standard deviation should be recorded. In Experiment 2, we found that when setting r to 0.6, the evaluation results were unstable, which is similar to the surge phenomenon mentioned in the paper.

## GMVSAE

### Configuration

GM-VSAE10, Detecting detour anomalies (D), d = 3, $\alpha = 0.3$

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Preprocessing dataset time | 692s     | 290s    |
| Max CPU usage              | 12.50%   | 43.70%  |
| Max memory usage           | 18.10%   | 64.00%  |
| Max GPU utility            | 0%       | 0%      |
| Max GPU memory             | 0%       | 0%      |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Pre-training time          | 1525s    | 638s    |
| Max CPU usage              | 32.10%   | 82.00%  |
| Max memory usage           | 3.51%    | 17.80%  |
| Max GPU utility            | 37.00%   | 26.00%  |
| Max GPU memory             | 5.82%    | 10.80%  |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Training time              | 547s     | 663s    |
| Max CPU usage              | 33.10%   | 50.90%  |
| Max memory usage           | 3.36%    | 21.10%  |
| Max GPU utility            | 37.00%   | 26.00%  |
| Max GPU memory             | 3.45%    | 10.80%  |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| AUC of Detour              | 0.9819   | 0.9987  |

## HIER

## CTLE

## STRNN

### Configuration

Running on 8 vCPU, 32 GB memory, 1 x V100-16 GB

Hyperparameters configuration: Same as the code released by the authors.
The original code is built on PyTorch 0.4.0

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Preprocessing dataset time | 1485s    |     |
| Max CPU usage              | 12.50%   |   |
| Max memory usage           | 7.51%    |    |
| Max GPU utility            | 9.00%    | 0%      |
| Max GPU memory             | 6.48%    | 0%      |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Training time              | 2520s    | 701s    |
| Max CPU usage              | 27.10%   | 31.00%  |
| Max memory usage           | 6.35%    | 15.20%  |
| Max GPU utility            | 34%      | 7%      |
| Max GPU memory             | 95.40%   | 3.16%   |

| Code                       | Original | TrajDL  |
|----------------------------|----------|---------|
| Acc1                       | 27.68%   | 44.79%  |
| Acc5                       | 43.40%   | 64.53%  |
| Macro-F1                   | 23.39%   | 31.42%  |