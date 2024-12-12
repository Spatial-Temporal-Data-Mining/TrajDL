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

from typing import Set, Tuple

import numpy as np
import torch

from ..utils import tiny_value_of_dtype


def _choice(num_words: int, num_samples: int, seed=None) -> Tuple[np.ndarray, int]:
    """
    在vocab里面做不放回采样
    Chooses `num_samples` samples without replacement from [0, ..., num_words).
    Returns a tuple (samples, num_tries).
    """
    num_tries = 0
    num_chosen = 0

    def get_buffer() -> np.ndarray:
        r"""
        实际上是$ \exp^{\alpha \log^{ \vert V \vert + 1 }} - 1$
        $\alpha$是[0, 1)之间的均匀分布。
        这个分布会使得采样时大部分样本集中在idx比较小的word上，idx比较大的word采样得到的概率很低
        """
        log_samples = np.random.random(num_samples) * np.log(num_words + 1)
        samples = np.exp(log_samples).astype("int64") - 1
        return np.clip(samples, a_min=0, a_max=num_words - 1)

    sample_buffer = get_buffer()
    buffer_index = 0
    samples: Set[int] = set()

    # 做不放回采样，并且要保证一定要采集到num_samples个元素，并且要记录取了多少次才取出这些元素
    while num_chosen < num_samples:
        num_tries += 1
        # choose sample
        sample_id = sample_buffer[buffer_index]
        if sample_id not in samples:
            samples.add(sample_id)
            num_chosen += 1

        buffer_index += 1
        if buffer_index == num_samples:
            # Reset the buffer
            sample_buffer = get_buffer()
            buffer_index = 0

    return np.array(list(samples)), num_tries


class SampledSoftmaxLoss(torch.nn.Module):
    """
    Based on the default log_uniform_candidate_sampler in tensorflow.
    Parameters
    ----------
    weights: torch.Tensor

    bias: torch.Tensor

    num_words: int

    num_samples: int

    reduction: str

    device

    use_sampled_softmax_in_eval: bool, optional
        Whether or not using sampled softmax in eval mode, default False.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        num_words: int,
        num_samples: int,
        reduction: str = "mean",
        device=None,
        use_sampled_softmax_in_eval: bool = False,
    ) -> None:
        super().__init__()

        assert num_samples < num_words
        self.use_sampled_softmax_in_eval = use_sampled_softmax_in_eval

        self.choice_func = _choice

        assert len(weights.shape) == 2, "weights should be a 2-D tensor"
        self.w = weights

        # torch.nn.functional.embedding can only be applied on 2-D tensor
        assert (
            len(bias.shape) == 2 and bias.shape[1] == 1
        ), "bias should be a 2-D tensor, and the size of the second axis should be 1"
        self.b = bias

        self._num_samples = num_samples
        self._num_words = num_words
        self._reduction = reduction
        self._device = device
        self.eval_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.initialize_num_words(self._num_words)

    def initialize_num_words(self, num_words: int):
        self._log_num_words_p1 = np.log1p(num_words)

        # compute the probability of each sampled id
        words_seq = np.arange(num_words)
        self._probs = (
            np.log(words_seq + 2) - np.log(words_seq + 1)
        ) / self._log_num_words_p1

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.LongTensor,
        mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        # embeddings is size (n, embedding_dim)
        # targets is (n_words, ) with the index of the actual target
        # when tieing weights, target_token_embedding is required.
        # it is size (n_words, embedding_dim)
        # returns log likelihood loss (batch_size, )
        # Does not do any count normalization / divide by batch size

        if embeddings.shape[0] == 0:
            # empty batch
            return torch.tensor(0.0).to(embeddings.device)

        if self.training or self.use_sampled_softmax_in_eval:
            batch_loss = self._forward_train(embeddings, targets, mask)
        else:
            batch_loss = self._forward_eval(embeddings, targets, mask)

        if self._reduction == "mean":
            return (
                batch_loss.sum() / mask.sum() if mask is not None else batch_loss.sum()
            )
        elif self._reduction == "sum":
            return batch_loss.sum()
        else:
            return batch_loss

    def _forward_train(
        self,
        embeddings: torch.Tensor,
        targets: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        embeddings: shape is (B, C)
        targets: shape is (B,)
        masks: shape is (B,)，表示当前的embedding是否需要纳入损失的计算，因为embeddings和targets里面可能存在padding
            因为要实现负采样，所以如果一个样本不需要计算损失，那么负采样的部分也不应该计算，为了实现起来的方便，需要在计算损失的部分剔除掉这些元素
            在当前实现方式里面，将这些元素的logits设置为-min，使得softmax之后的值为0，不贡献损失。
        """

        # want to compute (n, n_samples + 1) array with the log
        # probabilities where the first index is the true target
        # and the remaining ones are the the negative samples.
        # then we can just select the first column

        (
            sampled_ids,
            target_expected_count,
            sampled_expected_count,
        ) = self.log_uniform_candidate_sampler(targets, choice_func=self.choice_func)

        # Get the softmax weights (so we can compute logits)
        # shape is (B + num_samples)
        all_ids = torch.cat([targets, sampled_ids], dim=0)

        # (B + num_samples, C)
        all_w = torch.nn.functional.embedding(all_ids, self.w)
        all_b = torch.nn.functional.embedding(all_ids, self.b).squeeze(1)

        batch_size = targets.shape[0]

        true_w = all_w[:batch_size, :]
        sampled_w = all_w[batch_size:, :]

        true_b = all_b[:batch_size]
        sampled_b = all_b[batch_size:]

        # compute the logits and remove log expected counts
        # shape is (batch_size, 1)
        true_logits = (
            (true_w * embeddings).sum(dim=1)
            + true_b
            - torch.log(
                target_expected_count + tiny_value_of_dtype(target_expected_count.dtype)
            )
        ).unsqueeze(dim=1)

        # [batch_size, n_samples]
        sampled_logits = (
            torch.matmul(embeddings, sampled_w.t())
            + sampled_b
            - torch.log(
                sampled_expected_count
                + tiny_value_of_dtype(sampled_expected_count.dtype)
            )
        )

        # remove true labels -- we will take softmax, so set the sampled logits of true values to a large negative number
        # 因为是针对一个batch的label，共用一组负样本，所以每个负样本都要去与targets里面进行对比，形成一个(batch_size, n_samples)的矩阵
        true_in_sample_mask = sampled_ids == targets.unsqueeze(1)

        # 将不小心采错的负样本的logits设置为最小值即可，这样会导致softmax运算之后，这一项基本为0
        masked_sampled_logits = sampled_logits.masked_fill(
            true_in_sample_mask, torch.finfo(sampled_logits.dtype).min
        )

        # now concat the true logits as index 0
        # [batch_size, 1 + n_samples]，第一列是正样本的logits，后面的列都是采样得到的
        logits = torch.cat([true_logits, masked_sampled_logits], dim=1)

        # finally take log_softmax
        log_softmax = torch.nn.functional.log_softmax(logits, dim=1)

        # labels, shape is (batch_size, 1 + n_samples)
        labels = torch.cat(
            [
                torch.ones_like(true_logits, device=self._device),
                torch.zeros_like(masked_sampled_logits),
            ],
            dim=1,
        )

        # (batch_size,), (batch_size, 1 + num_sampled)
        batch_loss = -torch.sum(log_softmax * labels, dim=1)
        if mask is not None:
            batch_loss *= mask
        return batch_loss

    def _forward_eval(
        self,
        embeddings: torch.Tensor,
        targets: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        embeddings: shape is (batch_size, C)
        targets: shape is (batch_size,)
        mask: shape is (batch_size,)
        """

        # evaluation mode, use full softmax
        w = self.w
        b = self.b

        # (batch_size, num_classes)
        logits = torch.matmul(embeddings, w.t()) + b.t()

        # (batch_size,)
        batch_loss = self.eval_loss(logits, targets)
        if mask is not None:
            batch_loss *= mask

        return batch_loss

    def log_uniform_candidate_sampler(
        self, targets: torch.LongTensor, choice_func=_choice
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        returns sampled, true_expected_count, sampled_expected_count

        Parameters
        ----------
        targets: shape is (batch_size,)

        Returns
        ----------
        sampled_ids: shape is (n_samples,)
        true_expected_count: shape is (batch_size,)
        sampled_expected_count: shape is (n_samples,)
        """

        # based on: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/range_sampler.h
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/range_sampler.cc

        # algorithm: keep track of number of tries when doing sampling,
        #   then expected count is
        #   -expm1(num_tries * log1p(-p))
        # = (1 - (1-p)^num_tries) where p is self._probs[id]

        # 这里是对label进行不放回采样
        np_sampled_ids, num_tries = choice_func(self._num_words, self._num_samples)

        sampled_ids = torch.from_numpy(np_sampled_ids).to(targets.device)

        # P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
        target_probs = (
            torch.log((targets + 2.0) / (targets + 1.0)) / self._log_num_words_p1
        )

        # Compute expected count = (1 - (1-p)^num_tries) = -expm1(num_tries * log1p(-p)) for numerical stability
        # shape is (batch_size,)
        target_expected_count = -torch.expm1(num_tries * torch.log1p(-target_probs))

        sampled_probs = (
            torch.log((sampled_ids + 2.0) / (sampled_ids + 1.0))
            / self._log_num_words_p1
        )
        sampled_expected_count = -torch.expm1(num_tries * torch.log1p(-sampled_probs))

        sampled_ids.requires_grad_(False)
        target_expected_count.requires_grad_(False)
        sampled_expected_count.requires_grad_(False)

        return sampled_ids, target_expected_count, sampled_expected_count
