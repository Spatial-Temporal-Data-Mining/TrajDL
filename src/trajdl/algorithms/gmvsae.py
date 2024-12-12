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

import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..common.enum import Mode
from ..common.samples import GMVSAESample
from ..loss.sampled_softmax import SampledSoftmaxLoss
from ..utils import find_best_checkpoint, load_tokenizer, tiny_value_of_dtype
from .framework import PretrainTrainFramework


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.emb = embedding_layer
        self.rnn = nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, seq: torch.LongTensor, lengths: List[int]):
        # (B, T, C)
        emb = self.emb(seq)

        x = pack_padded_sequence(
            input=emb, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        # shape of last_encoder_hidden_state is (num_layers, batch_size, num_hiddens)
        _, last_encoder_hidden_state = self.rnn(x)

        # (num_layers, B, H)
        return last_encoder_hidden_state


class LatentSpace(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        c: int,
        reduction: str = "none",
    ):
        super(LatentSpace, self).__init__()
        self.mean_linear = nn.Linear(
            in_features=hidden_size * num_layers, out_features=hidden_size
        )
        self.log_var_linear = nn.Linear(
            in_features=hidden_size * num_layers, out_features=hidden_size
        )
        self.mean_c = nn.Parameter(torch.rand(size=(c, hidden_size)))
        self.log_var_c = nn.Parameter(
            torch.zeros(size=(c, hidden_size)), requires_grad=False
        )
        self.c = c
        self.reduction = reduction

    def get_mean_c(self, idx: int) -> torch.Tensor:
        assert 0 <= idx < self.c
        return self.mean_c[idx : idx + 1]

    def forward(self, encoder_state: torch.Tensor):
        """
        encoder_state shape is (num_layers, B, H)
        """
        batch_size = encoder_state.shape[1]

        # (B, num_layers * H)
        state = encoder_state.swapaxes(0, 1).reshape(batch_size, -1)

        # (B, H)
        mean_z = self.mean_linear(state)

        # (B, H)
        log_var_z = self.log_var_linear(state)

        # (B, H)
        eps_z = torch.normal(mean=0.0, std=1.0, size=mean_z.shape, device=mean_z.device)

        # (B, H)
        z = eps_z * torch.exp(log_var_z) + mean_z

        # (B, c, H)
        stack_mu_c = self.mean_c.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        # (B, c, H)
        stack_log_sigma_sq_c = self.log_var_c.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        # (B, c, H)
        stack_z = z.unsqueeze(dim=1).repeat(1, self.c, 1)

        # (B, c)
        pi_post_logits = torch.sum(
            (stack_z - stack_mu_c) ** 2 / torch.exp(stack_log_sigma_sq_c), dim=-1
        )

        # (B, c)
        pi_post = torch.softmax(pi_post_logits, dim=-1) + tiny_value_of_dtype(
            pi_post_logits.dtype
        )

        # (B, c, H)
        stack_mu_z = z.unsqueeze(dim=1).repeat(1, self.c, 1)

        # (B, c, H)
        stack_log_sigma_sq_z = log_var_z.unsqueeze(dim=1).repeat(1, self.c, 1)

        # (B,)
        batch_gaussian_loss = 0.5 * torch.sum(
            pi_post
            * torch.mean(
                stack_log_sigma_sq_c
                + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
                + (stack_mu_z - stack_mu_c) ** 2 / torch.exp(stack_log_sigma_sq_c),
                dim=-1,
            ),
            dim=-1,
        ) - 0.5 * torch.mean(1 + log_var_z, dim=-1)
        if self.reduction == "mean":
            batch_gaussian_loss = batch_gaussian_loss.mean()

        # shape is (1,)
        # batch_uniform_loss = torch.mean(
        #     torch.mean(pi_post, dim=0) * torch.log(torch.mean(pi_post, dim=0))
        # )
        batch_uniform_loss = torch.mean(pi_post * torch.log(pi_post), dim=1)
        if self.reduction == "mean":
            batch_uniform_loss = batch_uniform_loss.mean()

        return z, batch_gaussian_loss, batch_uniform_loss


class Decoder(nn.Module):
    def __init__(
        self,
        emb: nn.Embedding,
        hidden_size: int,
        padding_value: float,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Decoder, self).__init__()
        self.emb = emb
        self.rnn = nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self._padding_value = padding_value
        self._num_layers = num_layers

    def forward(
        self, seq: torch.LongTensor, lengths: List[int], init_hidden: torch.Tensor
    ):
        """
        init_hidden shape is (B, H)
        """
        # (B, T, C)
        emb = self.emb(seq)

        x = pack_padded_sequence(
            emb, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        # (B, T, C)
        output, _ = self.rnn(
            x, init_hidden.unsqueeze(dim=0).repeat(self._num_layers, 1, 1)
        )

        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=self._padding_value
        )
        return output


class GMVSAE(PretrainTrainFramework):
    """
    GMVSAE
    """

    def __init__(
        self,
        tokenizer: Union[str, Any],
        embedding_dim: int,
        hidden_size: int,
        mem_num: int,
        mode: str,
        num_layers: int = 1,
        num_neg_samples: int = 64,
        init_mu_c_pretrained_path: Optional[str] = None,
        pretrain_ckpt_folder: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        tokenizer: path of tokenzier or a tokenizer instance

        embedding_dim: size of emb dim

        hidden_size: rnn hidden size

        mem_num: num types of route

        num_layers: default 1, num layers of rnn

        mode: {"pretrain", "train", "eval"}

        num_neg_samples: default 64, num negative samples in sampled softmax loss

        init_mu_c_pretrained_path: default None, path of pretrained init_mu_c

        pretrain_ckpt_folder: default None, path of pretrained ckpt folder

        """
        super().__init__(mode=mode)
        self.save_hyperparameters()

        tokenizer = load_tokenizer(tokenizer)
        num_vocab = len(tokenizer)
        self.emb = nn.Embedding(num_vocab, embedding_dim)
        self.encoder = Encoder(
            embedding_layer=self.emb, hidden_size=hidden_size, num_layers=num_layers
        )
        self.latent = LatentSpace(
            hidden_size=hidden_size,
            num_layers=num_layers,
            c=mem_num,
            reduction="mean",
        )

        self.decoder = Decoder(
            emb=self.emb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            padding_value=tokenizer.pad,
        )

        # weights and bias are used for transform rnn output into label space
        # just create tensors to use as the embeddings Glorit init (std=(1.0 / sqrt(fan_in))
        self.w = torch.nn.Parameter(
            torch.randn(num_vocab, hidden_size) / np.sqrt(hidden_size)
        )
        self.b = torch.nn.Parameter(torch.zeros(size=(num_vocab, 1)))

        self.reconstruct_loss = SampledSoftmaxLoss(
            weights=self.w,
            bias=self.b,
            num_words=num_vocab,
            num_samples=num_neg_samples,
            reduction="mean",
            use_sampled_softmax_in_eval=True,
        )
        self.hidden_size = hidden_size
        self._mem_num = mem_num

        self._pretrain_ckpt_folder = pretrain_ckpt_folder
        self._init_mu_c_pretrained_path = init_mu_c_pretrained_path

    @property
    def mem_num(self) -> int:
        return self._mem_num

    def init_from_pretrained_ckpt(self) -> None:
        """
        当模型是训练模式的时候，从预训练的checkpoint进行参数的加载
        """
        if self.mode == Mode.TRAIN:
            if self._pretrain_ckpt_folder:
                ckpt_filename = find_best_checkpoint(
                    self._pretrain_ckpt_folder, is_maximizing=False
                )
                print(f"load weights from {ckpt_filename}")
                checkpoint = torch.load(
                    os.path.join(self._pretrain_ckpt_folder, ckpt_filename)
                )
                self.load_state_dict(checkpoint["state_dict"])

            if self._init_mu_c_pretrained_path:
                print(f"load init_mu_c from {self._init_mu_c_pretrained_path}")
                init_mu_c = np.load(self._init_mu_c_pretrained_path)
                with torch.no_grad():
                    self.latent.mean_c.copy_(
                        torch.from_numpy(init_mu_c).to(self.latent.mean_c.dtype)
                    )

    def init_decoder_state_for_inference(
        self, batch_size: int, c_idx: int
    ) -> torch.Tensor:
        """
        init a decoder state for inference
        """
        assert 0 <= c_idx < self.mem_num

        # shape is (1, hidden_size)
        mean_c = self.latent.get_mean_c(c_idx)

        # shape is (B, hidden_size)
        return mean_c.repeat(batch_size, 1)

    def decode(
        self,
        init_state: torch.Tensor,
        decoder_seq: torch.LongTensor,
        decoder_lengths: List[int],
        decoder_labels: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        init_state: shape is (B, H)

        decoder_seq: shape is (B, T + 1), each of seq added BOS
        decoder_lengths: List[int], encoder_lengths + 1
        decoder_labels: shape is (B, T + 1), each of seq added EOS
        mask: shape is (B, T + 1)
        """
        # (B, T, H)
        decoder_output = self.decoder(decoder_seq, decoder_lengths, init_state)

        # (B, T)
        logits = torch.sigmoid(
            (decoder_output * self.w[decoder_labels]).sum(dim=-1)
            + self.b[decoder_labels].squeeze(dim=-1)
        )

        # (B,)
        batch_likelihood = (logits * mask).sum(dim=1) / mask.sum(dim=1)

        return batch_likelihood

    def forward(self, batch: GMVSAESample) -> torch.Tensor:
        """
        推理的逻辑，在预训练阶段和评估阶段的生成结果不同。
        1. 预训练阶段是生成隐变量z
        2. 评估阶段是生成序列的异常分数。

        Parameters
        ----------
        batch: GMVSAESample
            输入样本

        batch_idx: int
            lightning框架使用的batch_idx

        Returns
        ----------
        z: torch.Tensor
            当模式为预训练的时候返回这一项，shape is (num_layers, B, H)，隐变量z

        inference_result: torch.Tensor
            当模式为评估的时候返回这一项，shape is (B,)，是各条序列的异常分数

        """
        if self.mode == Mode.PRETRAIN:
            # (num_layers, B, H)
            return self.generate_z(batch.encoder_seq, batch.encoder_lengths)
        elif self.mode == Mode.EVAL:
            return self.abnormal_detect(
                decoder_seq=batch.decoder_seq,
                decoder_lengths=batch.decoder_lengths,
                decoder_labels=batch.decoder_labels,
                mask=batch.mask,
            )

    def abnormal_detect(
        self,
        decoder_seq: torch.LongTensor,
        decoder_lengths: List[int],
        decoder_labels: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        使用解码器进行序列的异常检测

        Parameters
        ----------
        decoder_seq: torch.LongTensor
            解码器输入的序列

        decoder_lengths: List[int]
            解码器输入的各序列的长度

        decoder_labels: torch.LongTensor
            解码器需要的输出

        mask: torch.BoolTensor
            解码器输入序列长度对应的mask矩阵

        Returns
        ----------
        torch.Tensor
            shape is (B,)，表示每条序列的异常分数

        """

        batch_size = len(decoder_lengths)

        scores = []
        for c_idx in range(self.mem_num):
            init_state = self.init_decoder_state_for_inference(
                batch_size=batch_size, c_idx=c_idx
            )
            # (B,)
            batch_likelihood = self.decode(
                init_state, decoder_seq, decoder_lengths, decoder_labels, mask
            )
            scores.append(batch_likelihood.unsqueeze(dim=1))

        # (B, c)
        scores = torch.cat(scores, dim=1)

        # (B,)
        return scores.max(dim=1).values

    def generate_z(
        self,
        encoder_seq: torch.LongTensor,
        encoder_lengths: List[int],
        return_loss: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        这个是编码阶段进行z的生成

        Parameters
        ----------
        encoder_seq: torch.LongTensor
            编码器的输入序列，shape是(B, T)

        encoder_lengths: List[int]
            编码器输入序列各个序列的长度

        return_loss: bool, optional
            是否返回损失，默认是False

        Returns
        ----------
        (z, batch_gaussian_loss, batch_uniform_loss): Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            当return_loss为True的时候，再返回两个loss，三个tensor的shape都是(num_layers, B, H)

        z: torch.Tensor
            当return_loss为False的时候，只返回z，shape是(num_layers, B, H)

        """
        # (num_layers, B, H)
        encoder_last_hidden = self.encoder(encoder_seq, encoder_lengths)

        # (num_layers, B, H), (num_layers, B, H), (num_layers, B, H)
        z, batch_gaussian_loss, batch_uniform_loss = self.latent(encoder_last_hidden)

        if return_loss:
            return z, batch_gaussian_loss, batch_uniform_loss
        else:
            return z

    def compute_loss(self, batch: GMVSAESample):
        """
        计算损失

        Parameters
        ----------
        batch: GMVSAESample
            输入样本


        Returns
        ----------
        loss: torch.Tensor
            shape is (1,)

        batch_size: int
            batch size

        """

        batch_size = batch.encoder_seq.shape[0]

        # (num_layers, B, H), (num_layers, B, H), (num_layers, B, H)
        z, batch_gaussian_loss, batch_uniform_loss = self.generate_z(
            encoder_seq=batch.encoder_seq,
            encoder_lengths=batch.encoder_lengths,
            return_loss=True,
        )

        # (B, T, H)
        decoder_output = self.decoder(batch.decoder_seq, batch.decoder_lengths, z)

        # (B * T, H)
        decoder_output_reshape = decoder_output.reshape(-1, self.hidden_size)

        # (B * T,)
        decoder_labels_reshape = batch.decoder_labels.reshape(-1)

        # (B * T,)
        mask_reshape = batch.mask.reshape(-1)

        # (1,)
        reconstruct_loss = self.reconstruct_loss(
            decoder_output_reshape, decoder_labels_reshape, mask_reshape
        )

        if self.mode == Mode.PRETRAIN:
            return reconstruct_loss, batch_size
        elif self.mode == Mode.TRAIN:
            return (
                reconstruct_loss + batch_gaussian_loss + batch_uniform_loss,
                batch_size,
            )
        else:
            raise RuntimeError(f"Invalid model value: {self.mode}")

    def training_step(self, batch: GMVSAESample, batch_idx: int):
        loss, batch_size = self.compute_loss(batch)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch: GMVSAESample, batch_idx: int):
        loss, batch_size = self.compute_loss(batch)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss
