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

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TULERSample:
    """
    TULER的输入样本

    Parameters
    ----------
    src: torch.LongTensor
        shape is (B, T)

    seq_len: List[int]
        length of each sequence

    labels: torch.LongTensor, optional
        shape is (B,), label of each sequence, default is None

    """

    src: torch.LongTensor
    seq_len: List[int]
    labels: Optional[torch.LongTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.seq_len)


@dataclass
class T2VECSample:
    """
    T2VEC的输入样本

    Parameters
    ----------
    src: torch.LongTensor
        shape is (batch_size, seq_length)

    lengths: List[int]
        length of each src sequence

    target: torch.LongTensor, optional
        shape is (batch_size, seq_length'), default is None
    """

    src: torch.LongTensor
    lengths: List[int]
    target: Optional[torch.LongTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.lengths)


@dataclass
class GMVSAESample:
    """
    GMVSAE的输入样本

    Parameters
    ----------
    encoder_seq: torch.LongTensor
        shape is (B, T)

    encoder_lengths: List[int]
        length of each sequence

    decoder_seq: Optional[torch.LongTensor], optional
        shape is (B, T + 1), each of seq added BOS, default is None

    decoder_lengths: Optional[List[int]], optional
        encoder_lengths + 1, default is None

    decoder_labels: Optional[torch.LongTensor], optional
        shape is (B, T + 1), each of seq added EOS, default is None

    mask: Optional[torch.Tensor], optional
        shape is (B, T + 1), default is None

    """

    encoder_seq: torch.LongTensor
    encoder_lengths: List[int]
    decoder_seq: Optional[torch.LongTensor] = None
    decoder_lengths: Optional[List[int]] = None
    decoder_labels: Optional[torch.LongTensor] = None
    mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.encoder_lengths)


@dataclass
class STLSTMSample:
    """
    ST-LSTM的输入样本

    Parameters
    ----------
    loc_seq: torch.LongTensor
        shape is (B, T), 位置序列
    td_upper_seq: torch.LongTensor
        shape is (B, T), 这个是两个时间步之间的时间差被上界减去的batch
    td_lower_seq: torch.LongTensor
        shape is (B, T), 这个是两个时间步之间的时间差减去下界的batch
    sd_upper_seq: torch.LongTensor
        shape is (B, T), 这个是两个时间步之间的空间位移被上界减去的batch
    sd_lower_seq: torch.LongTensor
        shape is (B, T), 这个是两个时间步之间的空间位移减去下界的batch
    valid_lengths: List[int]
        每条序列的实际长度
    labels: torch.LongTensor, optional
        shape is (B, T), 训练时传入, LSTM的输出对应的标签
    mask: torch.LongTensor, optional
        shape is (B, T), 训练时传入, LSTM的输出对应的mask, 为1表示该位置应该计算损失, 为0不计算
    """

    loc_seq: torch.LongTensor
    td_upper_seq: torch.LongTensor
    td_lower_seq: torch.LongTensor
    sd_upper_seq: torch.LongTensor
    sd_lower_seq: torch.LongTensor
    valid_lengths: List[int]
    labels: Optional[torch.LongTensor] = None
    mask: Optional[torch.LongTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.valid_lengths)
