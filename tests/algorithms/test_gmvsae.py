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

import pytest
import torch
from torch import nn

from trajdl.algorithms.gmvsae import GMVSAE, Decoder, Encoder, LatentSpace
from trajdl.common.samples import GMVSAESample
from trajdl.tokenizers.locseq import LocSeqTokenizer


def test_encoder(locseq_tokenizer: LocSeqTokenizer):
    num_layers = 2
    num_hiddens = 16
    embedding_size = 3
    mem_num = 10
    emb = nn.Embedding(10, embedding_size)

    # test encoder
    batch_size = 2
    x = torch.LongTensor([[0, 1, 2], [1, 2, 3]])
    assert x.shape[0] == batch_size
    lengths = [3, 2]
    model = Encoder(embedding_layer=emb, hidden_size=num_hiddens, num_layers=num_layers)
    hidden_state = model(x, lengths)
    assert hidden_state.shape == (num_layers, batch_size, num_hiddens)

    # test latent space
    latent = LatentSpace(hidden_size=num_hiddens, num_layers=num_layers, c=mem_num)
    z, batch_gaussian_loss, batch_uniform_loss = latent(hidden_state)
    assert z.shape == (batch_size, num_hiddens)
    assert batch_gaussian_loss.shape == (batch_size,)
    assert batch_uniform_loss.shape == (batch_size,)

    for c_idx in range(mem_num):
        assert latent.get_mean_c(c_idx).shape == (1, num_hiddens)

    with pytest.raises(Exception):
        latent.get_mean_c(-1)

    # test decoder
    decoder = Decoder(
        emb=emb,
        hidden_size=num_hiddens,
        padding_value=locseq_tokenizer.pad,
        num_layers=num_layers,
    )
    decoder_output = decoder(x, lengths, z)
    assert decoder_output.shape == (num_layers, max(lengths), num_hiddens)
    assert (decoder_output[1, 2, :] - locseq_tokenizer.pad).sum() == 0

    # test gmvsae
    model = GMVSAE(
        tokenizer=locseq_tokenizer,
        embedding_dim=embedding_size,
        hidden_size=num_hiddens,
        mem_num=mem_num,
        num_layers=num_layers,
        mode="train",
    )
    encoder_seq = torch.LongTensor([[0, 1, 2], [1, 2, locseq_tokenizer.pad]])
    encoder_lengths = [3, 2]
    decoder_seq = torch.LongTensor(
        [[locseq_tokenizer.bos, 0, 1, 2], [locseq_tokenizer.bos, 1, 2, 3]]
    )
    decoder_lengths = [4, 3]
    decoder_labels = torch.LongTensor(
        [
            [0, 1, 2, locseq_tokenizer.eos],
            [1, 2, locseq_tokenizer.eos, locseq_tokenizer.pad],
        ]
    )
    mask = torch.BoolTensor([[True, True, True, True], [True, True, True, False]])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = GMVSAESample(
        encoder_seq=encoder_seq,
        encoder_lengths=encoder_lengths,
        decoder_seq=decoder_seq,
        decoder_lengths=decoder_lengths,
        decoder_labels=decoder_labels,
        mask=mask,
    )

    assert isinstance(model.training_step(batch=batch, batch_idx=0).item(), float)
    assert isinstance(model.validation_step(batch=batch, batch_idx=0).item(), float)

    for _ in range(10):
        loss, _ = model.compute_loss(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        model.eval()

        for c_idx in range(mem_num):
            init_state = model.init_decoder_state_for_inference(
                batch_size=batch_size, c_idx=c_idx
            )
            assert init_state.shape == (batch_size, num_hiddens)

            batch_likelihood = model.decode(
                init_state, decoder_seq, decoder_lengths, decoder_labels, mask
            )
            assert batch_likelihood.shape == (batch_size,)

        scores = model.abnormal_detect(
            decoder_seq, decoder_lengths, decoder_labels, mask
        )
        assert scores.shape == (batch_size,)
