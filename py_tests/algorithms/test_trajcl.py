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

"""Tests for TrajCL model components."""

import torch
from trajdl.algorithms.trajcl import (
    DualSTB,
    MoCo,
    PositionalEncoding,
    Projector,
    SpatialMSM,
    TrajCL,
    TrajCLLightning,
)
from trajdl.common.samples import TrajCLSample
from trajdl.tokenizers import TrajCLTokenizer


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        """Test output shape matches input."""
        emb_size = 64
        dropout = 0.1
        pe = PositionalEncoding(emb_size, dropout, maxlen=100)

        seq_len, batch_size = 20, 4
        x = torch.randn(seq_len, batch_size, emb_size)
        output = pe(x)

        assert output.shape == x.shape

    def test_different_positions(self):
        """Test different positions have different encodings."""
        emb_size = 64
        pe = PositionalEncoding(emb_size, 0.0, maxlen=100)

        # Get encodings at different positions
        x1 = torch.zeros(1, 1, emb_size)
        x2 = torch.zeros(2, 1, emb_size)

        out1 = pe(x1)
        out2 = pe(x2)

        # First positions should be the same
        assert torch.allclose(out1[0], out2[0])
        # Different positions should differ
        assert not torch.allclose(out2[0], out2[1])


class TestSpatialMSM:
    """Tests for SpatialMSM module."""

    def test_output_shape(self):
        """Test output shapes."""
        ninput, nhidden, nhead, nlayer = 4, 32, 1, 2
        spatial_msm = SpatialMSM(ninput, nhidden, nhead, nlayer, 0.1, 0.1)

        seq_len, batch_size = 20, 4
        src = torch.randn(seq_len, batch_size, ninput)
        src_len = torch.tensor([20, 15, 10, 5])
        src_padding_mask = torch.arange(seq_len).unsqueeze(0) >= src_len.unsqueeze(1)

        out, attn = spatial_msm(src, None, src_padding_mask, src_len)

        # Output shape is (batch_size, ninput) - original TrajCL uses d_model=ninput
        assert out.shape == (batch_size, ninput)
        assert attn.shape == (batch_size, seq_len, seq_len)


class TestDualSTB:
    """Tests for DualSTB encoder."""

    def test_output_shape(self):
        """Test output shape."""
        emb_dim, hidden_dim, nhead, nlayer = 64, 128, 4, 2
        encoder = DualSTB(emb_dim, hidden_dim, nhead, nlayer, 0.1, 0.1)

        seq_len, batch_size = 20, 4
        src = torch.randn(seq_len, batch_size, emb_dim)
        srcspatial = torch.randn(seq_len, batch_size, 4)
        src_len = torch.tensor([20, 15, 10, 5])
        src_padding_mask = torch.arange(seq_len).unsqueeze(0) >= src_len.unsqueeze(1)

        out = encoder(src, None, src_padding_mask, src_len, srcspatial)

        assert out.shape == (batch_size, emb_dim)


class TestMoCo:
    """Tests for MoCo framework."""

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        emb_dim, proj_dim, queue_size = 64, 32, 256
        batch_size = 4

        # Create simple encoders
        encoder_q = torch.nn.Linear(emb_dim, emb_dim)
        encoder_k = torch.nn.Linear(emb_dim, emb_dim)

        moco = MoCo(
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            nemb=emb_dim,
            nout=proj_dim,
            queue_size=queue_size,
        )

        # Create dummy inputs
        x = torch.randn(batch_size, emb_dim)

        # Simple kwargs
        kwargs_q = {"src": x}
        kwargs_k = {"src": x}

        # Modify encoders to accept kwargs
        class SimpleEncoder(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, src, **kwargs):
                return self.linear(src)

        moco.encoder_q = SimpleEncoder(encoder_q)
        moco.encoder_k = SimpleEncoder(encoder_k)

        logits, targets = moco(kwargs_q, kwargs_k)

        assert logits.shape == (batch_size, 1 + queue_size)
        assert targets.shape == (batch_size,)
        assert (targets == 0).all()


class TestProjector:
    """Tests for Projector module."""

    def test_output_shape(self):
        """Test output shape."""
        nin, nout = 64, 32
        proj = Projector(nin, nout)

        batch_size = 4
        x = torch.randn(batch_size, nin)
        out = proj(x)

        assert out.shape == (batch_size, nout)


class TestTrajCL:
    """Tests for TrajCL model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = TrajCL(
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.1,
            pos_dropout=0.1,
            proj_dim=32,
            queue_size=256,
            temperature=0.05,
        )
        assert model is not None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        emb_dim = 64
        model = TrajCL(
            embedding_dim=emb_dim,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.1,
            pos_dropout=0.1,
            proj_dim=32,
            queue_size=256,
            temperature=0.05,
        )

        seq_len, batch_size = 20, 4
        cell_emb = torch.randn(seq_len, batch_size, emb_dim)
        spatial_feat = torch.randn(seq_len, batch_size, 4)
        lengths = [20, 15, 10, 5]

        out = model(cell_emb, spatial_feat, lengths)
        assert out.shape == (batch_size, emb_dim)


class TestTrajCLLightning:
    """Tests for TrajCLLightning module."""

    def test_lightning_creation(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
    ):
        """Test Lightning module can be created."""
        model = TrajCLLightning(
            tokenizer=trajcl_tokenizer,
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.1,
            pos_dropout=0.1,
            proj_dim=32,
            queue_size=256,
            temperature=0.05,
        )
        assert model is not None

    def test_lightning_creation_from_str_path(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
        tmp_path,
    ):
        """standard §3.2 T1：yaml 纯路径字符串必须能直接实例化 Lightning 模块."""
        path = tmp_path / "tokenizer.pkl"
        trajcl_tokenizer.save_pretrained(path)
        model = TrajCLLightning(
            tokenizer=str(path),
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.1,
            pos_dropout=0.1,
            proj_dim=32,
            queue_size=256,
            temperature=0.05,
        )
        assert model is not None

    def test_training_step(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
    ):
        """Test training step works."""
        model = TrajCLLightning(
            tokenizer=trajcl_tokenizer,
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.1,
            pos_dropout=0.1,
            proj_dim=32,
            queue_size=256,
            temperature=0.05,
        )

        # Create dummy batch
        batch = TrajCLSample(
            view1_cell_seq=torch.randint(0, 100, (4, 20)),
            view1_spatial_feat=torch.randn(4, 20, 4),
            view1_lengths=[20, 15, 10, 5],
            view2_cell_seq=torch.randint(0, 100, (4, 18)),
            view2_spatial_feat=torch.randn(4, 18, 4),
            view2_lengths=[18, 12, 8, 4],
        )

        loss = model.training_step(batch, 0)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive loss

    def test_validation_step(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
    ):
        """Test validation step works."""
        model = TrajCLLightning(
            tokenizer=trajcl_tokenizer,
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.1,
            pos_dropout=0.1,
            proj_dim=32,
            queue_size=256,
            temperature=0.05,
        )

        batch = TrajCLSample(
            view1_cell_seq=torch.randint(0, 100, (4, 20)),
            view1_spatial_feat=torch.randn(4, 20, 4),
            view1_lengths=[20, 15, 10, 5],
            view2_cell_seq=torch.randint(0, 100, (4, 18)),
            view2_spatial_feat=torch.randn(4, 18, 4),
            view2_lengths=[18, 12, 8, 4],
        )

        loss = model.validation_step(batch, 0)
        assert loss.dim() == 0

    def test_forward_inference(
        self,
        trajcl_tokenizer: TrajCLTokenizer,
    ):
        """Test forward pass for inference."""
        model = TrajCLLightning(
            tokenizer=trajcl_tokenizer,
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            attn_dropout=0.1,
            pos_dropout=0.1,
            proj_dim=32,
            queue_size=256,
            temperature=0.05,
        )

        batch = TrajCLSample(
            view1_cell_seq=torch.randint(0, 100, (4, 20)),
            view1_spatial_feat=torch.randn(4, 20, 4),
            view1_lengths=[20, 15, 10, 5],
            view2_cell_seq=torch.randint(0, 100, (4, 20)),
            view2_spatial_feat=torch.randn(4, 20, 4),
            view2_lengths=[20, 15, 10, 5],
        )

        embeddings = model(batch)
        assert embeddings.shape == (4, 64)  # (batch_size, embedding_dim)
