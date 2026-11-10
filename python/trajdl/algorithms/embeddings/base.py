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

from abc import abstractmethod

import torch
import torch.nn as nn
from gensim.models import KeyedVectors, Word2Vec
from tqdm import tqdm

from ...tokenizers.abstract import AbstractTokenizer
from ...tokenizers.trajcl import TrajCLTokenizer


class BaseTokenEmbeddingLayer(nn.Module):
    """
    Base class for token embedding layers.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Computes the embedding for the input tokens.

    freeze_parameters() -> None
        Freezes the parameters of the embedding layer, preventing them from being trained.

    unfreeze_parameters() -> None
        Unfreezes the parameters of the embedding layer, allowing them to be trained.

    is_frozen() -> bool
        Returns whether the parameters are currently frozen.
    """

    def __init__(self):
        super(BaseTokenEmbeddingLayer, self).__init__()
        self._frozen = False

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError(
            "Subclasses should implement this method."
        )  # pragma: no cover

    @abstractmethod
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Must override in subclass to compute embeddings.

        Parameters
        ----------
        x : torch.LongTensor
            Input tensor containing token indices.

        Returns
        -------
        torch.Tensor
            Embedding tensor for the input tokens, with increased dimensions.
        """
        raise NotImplementedError("Must override forward method")  # pragma: no cover

    def freeze_parameters(self) -> None:
        """Freeze the parameters to prevent training."""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_parameters(self) -> None:
        """Unfreeze the parameters to allow training."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """
        Check if the parameters are frozen.

        Returns
        -------
        bool
            True if parameters are frozen, otherwise False.
        """
        return self._frozen


class SimpleEmbedding(BaseTokenEmbeddingLayer):
    """
    Token embedding layer that uses PyTorch's nn.Embedding.

    Parameters
    ----------
    tokenizer: AbstractTokenizer
        Tokenizer
    embedding_dim : int
        The dimensionality of the embeddings.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Computes the embeddings for the input tokens.
    """

    def __init__(self, tokenizer: AbstractTokenizer, embedding_dim: int):
        super(SimpleEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            # Tokenizer-keyed table: pad row follows tokenizer.pad, never a
            # hardcoded 0.
            len(tokenizer),
            embedding_dim,
            padding_idx=tokenizer.pad,
        )

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Computes the embeddings for the input tokens.

        Parameters
        ----------
        x : torch.LongTensor
            Input tensor containing token indices.

        Returns
        -------
        torch.Tensor
            Embeddings for the input tokens, with increased dimensions.
        """
        return self.embedding(x)


class Word2VecEmbedding(BaseTokenEmbeddingLayer):
    """
    Token embedding layer that uses a Gensim Word2Vec model.

    Parameters
    ----------
    tokenizer: AbstractTokenizer
        Tokenizer
    model_path : str
        Path to the Word2Vec model file.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Computes the Word2Vec embeddings for the input tokens.
    """

    def __init__(self, tokenizer: AbstractTokenizer, model_path: str):
        super(Word2VecEmbedding, self).__init__()
        self.embedding = self.load_pretrained_word2vec_embeddings(
            tokenizer=tokenizer, word2vec_model_path=model_path
        )

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def load_pretrained_word2vec_embeddings(
        self, tokenizer: AbstractTokenizer, word2vec_model_path: str
    ) -> nn.Embedding:
        """
        load word2vec embeddings

        ``word2vec_model_path`` accepts two gensim artifacts: a full
        ``Word2Vec`` model (as saved by ``scripts/word2vec.py``) or a
        ``KeyedVectors`` file (as exported by ``W2VTrainingFramework``).
        Both go through the same tokenizer alignment below.
        """
        try:
            model = Word2Vec.load(word2vec_model_path)
            wv = model.wv
        except AttributeError:
            # KeyedVectors artifacts lack Word2Vec-specific attributes.
            wv = KeyedVectors.load(word2vec_model_path)
        embedding_dim = wv.vector_size
        embedding_matrix = torch.zeros(
            size=(len(tokenizer), embedding_dim), dtype=torch.float32
        )

        pad_token = tokenizer.id_to_token(tokenizer.pad)

        for token_id in tqdm(range(len(tokenizer)), desc="loading word2vec embeddings"):
            token = tokenizer.id_to_token(token_id)
            if token in wv:
                embedding_matrix[token_id] = torch.from_numpy(wv[token].copy())
            else:
                # Special tokens or unseen tokens: PAD stays zero, others get random init
                if token != pad_token:
                    nn.init.normal_(embedding_matrix[token_id])

        return nn.Embedding.from_pretrained(
            # Tokenizer-keyed table: pad row follows tokenizer.pad, never a
            # hardcoded 0.
            embedding_matrix,
            freeze=False,
            padding_idx=tokenizer.pad,
        )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Computes the Word2Vec embeddings for the input tokens.

        Parameters
        ----------
        x : torch.LongTensor
            Input tensor containing token indices.

        Returns
        -------
        torch.Tensor
            Word2Vec embeddings for the input tokens, with increased dimensions.
        """
        return self.embedding(x)


class Node2VecEmbedding(BaseTokenEmbeddingLayer):
    """
    Token embedding layer that uses node2vec embeddings for grid cells.

    This class aligns node2vec embeddings (indexed by cell_id) with the
    tokenizer's vocabulary (indexed by token_id). Grid symbols are mapped
    to cell_ids via GridCoord, and special tokens are initialized separately.

    Parameters
    ----------
    tokenizer : TrajCLTokenizer
        Tokenizer containing vocab and grid system info.
    cell_embeddings : torch.Tensor
        Node2vec output embeddings, shape (total_cells, embedding_dim).
    freeze : bool, optional
        Whether to freeze the embedding weights, by default False.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Computes the node2vec embeddings for the input tokens.
    """

    def __init__(
        self,
        tokenizer: TrajCLTokenizer,
        cell_embeddings: torch.Tensor,
        freeze: bool = False,
    ):
        super().__init__()
        self.embedding = self._build_embedding(tokenizer, cell_embeddings, freeze)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def _build_embedding(
        self,
        tokenizer: TrajCLTokenizer,
        cell_embeddings: torch.Tensor,
        freeze: bool,
    ) -> nn.Embedding:
        """
        Build embedding layer from node2vec cell embeddings.

        Parameters
        ----------
        tokenizer : TrajCLTokenizer
            Tokenizer containing vocab and grid system info.
        cell_embeddings : torch.Tensor
            Node2vec output embeddings, shape (total_cells, embedding_dim).
        freeze : bool
            Whether to freeze the embedding weights.

        Returns
        -------
        nn.Embedding
            Embedding layer with aligned embeddings.
        """
        from trajdl import GridCoord

        vocab_size = len(tokenizer)
        emb_dim = cell_embeddings.shape[1]
        web_grid = tokenizer.web_grid

        # Initialize full_embedding with zeros
        full_embedding = torch.zeros(vocab_size, emb_dim)

        for token_id in range(vocab_size):
            grid_symbol = tokenizer.id_to_token(token_id)

            if grid_symbol.startswith("G_"):
                # token_id → grid_symbol → GridCoord → cell_id
                grid_coord = GridCoord.from_grid_symbol(grid_symbol)
                cell_id = web_grid.locate_by_grid_coordinate(grid_coord)
                full_embedding[token_id] = cell_embeddings[cell_id]
            else:
                # special token: PAD stays zero, others get random init
                pad_token = tokenizer.id_to_token(tokenizer.pad)
                if grid_symbol != pad_token:
                    nn.init.normal_(full_embedding[token_id])

        return nn.Embedding.from_pretrained(
            # Tokenizer-keyed table: pad row follows tokenizer.pad, never a
            # hardcoded 0.
            full_embedding,
            freeze=freeze,
            padding_idx=tokenizer.pad,
        )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Computes the node2vec embeddings for the input tokens.

        Parameters
        ----------
        x : torch.LongTensor
            Input tensor containing token indices.

        Returns
        -------
        torch.Tensor
            Node2vec embeddings for the input tokens, with increased dimensions.
        """
        return self.embedding(x)
