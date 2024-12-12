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
from gensim.models import Word2Vec
from tqdm import tqdm

from ...tokenizers.abstract import AbstractTokenizer


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
            len(tokenizer), embedding_dim, padding_idx=tokenizer.pad
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
        """
        model = Word2Vec.load(word2vec_model_path)
        embedding_dim = model.vector_size
        embedding_matrix = torch.zeros(
            size=(len(tokenizer), embedding_dim), dtype=torch.float32
        )

        words = model.wv.index_to_key
        for word in tqdm(words, desc="loading word2vec embeddings"):
            embedding_matrix[tokenizer.loc2idx(word)] = torch.from_numpy(
                model.wv[word].copy()
            )

        return nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

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
