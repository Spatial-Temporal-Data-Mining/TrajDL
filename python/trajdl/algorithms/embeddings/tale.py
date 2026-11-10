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

"""TALE: Pre-training Time-aware Location Embeddings from Spatial-Temporal
Trajectories (TKDE 2021).

Skip-Gram style training with a Huffman tree per discrete time slot
(hierarchical softmax). The implementation is a faithful port of the
baseline (Logan-Lin/TALE, ``embed/w2v.py`` + ``embed/tale.py``); semantics
that look surprising are baseline behavior and must not be "fixed":

- ``HuffmanTree`` builds via linear scan with tie-breaking by insertion
  order (POI id ascending); inner node ids start from 1 (0 is padding).
- ``TaleEmbedding`` loss uses ``logsigmoid(-score)`` for code=1 path nodes
  and ``logsigmoid(+score)`` for code=0 path nodes, plus a per-slot term;
  the reduction is a ``sum`` over the batch, not a mean.
"""

import itertools
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...tokenizers import AbstractTokenizer
from ...trajdl_rust import build_tale_huffman_forest
from .base import BaseTokenEmbeddingLayer, SimpleEmbedding


def gen_all_slots(minute, time_slice_length, influence_span_length):
    """
    Compute the time slots a visit belongs to and the proportions.

    Parameters
    ----------
    minute : float
        Timestamp in minute (may exceed one day; it is reduced modulo 24h).
    time_slice_length : int
        Length of one slot in minutes.
    influence_span_length : int
        Length of the influence span in minutes; 0 means the visit belongs
        to a single slot.

    Returns
    -------
    slices : List[int]
        Slot indices within a day.
    props : List[float]
        Proportions over the slots, summing to 1.
    """

    def _cal_slice(x):
        return int((x % (24 * 60)) / time_slice_length)

    if influence_span_length == 0:
        slices, props = [_cal_slice(minute)], [1.0]
    else:
        minute_floors = list(
            {minute - influence_span_length / 2, minute + influence_span_length / 2}
            | set(
                range(
                    (int((minute - influence_span_length / 2) / time_slice_length) + 1)
                    * time_slice_length,
                    int(minute + influence_span_length / 2),
                    time_slice_length,
                )
            )
        )
        minute_floors.sort()

        slices = [_cal_slice(time_minute) for time_minute in minute_floors[:-1]]
        props = [
            (minute_floors[index + 1] - minute_floors[index]) / influence_span_length
            for index in range(len(minute_floors) - 1)
        ]
    return slices, props


class HuffmanNode:
    """
    A node in the Huffman tree.
    """

    def __init__(self, id, frequency):
        self.id = id
        self.frequency = frequency

        self.left = None
        self.right = None
        self.father = None
        self.huffman_code = []
        self.path = []  # path from root node to leaf node

    def __str__(self):
        return "HuffmanNode#{},freq{}".format(self.id, self.frequency)


class HuffmanTree:
    """
    Huffman tree used for hierarchical softmax calculation.

    Parameters
    ----------
    freq_array : numpy.ndarray
        Array of ``(id, frequency)`` rows sorted by id ascending, e.g.
        ``np.array(sorted(Counter(poi_list).items()))``. The id order
        determines the tree shape on frequency ties.
    """

    def __init__(self, freq_array: np.ndarray):
        self.num_words = freq_array.shape[0]
        self.id2code: Dict[int, List[int]] = {}
        self.id2path: Dict[int, List[int]] = {}
        self.id2pos: Dict[int, List[int]] = {}
        self.id2neg: Dict[int, List[int]] = {}
        self.root = None  # type: ignore[assignment]
        self.num_inner_nodes = 0  # Records the number of inner nodes of this tree.

        unmerged_node_list = [
            HuffmanNode(id, frequency) for id, frequency in freq_array
        ]
        self.tree = {node.id: node for node in unmerged_node_list}
        self.id_offset = max(self.tree.keys())  # Starting-off id of inner nodes.

        self._offset = self.id_offset
        self._build_tree(unmerged_node_list)
        self._gen_path()
        self._get_all_pos_neg()

    def _merge_node(self, node1: HuffmanNode, node2: HuffmanNode) -> HuffmanNode:
        """
        Merge two nodes into one, adding their frequencies.
        """
        sum_freq = node1.frequency + node2.frequency
        self._offset += 1
        mid_node_id = self._offset
        father_node = HuffmanNode(mid_node_id, sum_freq)
        if node1.frequency >= node2.frequency:
            father_node.left, father_node.right = node1, node2
        else:
            father_node.left, father_node.right = node2, node1
        self.tree[mid_node_id] = father_node
        self.num_inner_nodes += 1
        return father_node

    def _build_tree(self, node_list):
        while len(node_list) > 1:
            i1, i2 = 0, 1
            if node_list[i2].frequency < node_list[i1].frequency:
                i1, i2 = i2, i1
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        i1, i2 = i2, i1
            father_node = self._merge_node(node_list[i1], node_list[i2])
            assert not i1 == i2
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            else:
                node_list.pop(i1)
                node_list.pop(i2)
            node_list.insert(0, father_node)
        self.root = node_list[0]

    def _gen_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            while node.left or node.right:
                code = node.huffman_code
                path = node.path
                node.left.huffman_code = code + [1]
                node.right.huffman_code = code + [0]
                node.left.path = path + [node.id]
                node.right.path = path + [node.id]
                stack.append(node.right)
                node = node.left
            id = node.id
            code = node.huffman_code
            path = node.path
            self.tree[id].huffman_code, self.tree[id].path = code, path
            self.id2code[id], self.id2path[id] = code, path

    def _get_all_pos_neg(self):
        for id in self.id2code.keys():
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.tree[id].huffman_code):
                if code == 1:
                    # Inner node ids start from 1 so that 0 stays reserved
                    # for padding in the output embedding.
                    pos_id.append(self.tree[id].path[i] - self.id_offset)
                else:
                    neg_id.append(self.tree[id].path[i] - self.id_offset)
            self.id2pos[id] = pos_id
            self.id2neg[id] = neg_id


class _SlotTreeView:
    """Array-rebuilt slot-tree view: exposes only the id2pos/id2neg of the parity contract surface (local ids)."""

    __slots__ = ("id2pos", "id2neg")

    def __init__(self, id2pos: Dict[int, List[int]], id2neg: Dict[int, List[int]]):
        self.id2pos = id2pos
        self.id2neg = id2neg


class SliceHuffmanForest:
    """
    One Huffman tree per discrete time slot, with cross-slot inner node id
    offsets. Array-backed: built by the Rust core
    ``trajdl_rust.build_tale_huffman_forest`` (rayon-parallel) and
    bit-identical to :class:`SliceHuffmanForestReference` (locked by parity
    fixtures). State is flat CSR arrays, so pickling carries arrays only —
    no per-node Python objects.

    Public API matches the reference implementation (``lookup`` /
    ``gen_path_pairs`` / ``num_slots`` / ``num_inner_nodes``). The reference's
    structural attributes (``visit2slice`` / ``visit2prop`` /
    ``slice2offset`` / ``slice2tree``) are reconstructed lazily from arrays
    as parity-contract views; production hot paths must use
    :meth:`lookup` / :meth:`gen_path_pairs` /
    :meth:`max_pos_path_width` / :meth:`max_neg_path_width` directly.

    Parameters
    ----------
    sentences : sequence of sequences of int
        Tokenized location sequences (token ids).
    timestamps : sequence of sequences of int
        Visited timestamps in seconds, aligned with ``sentences``.
    slice_len : int
        Length of one time slot in minutes.
    influ_len : int
        Influence span in minutes.
    """

    def __init__(self, sentences, timestamps, slice_len: int, influ_len: int):
        total = sum(len(s) for s in sentences)
        flat_ids = np.fromiter(
            itertools.chain.from_iterable(sentences), dtype=np.int64, count=total
        )
        flat_seconds = np.fromiter(
            itertools.chain.from_iterable(timestamps), dtype=np.int64, count=total
        )
        lens = np.fromiter(
            (len(s) for s in sentences), dtype=np.int64, count=len(sentences)
        )
        offsets = np.zeros(len(sentences) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])

        arrays = build_tale_huffman_forest(
            flat_ids, flat_seconds, offsets, int(slice_len), int(influ_len)
        )
        self._num_slots = int(arrays["num_slots"])
        self._num_inner_nodes = int(arrays["num_inner_nodes"])
        self._max_poi_id = int(arrays["max_poi_id"])
        self._slot_offset = arrays["slot_offset"]
        self._tree_poi_start = arrays["tree_poi_start"]
        self._tree_poi_flat = arrays["tree_poi_flat"]
        self._pos_start = arrays["pos_start"]
        self._pos_flat = arrays["pos_flat"]
        self._neg_start = arrays["neg_start"]
        self._neg_flat = arrays["neg_flat"]
        self._visit_slice_start = arrays["visit_slice_start"]
        self._visit_slice_flat = arrays["visit_slice_flat"]
        self._visit_prop_flat = arrays["visit_prop_flat"]
        self._seen_minute_start = arrays["seen_minute_start"]
        self._seen_minute_flat = arrays["seen_minute_flat"]

    @property
    def num_slots(self) -> int:
        """Number of slot embedding rows: max slot index plus one."""
        return self._num_slots

    @property
    def num_inner_nodes(self) -> int:
        return self._num_inner_nodes

    def contains(self, poi_id: int, minute: int) -> bool:
        """Whether (poi, minute) co-occurred in the build corpus."""
        if poi_id < 0 or poi_id + 1 >= self._seen_minute_start.shape[0]:
            return False
        lo = int(self._seen_minute_start[poi_id])
        hi = int(self._seen_minute_start[poi_id + 1])
        seg = self._seen_minute_flat[lo:hi]
        i = int(np.searchsorted(seg, minute))
        return i < seg.shape[0] and int(seg[i]) == minute

    def _poi_row(self, slot: int, poi_id: int) -> int:
        base = int(self._tree_poi_start[slot])
        end = int(self._tree_poi_start[slot + 1])
        idx = base + int(np.searchsorted(self._tree_poi_flat[base:end], poi_id))
        if idx >= end or int(self._tree_poi_flat[idx]) != poi_id:
            raise KeyError((poi_id, slot))
        return idx

    def lookup(
        self, poi_id: int, minute: int
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[float]]:
        """
        Look up the Huffman paths of a visit in all its time slots.

        Returns
        -------
        pos_paths : List[List[int]]
            For every slot: inner node ids where the huffman code is 1
            (offset-adjusted, global ids starting from 1).
        neg_paths : List[List[int]]
            For every slot: inner node ids where the huffman code is 0.
        slices : List[int]
        props : List[float]
        """
        if not self.contains(poi_id, minute):
            raise KeyError((poi_id, minute))
        lo = int(self._seen_minute_start[poi_id])
        hi = int(self._seen_minute_start[poi_id + 1])
        vrow = lo + int(np.searchsorted(self._seen_minute_flat[lo:hi], minute))
        s0 = int(self._visit_slice_start[vrow])
        s1 = int(self._visit_slice_start[vrow + 1])
        slices = self._visit_slice_flat[s0:s1]
        props = self._visit_prop_flat[s0:s1]
        pos_paths, neg_paths = [], []
        for s in slices.tolist():
            row = self._poi_row(s, poi_id)
            off = int(self._slot_offset[s])
            ps, pe = int(self._pos_start[row]), int(self._pos_start[row + 1])
            ns, ne = int(self._neg_start[row]), int(self._neg_start[row + 1])
            pos_paths.append((self._pos_flat[ps:pe] + off).tolist())
            neg_paths.append((self._neg_flat[ns:ne] + off).tolist())
        return pos_paths, neg_paths, slices.tolist(), props.tolist()

    def gen_path_pairs(
        self,
        sentences,
        timestamps,
        window_size: int,
        indi_context: bool,
        skip_missing: bool = False,
    ) -> List[list]:
        """
        Generate (context, huffman paths, slots, props) pairs with a sliding
        window over every sentence; identical semantics to the reference
        (rows, ordering and skip_missing behavior).
        """
        path_pairs = []
        for sentence, timestamp in zip(sentences, timestamps):
            for i in range(len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i + window_size]
                minute = math.floor(timestamp[i + window_size] % (24 * 60 * 60) / 60)
                if skip_missing and not self.contains(target, minute):
                    continue
                pos_paths, neg_paths, slices, props = self.lookup(target, minute)
                context = (
                    sentence[i : i + window_size]
                    + sentence[i + window_size + 1 : i + 2 * window_size + 1]
                )
                if indi_context:
                    path_pairs += [
                        [[c], pos_paths, neg_paths, slices, props] for c in context
                    ]
                else:
                    path_pairs.append([context, pos_paths, neg_paths, slices, props])
        return path_pairs

    def max_pos_path_width(self) -> int:
        """Global max pos-path length over all (slot, poi) rows."""
        if self._pos_start.shape[0] <= 1:
            return 0
        return int(np.diff(self._pos_start).max())

    def max_neg_path_width(self) -> int:
        """Global max neg-path length over all (slot, poi) rows."""
        if self._neg_start.shape[0] <= 1:
            return 0
        return int(np.diff(self._neg_start).max())

    # ── 结构视图：parity 契约面（黄金 fixture / 结构测试），热路径勿用 ──

    @property
    def visit2slice(self) -> Dict[Tuple[int, int], List[int]]:
        out: Dict[Tuple[int, int], List[int]] = {}
        vrow = 0
        for poi in range(self._seen_minute_start.shape[0] - 1):
            lo = int(self._seen_minute_start[poi])
            hi = int(self._seen_minute_start[poi + 1])
            for minute in self._seen_minute_flat[lo:hi].tolist():
                s0 = int(self._visit_slice_start[vrow])
                s1 = int(self._visit_slice_start[vrow + 1])
                out[(poi, minute)] = self._visit_slice_flat[s0:s1].tolist()
                vrow += 1
        return out

    @property
    def visit2prop(self) -> Dict[Tuple[int, int], List[float]]:
        out: Dict[Tuple[int, int], List[float]] = {}
        vrow = 0
        for poi in range(self._seen_minute_start.shape[0] - 1):
            lo = int(self._seen_minute_start[poi])
            hi = int(self._seen_minute_start[poi + 1])
            for minute in self._seen_minute_flat[lo:hi].tolist():
                s0 = int(self._visit_slice_start[vrow])
                s1 = int(self._visit_slice_start[vrow + 1])
                out[(poi, minute)] = self._visit_prop_flat[s0:s1].tolist()
                vrow += 1
        return out

    @property
    def slice2offset(self) -> Dict[int, int]:
        return {
            s: int(self._slot_offset[s])
            for s in range(self._num_slots)
            if int(self._tree_poi_start[s + 1]) > int(self._tree_poi_start[s])
        }

    @property
    def slice2tree(self) -> Dict[int, _SlotTreeView]:
        trees: Dict[int, _SlotTreeView] = {}
        for s in range(self._num_slots):
            b0 = int(self._tree_poi_start[s])
            b1 = int(self._tree_poi_start[s + 1])
            if b1 == b0:
                continue
            id2pos, id2neg = {}, {}
            for row in range(b0, b1):
                poi = int(self._tree_poi_flat[row])
                ps, pe = int(self._pos_start[row]), int(self._pos_start[row + 1])
                ns, ne = int(self._neg_start[row]), int(self._neg_start[row + 1])
                # pos/neg_flat 存的是树内局部 id（1-based），直接还原
                id2pos[poi] = self._pos_flat[ps:pe].tolist()
                id2neg[poi] = self._neg_flat[ns:ne].tolist()
            trees[s] = _SlotTreeView(id2pos, id2neg)
        return trees


class SliceHuffmanForestReference:
    """
    One Huffman tree per discrete time slot, with cross-slot inner node id
    offsets, ported from the baseline ``TaleData``.

    Parity oracle: the pure-Python reference for
    :class:`SliceHuffmanForest` (the array-backed Rust implementation).
    Kept bit-identical to the original integration; do not "optimize" it.

    Parameters
    ----------
    sentences : sequence of sequences of int
        Tokenized location sequences (token ids).
    timestamps : sequence of sequences of int
        Visited timestamps in seconds, aligned with ``sentences``.
    slice_len : int
        Length of one time slot in minutes.
    influ_len : int
        Influence span in minutes.
    """

    def __init__(self, sentences, timestamps, slice_len: int, influ_len: int):
        self.sentences = sentences
        self.timestamps = timestamps

        # Map a binary tuple (poi_index, minute of day) to slice indices.
        self.visit2slice: Dict[Tuple[int, int], List[int]] = {}
        # Map a binary tuple (poi_index, minute of day) to proportions.
        self.visit2prop: Dict[Tuple[int, int], List[float]] = {}
        slice2poi: Dict[int, List[int]] = (
            {}
        )  # Record the included POIs for every time slice.

        for sentence, timestamp in zip(sentences, timestamps):
            for poi_index, visited_second in zip(sentence, timestamp):
                slice, prop = gen_all_slots(visited_second / 60, slice_len, influ_len)
                for s in slice:
                    slice2poi[s] = slice2poi.get(s, []) + [poi_index]
                int_minute = math.floor(visited_second % (24 * 60 * 60) / 60)
                self.visit2slice[(poi_index, int_minute)] = slice
                self.visit2prop[(poi_index, int_minute)] = prop

        self.slice2tree: Dict[int, HuffmanTree] = {}  # Root of every temporal tree.
        self.slice2offset: Dict[int, int] = {}  # Inner node id offset per slot.
        _total_offset = 0
        for slice_index, poi_list in slice2poi.items():
            # One Huffman tree per temporal slot; the id-ascending order of
            # `sorted(Counter(...).items())` defines the tree on freq ties.
            poi_freq = np.array(sorted(Counter(poi_list).items()))
            huffman_tree = HuffmanTree(poi_freq)
            self.slice2tree[slice_index] = huffman_tree
            self.slice2offset[slice_index] = _total_offset
            _total_offset += huffman_tree.num_inner_nodes
        self.num_inner_nodes = _total_offset + 1

    @property
    def num_slots(self) -> int:
        """Number of slot embedding rows: max slot index plus one."""
        return max(self.slice2offset.keys(), default=-1) + 1

    def lookup(
        self, poi_id: int, minute: int
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[float]]:
        """
        Look up the Huffman paths of a visit in all its time slots.

        Returns
        -------
        pos_paths : List[List[int]]
            For every slot: inner node ids where the huffman code is 1
            (offset-adjusted, global ids starting from 1).
        neg_paths : List[List[int]]
            For every slot: inner node ids where the huffman code is 0.
        slices : List[int]
        props : List[float]
        """
        visit = (poi_id, minute)
        slices = self.visit2slice[visit]
        props = self.visit2prop[visit]
        pos_paths = [
            (
                np.array(self.slice2tree[s].id2pos[poi_id]) + self.slice2offset[s]
            ).tolist()
            for s in slices
        ]
        neg_paths = [
            (
                np.array(self.slice2tree[s].id2neg[poi_id]) + self.slice2offset[s]
            ).tolist()
            for s in slices
        ]
        return pos_paths, neg_paths, slices, props

    def gen_path_pairs(
        self,
        sentences,
        timestamps,
        window_size: int,
        indi_context: bool,
        skip_missing: bool = False,
    ) -> List[list]:
        """
        Generate (context, huffman paths, slots, props) pairs with a sliding
        window over every sentence, ported from baseline ``TaleData``.

        Each row is ``[context, pos_paths, neg_paths, slices, props]`` where
        the path/prop lists are aligned with ``slices``.

        ``skip_missing`` drops center visits absent from the forest (the
        baseline raises instead); it exists so held-out validation data can
        be evaluated without crashing on unseen (poi, minute) pairs.
        """
        path_pairs = []
        for sentence, timestamp in zip(sentences, timestamps):
            for i in range(len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i + window_size]
                minute = math.floor(timestamp[i + window_size] % (24 * 60 * 60) / 60)
                if skip_missing and (target, minute) not in self.visit2slice:
                    continue
                pos_paths, neg_paths, slices, props = self.lookup(target, minute)
                context = (
                    sentence[i : i + window_size]
                    + sentence[i + window_size + 1 : i + 2 * window_size + 1]
                )
                if indi_context:
                    path_pairs += [
                        [[c], pos_paths, neg_paths, slices, props] for c in context
                    ]
                else:
                    path_pairs.append([context, pos_paths, neg_paths, slices, props])
        return path_pairs


class TaleEmbedding(BaseTokenEmbeddingLayer):
    """
    TALE token embedding: a static lookup table for POIs plus the auxiliary
    hierarchical-softmax parameters (inner nodes and time slots) used during
    pre-training.

    Parameters
    ----------
    tokenizer : AbstractTokenizer
    embedding_dim : int
    forest : SliceHuffmanForest, optional
        Built offline (see ``SliceHuffmanForest``); required for pre-training.
        Pass ``None`` for downstream finetune/eval, which only use the static
        ``token_emb`` lookup — this skips the aux parameters entirely so the
        (potentially large) forest pickle never needs to be loaded.
    """

    def __init__(
        self,
        tokenizer: AbstractTokenizer,
        embedding_dim: int,
        forest: Optional[SliceHuffmanForest] = None,
    ):
        super().__init__()
        self.forest = forest
        self.token_emb = SimpleEmbedding(tokenizer, embedding_dim)

        init_range = 0.5 / embedding_dim
        self.token_emb.embedding.weight.data.uniform_(-init_range, init_range)
        if forest is not None:
            # Inner-node table keyed by global forest node ids (1-based, see
            # slice2offset), not tokenizer ids; 0 is the store reserved slot
            # used to pad huffman paths, so tokenizer.pad does not apply and
            # padding_idx=0 keeps that reserved row zeroed and gradient-free.
            self.inner_emb = nn.Embedding(
                forest.num_inner_nodes, embedding_dim, padding_idx=0
            )
            self.slot_emb = nn.Embedding(forest.num_slots, embedding_dim)
            self.inner_emb.weight.data.uniform_(-0, 0)
            self.slot_emb.weight.data.uniform_(-0, 0)

    @property
    def embedding_dim(self) -> int:
        return self.token_emb.embedding_dim

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """Downstream usage: static lookup of token embeddings."""
        return self.token_emb(x)

    def compute_pretrain_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Hierarchical softmax loss over (context, slot) rows, ported from
        baseline ``Tale.forward``.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            - ``pos_u``: context token ids, shape (B, 2 * window_size)
            - ``pos_w``: path node ids with code=1, padded with 0, (B, L1)
            - ``neg_w``: path node ids with code=0, padded with 0, (B, L2)
            - ``pos_s``: slot indices, shape (B,)
            - ``prob``: slot proportions, shape (B,)

        Notes
        -----
        The reduction is a sum over the batch (baseline behavior); the
        learning rate must be tuned accordingly.
        """
        pos_u, pos_w = batch["pos_u"], batch["pos_w"]
        neg_w, pos_s, prob = batch["neg_w"], batch["pos_s"], batch["prob"]

        pos_u_embed = self.token_emb(pos_u)  # (B, 2 * window_size, D)
        pos_u_embed = pos_u_embed.sum(1, keepdim=True)  # (B, 1, D)

        pos_w_mask = (pos_w == 0).bool()  # (B, L1)
        pos_w_embed = self.inner_emb(pos_w)  # (B, L1, D)
        score = torch.mul(pos_u_embed, pos_w_embed).sum(dim=-1)  # (B, L1)
        score = F.logsigmoid(-1 * score)
        score = score.masked_fill(pos_w_mask, 0.0)

        neg_w_mask = (neg_w == 0).bool()  # (B, L2)
        neg_w_embed = self.inner_emb(neg_w)  # (B, L2, D)
        neg_score = torch.mul(pos_u_embed, neg_w_embed).sum(dim=-1)  # (B, L2)
        neg_score = F.logsigmoid(neg_score)
        neg_score = neg_score.masked_fill(neg_w_mask, 0.0)

        s_embed = self.slot_emb(pos_s)  # (B, D)
        s_score = torch.mul(pos_u_embed, s_embed.unsqueeze(1)).sum(dim=-1).squeeze(-1)
        s_score = F.logsigmoid(-1 * s_score)  # (B,)

        return (
            -1
            * torch.mul(
                (torch.sum(score, dim=-1) + torch.sum(neg_score, dim=-1) + s_score),
                prob,
            ).sum()
        )
