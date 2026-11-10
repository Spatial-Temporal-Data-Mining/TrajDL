// Copyright 2024 All authors of TrajDL
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! TALE `SliceHuffmanForest` 构建（`python/trajdl/algorithms/embeddings/tale.py`
//! 的逐位复刻，性能重写不改语义）。
//!
//! 语义参照（改动前必读）：
//! - 槽偏移按 `slice2poi` 的 **插入序**（= 槽在语料点流中的首次出现序）累加，
//!   与 Python dict 的 insertion order 一致，保证全局 inner node id 逐位一致。
//! - 树构建是**线性扫描双最小选择**（O(n²)），非堆实现；频次打平时先扫描到的
//!   节点胜出，左右子树规则：`node1.freq >= node2.freq → left=node1`（node1
//!   为扫描序最小）。inner node id 从 max(叶子 id)+1 起。
//! - 叶子 path 为根到父的节点 id 序列（不含叶自身），code 位 左=1/右=0；
//!   pos/neg = path 中对应位节点 id 减 `id_offset`（树内 1-based 局部 id），
//!   全局 id 由消费端加 `slot_offset`。
//! - `influ_len > 0` 时同一 `int_minute` 内不同秒会算出不同 slices/props，
//!   Python dict 为点序**最后写入者胜出**——此处按同规则折叠。
//! - `num_inner_nodes`（Python 语义）= 各树内节点总数 + 1（0 号留给 padding）。
//!
//! 并行与确定性：点扫描与建树用 rayon，全部结果按 index 收集/整数求和，
//! 与调度顺序无关；线程数由 `TRAJDL_RAYON_THREADS` 控制（专用线程池，
//! 不与 polars/torch 争核）。

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

/// Python float modulo: the result sign follows the divisor (Rust's f64 `%` sign follows the dividend instead).
fn py_mod(x: f64, m: f64) -> f64 {
    let r = x % m;
    if r == 0.0 {
        r.copysign(m)
    } else if (r < 0.0) != (m < 0.0) {
        r + m
    } else {
        r
    }
}

/// Python `int(x)`: truncation toward zero.
fn py_int(x: f64) -> i64 {
    if !x.is_finite() || x.abs() >= 9.0e18 {
        panic!("tale forest: float out of i64 range: {x}");
    }
    x as i64
}

/// Bit-for-bit replica of `gen_all_slots` (tale.py:43-89). `minute` is the floating-point value of seconds/60.
fn gen_all_slots(minute: f64, slice_len: i64, influ_len: i64) -> (Vec<i64>, Vec<f64>) {
    let tsl = slice_len as f64;
    let cal_slice = |x: f64| -> i64 { py_int(py_mod(x, 1440.0) / tsl) };

    if influ_len == 0 {
        return (vec![cal_slice(minute)], vec![1.0]);
    }

    let influ = influ_len as f64;
    let lo = minute - influ / 2.0;
    let hi = minute + influ / 2.0;
    let start = (py_int(lo / tsl) + 1) * slice_len;
    let stop = py_int(hi);
    let mut floors = vec![lo, hi];
    let mut t = start;
    while t < stop {
        floors.push(t as f64);
        t += slice_len;
    }
    floors.sort_by(|a, b| a.partial_cmp(b).expect("tale forest: NaN in minute floors"));
    floors.dedup();
    let n = floors.len();
    let slices = floors[..n - 1].iter().map(|&t| cal_slice(t)).collect();
    let props = (0..n - 1)
        .map(|i| (floors[i + 1] - floors[i]) / influ)
        .collect();
    (slices, props)
}

#[derive(Debug, Clone)]
struct TreeNode {
    id: i64,
    freq: i64,
    left: Option<usize>,
    right: Option<usize>,
    parent: Option<(usize, u8)>, // (父 arena 下标, code 位: 左=1 右=0)
}

/// Construction result of a single tree. `poi_ids` is aligned with the pos/neg rows and holds the poi ids within the slot in ascending order.
#[derive(Debug, Clone)]
struct SlotTree {
    num_inner_nodes: i64,
    poi_ids: Vec<i64>,
    pos: Vec<Vec<i64>>,
    neg: Vec<Vec<i64>>,
}

/// Huffman construction via linear-scan double-minimum selection, a bit-for-bit replica of `HuffmanTree._build_tree`.
fn build_slot_tree(freq: &[(i64, i64)]) -> SlotTree {
    let n = freq.len();
    debug_assert!(n > 0);
    debug_assert!(freq.windows(2).all(|w| w[0].0 < w[1].0), "ids must ascend");

    let id_offset = freq[n - 1].0; // id 升序 → 末位即最大
    let mut arena: Vec<TreeNode> = freq
        .iter()
        .map(|&(id, f)| TreeNode {
            id,
            freq: f,
            left: None,
            right: None,
            parent: None,
        })
        .collect();
    let mut ordered: Vec<usize> = (0..n).collect();
    let mut offset = id_offset;
    let mut inner_count = 0i64;

    while ordered.len() > 1 {
        let mut i1 = 0usize;
        let mut i2 = 1usize;
        if arena[ordered[1]].freq < arena[ordered[0]].freq {
            std::mem::swap(&mut i1, &mut i2);
        }
        for i in 2..ordered.len() {
            if arena[ordered[i]].freq < arena[ordered[i2]].freq {
                i2 = i;
                if arena[ordered[i2]].freq < arena[ordered[i1]].freq {
                    std::mem::swap(&mut i1, &mut i2);
                }
            }
        }
        offset += 1;
        let a = ordered[i1]; // 扫描序最小（打平时先者胜）
        let b = ordered[i2];
        let (left, right) = if arena[a].freq >= arena[b].freq {
            (a, b)
        } else {
            (b, a)
        };
        arena.push(TreeNode {
            id: offset,
            freq: arena[a].freq + arena[b].freq,
            left: Some(left),
            right: Some(right),
            parent: None,
        });
        let fi = arena.len() - 1;
        arena[left].parent = Some((fi, 1));
        arena[right].parent = Some((fi, 0));
        inner_count += 1;

        // 与 Python 一致：按下标先弹大再弹小，father 插到头部
        if i1 < i2 {
            ordered.remove(i2);
            ordered.remove(i1);
        } else {
            ordered.remove(i1);
            ordered.remove(i2);
        }
        ordered.insert(0, fi);
    }

    // 每个叶子：自底向上收集 (father_id, bit)，反转即根→父的 path
    let mut pos = Vec::with_capacity(n);
    let mut neg = Vec::with_capacity(n);
    for leaf in 0..n {
        let mut path = Vec::new();
        let mut bits = Vec::new();
        let mut cur = leaf;
        while let Some((p, bit)) = arena[cur].parent {
            path.push(arena[p].id);
            bits.push(bit);
            cur = p;
        }
        path.reverse();
        bits.reverse();
        let mut p_ids = Vec::new();
        let mut n_ids = Vec::new();
        for (node_id, bit) in path.into_iter().zip(bits) {
            if bit == 1 {
                p_ids.push(node_id - id_offset);
            } else {
                n_ids.push(node_id - id_offset);
            }
        }
        pos.push(p_ids);
        neg.push(n_ids);
    }

    SlotTree {
        num_inner_nodes: inner_count,
        poi_ids: freq.iter().map(|&(id, _)| id).collect(),
        pos,
        neg,
    }
}

/// Forest-build artifacts: all flat arrays (friendly to persistence / zero-copy handoff).
#[derive(Debug, Clone, Default)]
pub struct TaleForestArrays {
    /// Max visited slot id + 1 (Python `num_slots`).
    pub num_slots: i64,
    /// Total number of nodes across all trees + 1 (id 0 reserved for padding, Python `num_inner_nodes`).
    pub num_inner_nodes: i64,
    pub slice_len: i64,
    pub influ_len: i64,
    pub max_poi_id: i64,
    /// Slot id → global inner node id offset; accumulated in slot first-appearance order (empty slots carry the running value, unreachable).
    pub slot_offset: Vec<i64>,
    /// Slot CSR: tree_poi_start[num_slots+1] → tree_poi_flat (poi ids within a slot in ascending order).
    pub tree_poi_start: Vec<i64>,
    pub tree_poi_flat: Vec<i64>,
    /// poi-row CSR: pos_start[len(tree_poi_flat)+1] → pos_flat (local 1-based ids).
    pub pos_start: Vec<i64>,
    pub pos_flat: Vec<i64>,
    pub neg_start: Vec<i64>,
    pub neg_flat: Vec<i64>,
    /// Slot/prop CSR of the (poi, minute) rows (aligned with the seen_minute_flat row order;
    /// each row value is the last write of that (poi, minute) in the point stream, matching Python dict semantics).
    pub visit_slice_start: Vec<i64>,
    pub visit_slice_flat: Vec<i64>,
    pub visit_prop_flat: Vec<f64>,
    /// (poi, minute) co-occurrence CSR (poi row order, minutes ascending and deduplicated within a row).
    pub seen_minute_start: Vec<i64>,
    pub seen_minute_flat: Vec<i64>,
}

/// Main entry point of the forest build.
///
/// `offsets` is the per-sentence prefix offsets (length = number of sentences + 1); `flat_ids`/`flat_seconds`
/// are the token ids and epoch seconds of all points.
pub fn build_tale_forest(
    flat_ids: &[i64],
    flat_seconds: &[i64],
    offsets: &[i64],
    slice_len: i64,
    influ_len: i64,
    num_threads: usize,
) -> Result<TaleForestArrays, String> {
    if slice_len <= 0 {
        return Err(format!("slice_len must be positive, got {slice_len}"));
    }
    if influ_len < 0 {
        return Err(format!("influ_len must be non-negative, got {influ_len}"));
    }
    let n_sentences = offsets
        .len()
        .checked_sub(1)
        .ok_or_else(|| "offsets must have at least one element".to_string())?;
    if flat_ids.len() != flat_seconds.len() || flat_ids.len() != offsets[n_sentences] as usize {
        return Err(format!(
            "shape mismatch: ids={}, seconds={}, offsets.total={}",
            flat_ids.len(),
            flat_seconds.len(),
            offsets[n_sentences]
        ));
    }
    if flat_seconds.iter().any(|&s| s < 0) {
        return Err("timestamps must be non-negative epoch seconds".to_string());
    }
    if flat_ids.iter().any(|&p| p < 0) {
        return Err("poi token ids must be non-negative".to_string());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.max(1))
        .build()
        .map_err(|e| format!("failed to build rayon pool: {e}"))?;

    // Phase A：按句并行，产出 (slot, poi) 计数、(poi, minute) 共现、句内槽首现序、
    // 句内 (poi, minute) → (slices, props) 的句内最后写入
    let per_sentence: Vec<PhaseAOut> = pool.install(|| {
        (0..n_sentences)
            .into_par_iter()
            .map(|si| {
                let mut counts: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
                let mut seen: Vec<(i64, i64)> = Vec::new();
                let mut first_slots: Vec<i64> = Vec::new();
                let mut first_seen: HashSet<i64> = HashSet::new();
                // 句内 (poi, minute) 最后写入（influ>0 时同分钟不同秒结果不同）
                let mut visit_latest: HashMap<(i64, i64), (Vec<i64>, Vec<f64>)> = HashMap::new();
                for p in offsets[si] as usize..offsets[si + 1] as usize {
                    let poi = flat_ids[p];
                    let second = flat_seconds[p];
                    let minute_f = second as f64 / 60.0;
                    let (slices, props) = gen_all_slots(minute_f, slice_len, influ_len);
                    for &s in &slices {
                        *counts.entry(s).or_default().entry(poi).or_insert(0) += 1;
                        if first_seen.insert(s) {
                            first_slots.push(s);
                        }
                    }
                    let int_minute = ((second % 86_400) as f64 / 60.0).floor() as i64;
                    seen.push((poi, int_minute));
                    visit_latest.insert((poi, int_minute), (slices, props));
                }
                PhaseAOut {
                    counts,
                    seen,
                    first_slots,
                    visit_latest,
                }
            })
            .collect()
    });

    // 顺序折叠：计数可交换；槽首现序与 minute 最后写入必须保持点流顺序
    let mut global_counts: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
    let mut slot_order: Vec<i64> = Vec::new();
    let mut slot_registered: HashSet<i64> = HashSet::new();
    let mut seen_pairs: Vec<(i64, i64)> = Vec::new();
    let mut visit_map: HashMap<(i64, i64), (Vec<i64>, Vec<f64>)> = HashMap::new();
    for out in per_sentence {
        for (s, m) in out.counts {
            let entry = global_counts.entry(s).or_default();
            for (poi, c) in m {
                *entry.entry(poi).or_insert(0) += c;
            }
        }
        for s in out.first_slots {
            if slot_registered.insert(s) {
                slot_order.push(s);
            }
        }
        seen_pairs.extend(out.seen);
        for (pair, val) in out.visit_latest {
            visit_map.insert(pair, val); // 句序折叠 → 全局点序最后写入者胜出
        }
    }

    let num_slots = slot_order.iter().max().copied().unwrap_or(-1) + 1;
    let max_poi_id = flat_ids.iter().copied().max().unwrap_or(-1);

    // Phase B：每槽并行建树（结果按槽 id 序收集，确定性）
    let trees: Vec<Option<SlotTree>> = pool.install(|| {
        (0..num_slots)
            .into_par_iter()
            .map(|s| match global_counts.get(&s) {
                None => None,
                Some(poi_counts) => {
                    let mut freq: Vec<(i64, i64)> =
                        poi_counts.iter().map(|(&k, &v)| (k, v)).collect();
                    freq.sort_unstable(); // id 升序 = Python sorted(Counter(...).items())
                    Some(build_slot_tree(&freq))
                }
            })
            .collect::<Vec<_>>()
    });

    // 组装：槽偏移（首现序累加）+ 槽/poi CSR
    let mut slot_offset = vec![0i64; num_slots.max(0) as usize];
    let mut running = 0i64;
    let mut total_inner = 0i64;
    for &s in &slot_order {
        slot_offset[s as usize] = running;
        let inner = trees[s as usize]
            .as_ref()
            .map(|t| t.num_inner_nodes)
            .unwrap_or(0);
        running += inner;
        total_inner += inner;
    }

    let mut tree_poi_start = Vec::with_capacity(num_slots.max(0) as usize + 1);
    let mut tree_poi_flat: Vec<i64> = Vec::new();
    let mut pos_start: Vec<i64> = vec![0];
    let mut pos_flat: Vec<i64> = Vec::new();
    let mut neg_start: Vec<i64> = vec![0];
    let mut neg_flat: Vec<i64> = Vec::new();
    tree_poi_start.push(0);
    for slot_tree in &trees {
        if let Some(t) = slot_tree {
            for (poi, p, n) in t
                .poi_ids
                .iter()
                .zip(t.pos.iter())
                .zip(t.neg.iter())
                .map(|((poi, p), n)| (*poi, p, n))
            {
                tree_poi_flat.push(poi);
                pos_flat.extend_from_slice(p);
                neg_flat.extend_from_slice(n);
                pos_start.push(pos_flat.len() as i64);
                neg_start.push(neg_flat.len() as i64);
            }
        }
        tree_poi_start.push(tree_poi_flat.len() as i64);
    }

    // (poi, minute) 共现 CSR；行数 = max_poi_id + 1（空语料时为 0）
    seen_pairs.sort_unstable();
    seen_pairs.dedup();
    let seen_rows = (max_poi_id + 1).max(0) as usize;
    let mut seen_minute_start = vec![0i64; seen_rows + 1];
    let mut seen_minute_flat: Vec<i64> = Vec::new();
    for &(poi, minute) in &seen_pairs {
        seen_minute_flat.push(minute);
        seen_minute_start[poi as usize + 1] += 1;
    }
    for i in 1..seen_minute_start.len() {
        seen_minute_start[i] += seen_minute_start[i - 1];
    }

    // (poi, minute) 行 CSR：与排序去重后的共现行一一对应
    let mut visit_slice_start = Vec::with_capacity(seen_pairs.len() + 1);
    let mut visit_slice_flat: Vec<i64> = Vec::new();
    let mut visit_prop_flat: Vec<f64> = Vec::new();
    visit_slice_start.push(0);
    for pair in &seen_pairs {
        if let Some((slices, props)) = visit_map.get(pair) {
            visit_slice_flat.extend_from_slice(slices);
            visit_prop_flat.extend_from_slice(props);
        }
        visit_slice_start.push(visit_slice_flat.len() as i64);
    }

    Ok(TaleForestArrays {
        num_slots,
        num_inner_nodes: total_inner + 1,
        slice_len,
        influ_len,
        max_poi_id,
        slot_offset,
        tree_poi_start,
        tree_poi_flat,
        pos_start,
        pos_flat,
        neg_start,
        neg_flat,
        visit_slice_start,
        visit_slice_flat,
        visit_prop_flat,
        seen_minute_start,
        seen_minute_flat,
    })
}

struct PhaseAOut {
    counts: HashMap<i64, HashMap<i64, i64>>,
    seen: Vec<(i64, i64)>,
    first_slots: Vec<i64>,
    visit_latest: HashMap<(i64, i64), (Vec<i64>, Vec<f64>)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three leaves with equal frequency: [(1,1),(2,1),(3,1)] → parent 4 (left 1, right 2), parent 5 (left 4, right 3).
    /// Leaf 1: path [5,4] code[1,1] → pos [2,1]; leaf 2: path [5,4] code[1,0] → pos[2] neg[1];
    /// leaf 3: path [5] code[0] → neg [2].
    #[test]
    fn tree_three_leaves_tie() {
        let t = build_slot_tree(&[(1, 1), (2, 1), (3, 1)]);
        assert_eq!(t.num_inner_nodes, 2);
        assert_eq!(t.pos[0], vec![2, 1]);
        assert_eq!(t.neg[0], Vec::<i64>::new());
        assert_eq!(t.pos[1], vec![2]);
        assert_eq!(t.neg[1], vec![1]);
        assert_eq!(t.pos[2], Vec::<i64>::new());
        assert_eq!(t.neg[2], vec![2]);
    }

    /// Unequal frequencies: [(1,1),(2,2)] → node1.freq < node2.freq → left = the higher-frequency one, 2.
    /// Leaf 2: code[1] pos[1]; leaf 1: code[0] neg[1].
    #[test]
    fn tree_two_leaves_unequal() {
        let t = build_slot_tree(&[(1, 1), (2, 2)]);
        assert_eq!(t.num_inner_nodes, 1);
        assert_eq!(t.pos[0], Vec::<i64>::new());
        assert_eq!(t.neg[0], vec![1]);
        assert_eq!(t.pos[1], vec![1]);
        assert_eq!(t.neg[1], Vec::<i64>::new());
    }

    /// Single-leaf tree: no inner nodes, pos/neg both empty.
    #[test]
    fn tree_single_leaf() {
        let t = build_slot_tree(&[(7, 3)]);
        assert_eq!(t.num_inner_nodes, 0);
        assert_eq!(t.pos[0], Vec::<i64>::new());
        assert_eq!(t.neg[0], Vec::<i64>::new());
    }

    /// influ=0: the slot is a pure function of the minute; influ>0: span splitting + proportional normalization.
    #[test]
    fn slot_semantics() {
        let (s, p) = gen_all_slots(240.0, 240, 0);
        assert_eq!(s, vec![1]);
        assert_eq!(p, vec![1.0]);
        // influ=4, minute=240 → floors {238, 240, 242} → 槽 [0, 1]，比例 [0.5, 0.5]
        let (s, p) = gen_all_slots(240.0, 240, 4);
        assert_eq!(s, vec![0, 1]);
        assert_eq!(p, vec![0.5, 0.5]);
        // 跨负值：minute=1.0, influ=4 → lo=-1.0，Python 取模后 1439 分钟 → int(1439/240)=槽5
        let (s, p) = gen_all_slots(1.0, 240, 4);
        assert_eq!(s.first().copied(), Some(5));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// End to end: two points of one sentence fall into different slots (slot 0 and slot 5), single-leaf trees, all offsets 0.
    #[test]
    fn forest_end_to_end_two_slots() {
        let ids = [3, 5];
        let secs = [0, 72_000]; // minute 0 → 槽0；minute 1200 → 槽5
        let out = build_tale_forest(&ids, &secs, &[0, 2], 240, 0, 2).unwrap();
        assert_eq!(out.num_slots, 6);
        assert_eq!(out.num_inner_nodes, 1); // 两棵单叶树 → 0 内节点 + 1 padding
        assert_eq!(out.slot_offset, [0, 0, 0, 0, 0, 0]);
        assert_eq!(out.tree_poi_flat, vec![3, 5]);
        assert_eq!(out.pos_flat, Vec::<i64>::new());
        assert_eq!(out.neg_flat, Vec::<i64>::new());
        // (poi, minute) 行 CSR：行序对齐共现表 [(3,0), (5,1200)]
        assert_eq!(out.visit_slice_flat, vec![0, 5]);
        assert_eq!(out.visit_prop_flat, vec![1.0, 1.0]);
        assert_eq!(out.visit_slice_start, vec![0, 1, 2]);
        // 共现：(3, minute0)、(5, minute1200)
        assert_eq!(out.seen_minute_flat, vec![0, 1200]);
        assert_eq!(out.seen_minute_start, vec![0, 0, 0, 0, 1, 1, 2]);
    }

    /// End to end: two POIs in the same slot with equal frequency → tree built by the tie rule; offsets and local ids assembled correctly.
    #[test]
    fn forest_two_pois_same_slot() {
        let ids = [3, 5];
        let secs = [0, 3_600]; // minute 0 与 60 → 同槽 0
        let out = build_tale_forest(&ids, &secs, &[0, 2], 240, 0, 2).unwrap();
        assert_eq!(out.num_slots, 1);
        assert_eq!(out.num_inner_nodes, 2); // 1 内节点 + 1 padding
        assert_eq!(out.slot_offset, [0]);
        // 等频打平：id 3 先扫描 → left(code1)，id 5 → right(code0)；father id 6
        assert_eq!(out.tree_poi_flat, vec![3, 5]);
        assert_eq!(out.pos_flat, vec![1]); // leaf3: path[6] code1 → 6-5=1
        assert_eq!(out.neg_flat, vec![1]); // leaf5: path[6] code0 → 6-5=1
        assert_eq!(out.pos_start, vec![0, 1, 1]);
        assert_eq!(out.neg_start, vec![0, 0, 1]);
    }
}
