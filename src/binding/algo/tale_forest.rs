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

//! `SliceHuffmanForest` 构建的 Python 绑定：单函数入口，产物以
//! `dict[str, numpy.ndarray | int]` 返回，Python 侧 `SliceHuffmanForest`
//! 是唯一消费者与公共 API（pickle 兼容、行为逐位一致）。裸数组不属于
//! 公开接口，语义文档见 `src/algo/tale_forest.rs` 模块文档。

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

/// rayon thread count: `TRAJDL_RAYON_THREADS` > default min(#cores, 8).
/// The isolation design uses a dedicated pool, never competing with polars/torch for cores.
fn default_rayon_threads() -> usize {
    if let Ok(s) = std::env::var("TRAJDL_RAYON_THREADS") {
        if let Ok(n) = s.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8)
}

/// Build the TALE `SliceHuffmanForest` artifacts (rayon-parallel, behavior bit-for-bit identical to the Python implementation).
///
/// Keys of the returned dict:
/// - scalars: `num_slots` (max visited slot id + 1), `num_inner_nodes`
///   (total number of nodes across all trees + 1, id 0 reserved for padding), `max_poi_id`
/// - arrays (CSR = `*_start` row bounds + `*_flat` values, i64; `visit_prop_flat` is f64):
///   `slot_offset` (slot id → global inner node id offset, accumulated in slot first-appearance order),
///   `tree_poi_start`/`tree_poi_flat` (slot → poi, ascending within a row = tree-build input order),
///   `pos_start`/`pos_flat`, `neg_start`/`neg_flat` (Huffman paths of the poi rows,
///   1-based local ids within the tree, 0 is the reserved padding value),
///   `visit_slice_start`/`visit_slice_flat`/`visit_prop_flat` (time slots and
///   proportions of the (poi, minute) rows), `seen_minute_start`/`seen_minute_flat` ((poi, minute)
///   co-occurrence table, poi row order, ascending and deduplicated within a row)
///
/// Parameters
/// ----------
/// flat_ids : numpy.ndarray[int64]
///     Token ids of all points (concatenated per sentence).
/// flat_seconds : numpy.ndarray[int64]
///     Epoch seconds of all points (aligned with flat_ids, non-negative).
/// offsets : numpy.ndarray[int64]
///     Per-sentence prefix offsets, length = number of sentences + 1, last element = len(flat_ids).
/// slice_len : int
///     Time-slot length (minutes).
/// influ_len : int
///     Influence span (minutes), 0 = single-slot attribution.
/// num_threads : int, optional
///     rayon thread count; 0 = follow `TRAJDL_RAYON_THREADS` / min(#cores, 8).
///
/// Returns
/// -------
/// dict[str, numpy.ndarray | int]
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (flat_ids, flat_seconds, offsets, slice_len, influ_len, num_threads=0))]
pub fn build_tale_huffman_forest<'py>(
    py: Python<'py>,
    flat_ids: PyReadonlyArray1<'_, i64>,
    flat_seconds: PyReadonlyArray1<'_, i64>,
    offsets: PyReadonlyArray1<'_, i64>,
    slice_len: i64,
    influ_len: i64,
    num_threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let threads = if num_threads == 0 {
        default_rayon_threads()
    } else {
        num_threads.max(1)
    };
    let inner = match crate::algo::tale_forest::build_tale_forest(
        flat_ids.as_slice()?,
        flat_seconds.as_slice()?,
        offsets.as_slice()?,
        slice_len,
        influ_len,
        threads,
    ) {
        Ok(inner) => inner,
        Err(e) => return Err(PyValueError::new_err(format!("InvalidParams: {e}"))),
    };

    let result = PyDict::new(py);
    result.set_item("num_slots", inner.num_slots)?;
    result.set_item("num_inner_nodes", inner.num_inner_nodes)?;
    result.set_item("max_poi_id", inner.max_poi_id)?;
    result.set_item("slot_offset", PyArray1::from_vec(py, inner.slot_offset))?;
    result.set_item(
        "tree_poi_start",
        PyArray1::from_vec(py, inner.tree_poi_start),
    )?;
    result.set_item("tree_poi_flat", PyArray1::from_vec(py, inner.tree_poi_flat))?;
    result.set_item("pos_start", PyArray1::from_vec(py, inner.pos_start))?;
    result.set_item("pos_flat", PyArray1::from_vec(py, inner.pos_flat))?;
    result.set_item("neg_start", PyArray1::from_vec(py, inner.neg_start))?;
    result.set_item("neg_flat", PyArray1::from_vec(py, inner.neg_flat))?;
    result.set_item(
        "visit_slice_start",
        PyArray1::from_vec(py, inner.visit_slice_start),
    )?;
    result.set_item(
        "visit_slice_flat",
        PyArray1::from_vec(py, inner.visit_slice_flat),
    )?;
    result.set_item(
        "visit_prop_flat",
        PyArray1::from_vec(py, inner.visit_prop_flat),
    )?;
    result.set_item(
        "seen_minute_start",
        PyArray1::from_vec(py, inner.seen_minute_start),
    )?;
    result.set_item(
        "seen_minute_flat",
        PyArray1::from_vec(py, inner.seen_minute_flat),
    )?;
    Ok(result)
}
