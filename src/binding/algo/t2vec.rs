// Copyright 2024 All authors of TrajDL
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use numpy::PyReadonlyArray1;
use pyo3::{PyResult, pyfunction};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

/// Bucketing function for t2vec: places sequences into buckets based on their input and output sequence lengths
#[gen_stub_pyfunction]
#[pyfunction]
pub fn bucketize(
    src_lengths: PyReadonlyArray1<i64>,
    src_label_indices: PyReadonlyArray1<i64>,
    trg_lengths: PyReadonlyArray1<i64>,
    src_bound: Vec<i64>,
    trg_bound: Vec<i64>,
) -> PyResult<HashMap<usize, Vec<usize>>> {
    let src_lengths_slice = src_lengths.as_slice()?;
    let src_label_indices_slice = src_label_indices.as_slice()?;
    let trg_lengths_slice = trg_lengths.as_slice()?;

    let result = crate::algo::t2vec::bucketize_logic(
        src_lengths_slice,
        src_label_indices_slice,
        trg_lengths_slice,
        &src_bound,
        &trg_bound,
    );

    Ok(result)
}
