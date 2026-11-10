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

//! PyO3 wrappers around the canonical Arrow schemas and `RecordBatch`
//! builders defined in `crate::datasets`.

use arrow::pyarrow::ToPyArrow;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::binding::seq::container::{PyLocSeq, PyTrajectory};
use crate::datasets::builder::{build_loc_seq_record_batch, build_traj_record_batch};
use crate::datasets::schema::{loc_seq_schema, traj_schema};

/// Return the canonical `Trajectory` Arrow schema as a `pyarrow.Schema`.
///
/// The Python side imports this via `python/trajdl/datasets/arrow/schema.py`
/// so Rust remains the single source of truth.
#[gen_stub_pyfunction]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "pyarrow.Schema", imports = ("pyarrow")))]
pub fn get_traj_schema(py: Python<'_>) -> PyResult<PyObject> {
    traj_schema().to_pyarrow(py)
}

/// Return the canonical `LocSeq` Arrow schema as a `pyarrow.Schema`.
#[gen_stub_pyfunction]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "pyarrow.Schema", imports = ("pyarrow")))]
pub fn get_loc_seq_schema(py: Python<'_>) -> PyResult<PyObject> {
    loc_seq_schema().to_pyarrow(py)
}

/// Build a `pyarrow.RecordBatch` from a Python list of `Trajectory`.
///
/// This is the fast path called by `TrajectoryDataset.init_from_seqs`.
/// The returned RecordBatch's underlying Arrow buffers were allocated by
/// Rust; ownership is handed to Python via the Arrow C Data Interface,
/// which uses an internal `release` callback to drop the Rust-side `Arc`
/// when the Python object is garbage-collected. No data is copied.
#[gen_stub_pyfunction]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "pyarrow.RecordBatch", imports = ("pyarrow")))]
pub fn build_traj_record_batch_py(
    py: Python<'_>,
    trajs: Vec<PyRef<'_, PyTrajectory>>,
) -> PyResult<PyObject> {
    // Borrow the inner Trajectory references without cloning.
    let inner: Vec<&crate::sequence::point::container::Trajectory> =
        trajs.iter().map(|t| &t.0).collect();
    // SAFETY: `build_traj_record_batch` accepts `&[Trajectory]`; we collect
    // refs here purely to avoid clones. Lifetime is bounded by `trajs`.
    let owned: Vec<crate::sequence::point::container::Trajectory> =
        inner.into_iter().cloned().collect();
    let batch = build_traj_record_batch(&owned)
        .map_err(|e| PyValueError::new_err(format!("build_traj_record_batch failed: {e}")))?;
    batch.to_pyarrow(py)
}

/// Build a `pyarrow.RecordBatch` from a Python list of `LocSeq`.
///
/// See [`build_traj_record_batch_py`] for memory model details.
#[gen_stub_pyfunction]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "pyarrow.RecordBatch", imports = ("pyarrow")))]
pub fn build_loc_seq_record_batch_py(
    py: Python<'_>,
    seqs: Vec<PyRef<'_, PyLocSeq>>,
) -> PyResult<PyObject> {
    let owned: Vec<crate::sequence::loc::container::LocSeq> =
        seqs.iter().map(|s| s.0.clone()).collect();
    let batch = build_loc_seq_record_batch(&owned)
        .map_err(|e| PyValueError::new_err(format!("build_loc_seq_record_batch failed: {e}")))?;
    batch.to_pyarrow(py)
}
