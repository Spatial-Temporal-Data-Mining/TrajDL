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

use pyo3::{Bound, PyAny, PyResult, pyfunction};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::{
    binding::seq::{container::PyTrajectory, types::PyTrajectoryType},
    ops,
};

#[gen_stub_pyfunction]
#[pyfunction]
pub fn rdp(seq: &Bound<'_, PyAny>, epsilon: f64) -> PyResult<PyTrajectory> {
    let seq = PyTrajectoryType::try_from(seq)?;
    Ok(PyTrajectory(ops::simplify::rdp(&seq, epsilon)?))
}
