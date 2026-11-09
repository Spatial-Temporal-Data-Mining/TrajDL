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

use crate::binding::seq::container::PyTrajectory;
use crate::err::TrajDLError;
use crate::sequence::loc::view::LocSeqView;
use crate::sequence::point::view::TrajectoryView;
use crate::sequence::traits::LocSequence;
use crate::sequence::traits::PointSequence;
use crate::sequence::traits::ReadSequenceMeta;
use crate::spatial::coord::CoordinateSystem;
use arrow::array::RecordBatch;
use arrow::error::ArrowError;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use pyo3::{Bound, Py, PyAny, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "LocSeqView", module = "trajdl.trajdl_rust")]
#[derive(Debug)]
pub struct PyLocSeqView {
    table: Py<PyAny>,
    idx: usize,
    pub view: LocSeqView,
}

#[gen_stub_pyclass]
#[pyclass(name = "TrajectoryView", module = "trajdl.trajdl_rust")]
#[derive(Debug)]
pub struct PyTrajectoryView {
    table: Py<PyAny>,
    idx: usize,
    pub view: TrajectoryView,
}

impl Clone for PyLocSeqView {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            table: self.table.clone_ref(py),
            idx: self.idx,
            view: self.view.clone(),
        })
    }
}

impl Clone for PyTrajectoryView {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            table: self.table.clone_ref(py),
            idx: self.idx,
            view: self.view.clone(),
        })
    }
}

/// Convert the pyarrow.Table passed in from Python into the Rust version of `Vec<RecordBatch>`
fn pyany_to_vec_record_batch(table: &Bound<'_, PyAny>) -> Result<Vec<RecordBatch>, TrajDLError> {
    ArrowArrayStreamReader::from_pyarrow_bound(table)
        .map_err(|e| {
            TrajDLError::DataConvertionError(format!(
                "convert pyarrow.Table into rust ArrowArrayStreamReader failed with {}",
                e.to_string()
            ))
        })?
        .collect::<Result<Vec<RecordBatch>, ArrowError>>()
        .map_err(|e| {
            TrajDLError::DataConvertionError(format!(
                "convert pyarrow.Table into rust Vec<RecordBatch> failed with {}",
                e.to_string()
            ))
        })
}

/// Macro that generates common boilerplate code for the view types
macro_rules! impl_sequence_methods_for_view {
    // 宏匹配：类型名，和一个包含额外方法的代码块 {}
    ($struct_name:ty, { $($extra_methods:tt)* }) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl $struct_name {
            /// Length of the sequence
            fn __len__(&self) -> usize {
                self.view.len()
            }

            #[getter]
            fn get_entity_id(&self) -> Option<&str> {
                self.view.get_entity_id()
            }

            #[getter]
            fn get_start_ts(&self) -> Option<i64> {
                self.view.get_start_ts()
            }

            #[getter]
            fn get_ts_seq(&self) -> PyResult<Option<Vec<i64>>> {
                Ok(self.view.get_ts_seq()?)
            }

            // 暂时不支持对视图的序列化
            // fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
            //     let dict = PyDict::new(py);
            //     dict.set_item("table", self.table.clone_ref(py))?;
            //     dict.set_item("idx", self.idx)?;
            //     Ok(dict.unbind())
            // }

            // fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
            //     let table = state.get_item("table")?.unwrap().unbind();
            //     let idx = state.get_item("idx")?.unwrap().extract::<usize>()?;
            //     self.table = table;
            //     self.idx = idx;
            //     Ok(())
            // }

            // fn __getnewargs__<'py>(&'py self, py: Python<'py>) -> (&'py Bound<'py, PyAny>, usize) {
            //     (self.table.bind(py), self.idx)
            // }

            // 插入额外的方法
            $($extra_methods)*
        }
    };
}

impl PyLocSeqView {
    pub fn iter_locs(&self) -> impl Iterator<Item = Option<String>> {
        self.view.seq.iter().map(|x| match x {
            None => None,
            Some(s) => Some(s.to_string()),
        })
    }
}

impl_sequence_methods_for_view!(PyLocSeqView, {
    #[new]
    fn new(table: &Bound<'_, PyAny>, idx: usize) -> PyResult<Self> {
        let x = table.clone().unbind();
        Ok(PyLocSeqView {
            table: x,
            idx: idx,
            view: LocSeqView::from_arrow_table(&pyany_to_vec_record_batch(table)?, idx)?,
        })
    }

    #[getter]
    fn get_seq(&self) -> PyResult<Vec<Option<String>>> {
        Ok(self.iter_locs().collect())
    }

    #[getter]
    fn o(&self) -> PyResult<Option<String>> {
        Ok(self.view.o().map(|s| s.to_string()))
    }

    #[getter]
    fn d(&self) -> PyResult<Option<String>> {
        Ok(self.view.d().map(|s| s.to_string()))
    }
});

impl_sequence_methods_for_view!(PyTrajectoryView, {
    #[new]
    #[pyo3(signature = (table, idx, coord_system=None))]
    fn new(
        table: &Bound<'_, PyAny>,
        idx: usize,
        coord_system: Option<CoordinateSystem>,
    ) -> PyResult<Self> {
        let x = table.clone().unbind();
        let coord_system = coord_system.unwrap_or(CoordinateSystem::Wgs84);
        Ok(PyTrajectoryView {
            table: x,
            idx: idx,
            view: TrajectoryView::from_arrow_table_with_coord(
                &pyany_to_vec_record_batch(table)?,
                idx,
                coord_system,
            )?,
        })
    }

    /// Get the seq attribute, returned as List[List[float]]
    #[getter]
    fn get_seq(&self) -> PyResult<Vec<[f64; 2]>> {
        Ok(self.view.iter_seq().collect())
    }

    /// Get the coordinate system of the trajectory
    #[getter]
    fn get_coord_system(&self) -> CoordinateSystem {
        self.view.coord_system()
    }

    /// Convert the current trajectory view into a Trajectory container in the Web Mercator coordinate system
    fn to_web_mercator(&self) -> PyResult<PyTrajectory> {
        let traj = self.view.to_web_mercator()?;
        Ok(PyTrajectory(traj))
    }

    /// Convert the current trajectory view into a Trajectory container in the WGS84 coordinate system
    fn to_wgs84(&self) -> PyResult<PyTrajectory> {
        let traj = self.view.to_wgs84()?;
        Ok(PyTrajectory(traj))
    }
});
