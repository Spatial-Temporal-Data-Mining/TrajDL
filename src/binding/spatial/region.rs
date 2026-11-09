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

use crate::{
    binding::seq::types::PyTrajectoryType,
    err::TrajDLError,
    spatial::{coord::Coordinate, region::RectangleBoundary},
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};
use traj_dist_rs::traits::AsCoord;

#[gen_stub_pyclass]
#[pyclass(eq, name = "RectangleBoundary", module = "trajdl.trajdl_rust")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyRectangleBoundary(pub RectangleBoundary);

#[gen_stub_pymethods]
#[pymethods]
impl PyRectangleBoundary {
    #[new]
    fn new(min_point: Coordinate, max_point: Coordinate) -> PyResult<Self> {
        // Perform validation. Returning an Err here will raise an exception in Python.
        if min_point.x() > max_point.x() || min_point.y() > max_point.y() {
            return Err(PyTypeError::new_err(
                "max_point must be greater than or equal to min_point",
            ));
        }

        Ok(PyRectangleBoundary(RectangleBoundary::new(
            min_point, max_point,
        )))
    }

    /// Decode a boundary from a Python `List[float]`
    ///
    /// Parameters
    /// ----------
    /// data: List[float]
    ///     length 4: min_x, min_y, max_x, max_y
    ///
    /// Returns
    /// ----------
    /// RectangleBoundary
    ///     a RectangleBoundary object
    #[classmethod]
    fn from_list(_cls: &Bound<'_, PyType>, data: Vec<f64>) -> PyResult<Self> {
        if data.len() != 4 {
            return Err(PyErr::new::<PyValueError, _>(
                "Expected a list of 4 f64 values: [min_lng, min_lat, max_lng, max_lat]",
            ));
        }

        Ok(PyRectangleBoundary(RectangleBoundary::new(
            Coordinate::Wgs84 {
                lng: data[0],
                lat: data[1],
            },
            Coordinate::Wgs84 {
                lng: data[2],
                lat: data[3],
            },
        )))
    }

    /// Convert the RectangleBoundary into a List[float]
    ///
    /// Returns
    /// ----------
    /// List[float]
    ///     length 4: min_x, min_y, max_x, max_y
    fn to_list(&self) -> [f64; 4] {
        self.0.to_arr()
    }

    fn __repr__(&self) -> String {
        format!(
            "RectangleBoundary(min_x={}, min_y={}, max_x={}, max_y={})",
            self.0.min_point.x(),
            self.0.min_point.y(),
            self.0.max_point.x(),
            self.0.max_point.y()
        )
    }

    /// Check whether a point lies inside the region
    pub fn in_boundary(&self, coord: &Coordinate) -> bool {
        self.0.in_boundary(coord)
    }

    /// Check whether a trajectory sequence lies inside the region
    pub fn in_boundary_seq(&self, seq: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
        let seq = PyTrajectoryType::try_from(seq)?;
        Ok(self.0.in_boundary_seq(&seq))
    }

    fn to_web_mercator(&self) -> Self {
        PyRectangleBoundary(self.0.to_web_mercator())
    }

    fn to_wgs84(&self) -> Self {
        PyRectangleBoundary(self.0.to_wgs84())
    }

    #[getter]
    fn get_min_x(&self) -> f64 {
        self.0.min_point.x()
    }

    #[getter]
    fn get_min_y(&self) -> f64 {
        self.0.min_point.y()
    }

    #[getter]
    fn get_max_x(&self) -> f64 {
        self.0.max_point.x()
    }

    #[getter]
    fn get_max_y(&self) -> f64 {
        self.0.max_point.y()
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(bincode::serialize(&self.0)
            .map_err(|e| TrajDLError::DataConvertionError(e.to_string()))?)
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.0 = bincode::deserialize(&state)
            .map_err(|e| TrajDLError::DataConvertionError(e.to_string()))?;
        Ok(())
    }

    fn __getnewargs__(&self) -> (Coordinate, Coordinate) {
        (self.0.min_point, self.0.max_point)
    }
}

define_stub_info_gatherer!(stub_info);
