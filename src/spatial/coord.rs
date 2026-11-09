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

use pyo3::{prelude::*, types::PyTuple};
use pyo3_stub_gen::{
    define_stub_info_gatherer, derive::gen_stub_pyclass, derive::gen_stub_pyclass_enum,
    derive::gen_stub_pymethods,
};
use pyo3_stub_gen_derive::gen_stub_pyfunction;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use strum_macros::Display;
use traj_dist_rs::traits::AsCoord;

// 地球半径 (WGS84)
const R: f64 = 6378137.0;

/// Coordinate system type enum
///
/// Identifies the coordinate system of trajectory data; supports WGS84 and Web Mercator.
/// WGS84 is the raw GPS coordinate system (longitude/latitude), Web Mercator is a projected one (meters).
#[gen_stub_pyclass_enum]
#[pyclass(module = "trajdl.trajdl_rust", eq, eq_int)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default, Display)]
pub enum CoordinateSystem {
    /// WGS84 coordinate system (longitude/latitude, default)
    #[default]
    Wgs84,
    /// Web Mercator coordinate system (projected coordinates, unit: meters)
    WebMercator,
}

#[gen_stub_pymethods]
#[pymethods]
impl CoordinateSystem {
    fn __hash__(&self) -> usize {
        match self {
            CoordinateSystem::Wgs84 => 0,
            CoordinateSystem::WebMercator => 1,
        }
    }
}

#[gen_stub_pyclass_enum]
#[pyclass(module = "trajdl.trajdl_rust", eq)]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum Coordinate {
    Wgs84 { lng: f64, lat: f64 },
    WebMercator { x: f64, y: f64 },
}

/// Quickly construct a wgs84 coordinate
///
/// Parameters
/// ----------
/// lng: float
///
/// lat: float
///
/// Returns
/// ----------
/// Coordinate
#[gen_stub_pyfunction]
#[pyfunction]
pub fn wgs84(lng: f64, lat: f64) -> Coordinate {
    Coordinate::Wgs84 { lng, lat }
}

impl Default for Coordinate {
    fn default() -> Self {
        Coordinate::Wgs84 { lng: 0.0, lat: 0.0 }
    }
}

impl AsCoord for Coordinate {
    fn x(&self) -> f64 {
        match self {
            Coordinate::Wgs84 { lng, .. } => *lng,
            Coordinate::WebMercator { x, .. } => *x,
        }
    }

    fn y(&self) -> f64 {
        match self {
            Coordinate::Wgs84 { lng: _, lat } => *lat,
            Coordinate::WebMercator { x: _, y } => *y,
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(module = "trajdl.trajdl_rust")]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GridCoord {
    #[pyo3(get)]
    pub grid_x: i64,

    #[pyo3(get)]
    pub grid_y: i64,
}

impl Coordinate {
    /// Extract the raw [x, y] pair from this coordinate.
    ///
    /// This is intentionally NOT exposed to Python to prevent
    /// users from silently losing coordinate system information.
    /// Use `Coordinate` objects directly in Python instead.
    pub fn to_pair(&self) -> [f64; 2] {
        [self.x(), self.y()]
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Coordinate {
    #[getter]
    fn get_x(&self) -> f64 {
        self.x()
    }

    #[getter]
    fn get_y(&self) -> f64 {
        self.y()
    }

    pub fn to_web_mercator(&self) -> Self {
        match self {
            Self::Wgs84 { lng, lat } => {
                let lng_rad = lng.to_radians();
                let lat_rad = lat.to_radians();
                let x = R * lng_rad;
                let y = R * ((PI / 4.0) + (lat_rad / 2.0)).tan().ln();
                Self::WebMercator { x, y }
            }
            Self::WebMercator { .. } => *self,
        }
    }

    pub fn to_wgs84(&self) -> Self {
        match self {
            Self::WebMercator { x, y } => {
                let lng_deg = (x / R).to_degrees();
                let lat_rad = 2.0 * ((y / R).exp().atan()) - PI / 2.0;
                let lat_deg = lat_rad.to_degrees();
                Self::Wgs84 {
                    lng: lng_deg,
                    lat: lat_deg,
                }
            }
            Self::Wgs84 { .. } => *self,
        }
    }

    #[staticmethod]
    fn from_wgs84(lng: f64, lat: f64) -> Self {
        Coordinate::Wgs84 { lng, lat }
    }

    #[staticmethod]
    fn from_web_mercator(x: f64, y: f64) -> Self {
        Coordinate::WebMercator { x, y }
    }

    // 实现 __reduce__ 来代替 __getstate__/__setstate__
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (constructor_name, args) = match self {
            Coordinate::Wgs84 { lng, lat } => ("from_wgs84", [lng, lat]),
            Coordinate::WebMercator { x, y } => ("from_web_mercator", [x, y]),
        };
        let constructor = py.get_type::<Self>().getattr(constructor_name)?;
        Ok(PyTuple::new(
            py,
            [constructor, PyTuple::new(py, args)?.into_any()],
        )?)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (a, b) = match self {
            Coordinate::Wgs84 { lng, lat } => (*lng, *lat),
            Coordinate::WebMercator { x, y } => (*x, *y),
        };
        let tuple = PyTuple::new(py, [a, b])?;
        Ok(tuple.into_any().try_iter()?.into_any())
    }

    fn __repr__(&self) -> PyResult<String> {
        match self {
            Coordinate::Wgs84 { lng, lat } => Ok(format!("WGS84(lng={}, lat={})", lng, lat)),
            Coordinate::WebMercator { x, y } => Ok(format!("WebMercator(x={}, y={})", x, y)),
        }
    }
}

impl GridCoord {
    /// Construct a GridCoord from a grid symbol
    ///
    /// The grid symbol format is "G_{grid_x}_{grid_y}"
    pub fn from_grid_symbol(grid_symbol: String) -> Self {
        let parts: Vec<&str> = grid_symbol.split('_').collect();
        let err_msg = format!("invalid grid symbol: {}", grid_symbol);
        GridCoord {
            grid_x: parts[1].parse().expect(&err_msg),
            grid_y: parts[2].parse().expect(&err_msg),
        }
    }

    /// Get the grid symbol
    ///
    /// The format is "G_{grid_x}_{grid_y}"
    pub fn get_grid_symbol(&self) -> String {
        format!("G_{}_{}", self.grid_x, self.grid_y)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl GridCoord {
    #[new]
    fn __new__(grid_x: i64, grid_y: i64) -> Self {
        GridCoord {
            grid_x: grid_x,
            grid_y: grid_y,
        }
    }

    /// Construct a GridCoord from a grid symbol
    ///
    /// The grid symbol format is "G_{grid_x}_{grid_y}"
    ///
    /// Parameters
    /// ----------
    /// grid_symbol : str
    ///     Grid symbol string in format "G_{grid_x}_{grid_y}"
    ///
    /// Returns
    /// -------
    /// GridCoord
    ///     The GridCoord object
    #[staticmethod]
    #[pyo3(name = "from_grid_symbol")]
    fn py_from_grid_symbol(grid_symbol: String) -> Self {
        Self::from_grid_symbol(grid_symbol)
    }

    /// Get the grid symbol of the current grid coordinate
    ///
    /// The format is "G_{grid_x}_{grid_y}"
    #[getter]
    fn grid_symbol(&self) -> PyResult<String> {
        Ok(self.get_grid_symbol())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "GridCoord(grid_x={}, grid_y={})",
            self.grid_x, self.grid_y
        ))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let tuple = PyTuple::new(py, [self.grid_x, self.grid_y])?;
        Ok(tuple.into_any().try_iter()?.into_any())
    }
}

define_stub_info_gatherer!(stub_info);
