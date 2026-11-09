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

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::PyTuple;
use pyo3_stub_gen_derive::gen_stub_pyclass;
use pyo3_stub_gen_derive::gen_stub_pymethods;

use crate::binding::seq::types::PyTrajectoryType;
use crate::binding::spatial::region::PyRectangleBoundary;
use crate::err::TrajDLError;
use crate::spatial::coord::Coordinate;
use crate::spatial::coord::GridCoord;
use crate::spatial::grid::SimpleGridSystem;
use crate::spatial::grid::UniformDegreeGridSystem;

#[gen_stub_pyclass]
#[pyclass(name = "SimpleGridSystem", module = "trajdl.trajdl_rust")]
pub struct PySimpleGridSystem(pub SimpleGridSystem);

#[gen_stub_pymethods]
#[pymethods]
impl PySimpleGridSystem {
    /// Constructor
    ///
    /// Parameters
    /// ----------
    /// boundary: RectangleBoundary
    ///     boundary
    ///
    /// step_x: f64
    ///     grid length along the x axis
    ///
    /// step_y: f64
    ///     grid length along the y axis
    ///
    /// column_major: bool
    ///     Whether to compute cell_id in column-major order. Defaults to False (row-major).
    ///     Row-major: cell_id = grid_y * num_x_grids + grid_x
    ///     Column-major: cell_id = grid_x * num_y_grids + grid_y
    ///
    /// Returns
    /// ----------
    /// PySimpleGridSystem
    ///     grid system
    #[new]
    #[pyo3(signature = (boundary, step_x, step_y, column_major = false))]
    fn new(boundary: PyRectangleBoundary, step_x: f64, step_y: f64, column_major: bool) -> Self {
        Self::_internal_from_py(boundary, step_x, step_y, column_major)
    }

    #[staticmethod]
    #[pyo3(signature = (boundary, step_x, step_y, column_major = false))]
    fn _internal_from_py(
        boundary: PyRectangleBoundary,
        step_x: f64,
        step_y: f64,
        column_major: bool,
    ) -> Self {
        Self(SimpleGridSystem::with_column_major(
            boundary.0,
            step_x,
            step_y,
            column_major,
        ))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let args = (
            PyRectangleBoundary(self.0.boundary),
            self.0.step_x,
            self.0.step_y,
            self.0.column_major,
        )
            .into_py_any(py)?
            .into_bound(py);
        let constructor = py.get_type::<Self>().getattr("_internal_from_py")?;
        Ok(PyTuple::new(py, [constructor, args])?)
    }

    #[getter]
    fn get_column_major(&self) -> bool {
        self.0.column_major
    }

    #[getter]
    fn get_step_x(&self) -> f64 {
        self.0.step_x
    }

    #[getter]
    fn get_step_y(&self) -> f64 {
        self.0.step_y
    }

    #[getter]
    fn get_min_x(&self) -> f64 {
        self.0.min_x
    }

    #[getter]
    fn get_min_y(&self) -> f64 {
        self.0.min_y
    }

    #[getter]
    fn get_max_x(&self) -> f64 {
        self.0.max_x
    }

    #[getter]
    fn get_max_y(&self) -> f64 {
        self.0.max_y
    }

    #[getter]
    fn get_num_x_grids(&self) -> i64 {
        self.0.num_x_grids
    }

    #[getter]
    fn get_num_y_grids(&self) -> i64 {
        self.0.num_y_grids
    }

    #[getter]
    fn get_boundary(&self) -> PyRectangleBoundary {
        PyRectangleBoundary(self.0.boundary)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PySimpleGridSystem(boundary={:?}, step_x={}, step_y={})",
            self.0.boundary, self.0.step_x, self.0.step_y
        ))
    }

    pub fn in_boundary(&self, coord: &Coordinate) -> bool {
        self.0.in_boundary(coord)
    }

    pub fn in_boundary_seq(&self, seq: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
        Ok(self.0.in_boundary_seq(&PyTrajectoryType::try_from(seq)?))
    }

    /// Locate the grid id by grid coordinate
    ///
    /// Parameters
    /// ----------
    /// grid_coord: GridCoord
    ///     grid coordinate
    ///
    /// Returns
    /// ----------
    /// int
    ///     grid id
    pub fn locate_by_grid_coordinate(&self, grid_coord: &GridCoord) -> i64 {
        self.0.locate_by_grid_coordinate(grid_coord)
    }

    /// Convert a longitude/latitude coordinate into a grid coordinate
    ///
    /// Parameters
    /// ----------
    /// coord: Coordinate
    ///     coordinate
    ///
    /// Returns
    /// ----------
    /// GridCoord
    ///     grid coordinate
    pub fn transform_coord_into_grid_coord(&self, coord: &Coordinate) -> GridCoord {
        self.0.transform_coord_into_grid_coord(coord)
    }

    /// Convert a longitude/latitude coordinate into the grid's token symbol
    ///
    /// Converts the coordinate into a grid coordinate, then generates the corresponding token symbol (format "G_{grid_x}_{grid_y}").
    ///
    /// Parameters
    /// ----------
    /// coord: Coordinate
    ///     longitude/latitude coordinate
    ///
    /// Returns
    /// ----------
    /// str
    ///     the grid's grid symbol, format "G_{grid_x}_{grid_y}"
    pub fn coord_to_grid_symbol(&self, coord: &Coordinate) -> String {
        self.0
            .transform_coord_into_grid_coord(coord)
            .get_grid_symbol()
    }

    /// Convert a longitude/latitude sequence into a grid-coordinate sequence
    /// TODO: complete the docstring
    pub fn transform_coord_into_grid_coord_seq(
        &self,
        seq: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<GridCoord>> {
        Ok(self
            .0
            .transform_coord_into_grid_coord_seq(&PyTrajectoryType::try_from(seq)?)?)
    }

    /// Convert a longitude/latitude coordinate into a grid id
    ///
    /// Parameters
    /// ----------
    /// coord: Coordinate
    ///     coordinate
    ///
    /// Returns
    /// ----------
    /// int
    ///     grid id
    pub fn locate_in_grid(&self, coord: &Coordinate) -> i64 {
        self.0.locate_in_grid(coord)
    }

    /// Convert a longitude/latitude sequence into grid ids (single-trajectory version)
    ///
    /// Parameters
    /// ----------
    /// seq: Trajectory
    ///     trajectory sequence
    ///
    /// unk_loc: Optional[int]
    ///     Optional grid id used when a longitude/latitude point falls outside the region. If no value is passed (`None`), the value computed by the grid system is used (which may be negative)
    ///
    /// Returns
    /// ----------
    /// List[List[int]]
    ///     grid id that each longitude/latitude coordinate is converted into
    #[pyo3(signature = (seq, unk_loc = None))]
    pub fn locate_in_grid_seq(
        &self,
        seq: &Bound<'_, PyAny>,
        unk_loc: Option<i64>,
    ) -> PyResult<Vec<i64>> {
        Ok(self
            .0
            .locate_in_grid_seq(&PyTrajectoryType::try_from(seq)?, unk_loc))
    }

    /// Check whether a grid coordinate lies inside the region
    /// TODO: complete the docstring
    pub fn in_boundary_by_grid_coordinate(&self, grid_coord: &GridCoord) -> bool {
        self.0.in_boundary_by_grid_coordinate(grid_coord)
    }

    /// Convert a grid id into a grid coordinate
    ///
    /// Parameters
    /// ----------
    /// loc: str
    ///     grid id
    ///
    /// Returns
    /// ----------
    /// GridCoord
    ///     grid coordinate
    pub fn to_grid_coordinate(&self, loc: &str) -> PyResult<GridCoord> {
        let cell_id = loc
            .parse::<i64>()
            .map_err(|e| TrajDLError::DataConvertionError(e.to_string()))?;

        Ok(self.0.to_grid_coordinate(cell_id)?)
    }

    /// Convert a grid coordinate into the centroid coordinate of that grid
    ///
    /// Parameters
    /// ----------
    /// grid_coord: GridCoord
    ///     grid coordinate
    ///
    /// Returns
    /// ----------
    /// Coordinate
    ///     coordinate of the grid centroid
    pub fn grid_coord_to_centroid_point(&self, grid_coord: &GridCoord) -> Coordinate {
        self.0.grid_coord_to_centroid_point(grid_coord)
    }

    /// Convert a group of grid symbols into their grid centroid coordinates
    ///
    /// A grid symbol has the format "G_{grid_x}_{grid_y}"; this method parses it into a grid coordinate,
    /// then computes and returns the centroid coordinate of that grid.
    ///
    /// Parameters
    /// ----------
    /// grid_symbols: List[str]
    ///     list of grid symbols, each element in the format "G_{grid_x}_{grid_y}"
    ///
    /// Returns
    /// ----------
    /// List[Coordinate]
    ///     list of corresponding grid centroid coordinates
    fn grid_symbols_to_centroid_coords(&self, grid_symbols: Vec<String>) -> Vec<Coordinate> {
        self.0.grid_symbols_to_centroid_coords(&grid_symbols)
    }

    /// Convert one grid symbol into its grid centroid coordinate
    ///
    /// A grid symbol has the format "G_{grid_x}_{grid_y}"; this method parses it into a grid coordinate,
    /// then computes and returns the centroid coordinate of that grid.
    ///
    /// Parameters
    /// ----------
    /// grid_symbol: str
    ///     grid symbol, format "G_{grid_x}_{grid_y}"
    ///
    /// Returns
    /// ----------
    /// Coordinate
    ///     the corresponding grid centroid coordinate
    fn grid_symbol_to_centroid_coord(&self, grid_symbol: String) -> Coordinate {
        self.0.grid_symbol_to_centroid_coord(grid_symbol)
    }

    /// Enumerate all grid symbols in the grid system, producing a token id sequence
    /// exactly identical to ``locate_by_grid_coordinate`` according to the current ``column_major`` setting.
    ///
    /// The return value is a ``List[Tuple[str, int]]`` in enumeration order:
    ///   - ``column_major=True`` (column-major): outer ``i_x ∈ [0, num_x)``, inner
    ///     ``i_y ∈ [0, num_y)``, the id of the ``k``-th element equals ``i_x * num_y + i_y == k``.
    ///   - ``column_major=False`` (row-major): outer ``i_y ∈ [0, num_y)``, inner
    ///     ``i_x ∈ [0, num_x)``, the id of the ``k``-th element equals ``i_y * num_x + i_x == k``.
    ///
    /// This method guarantees: length equals ``num_x_grids * num_y_grids``; ids
    /// ``0..num_x_grids * num_y_grids`` are contiguous with no gaps; ids match
    /// ``locate_by_grid_coordinate(GridCoord{grid_x, grid_y})`` exactly;
    /// symbols and ids are aligned in both directions. The Python side can directly do ``dict(grid.enumerate_grid_symbols())``
    /// to obtain an ordered ``Dict[str, int]`` to feed into the tokenizer vocabulary.
    ///
    /// Returns
    /// ----------
    /// List[Tuple[str, int]]
    ///     ``(grid_symbol, token_id)`` list, length equals ``num_x_grids * num_y_grids``.
    pub fn enumerate_grid_symbols(&self) -> Vec<(String, i64)> {
        self.0.enumerate_grid_symbols()
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
}

/// Python binding of the uniform degree-space grid system (the UniTE basis).
///
/// For the differences from [`PySimpleGridSystem`] see the `UniformDegreeGridSystem` docs: the binning
/// formula is `ceil((x - min) / span * num) - 1` + clip (boundary points go to the lower cell), and the
/// bounds are fitted from the data; tokens are row-major `grid_y * num_x + grid_x`, bit-for-bit identical
/// to UniTE `project_to_grid`. The locating interfaces directly accept raw longitude/latitude scalars/arrays
/// (the UniTE pipeline projects column-wise in batches and does not construct Coordinate objects).
#[gen_stub_pyclass]
#[pyclass(name = "UniformDegreeGridSystem", module = "trajdl.trajdl_rust")]
pub struct PyUniformDegreeGridSystem(pub UniformDegreeGridSystem);

#[gen_stub_pymethods]
#[pymethods]
impl PyUniformDegreeGridSystem {
    /// Constructor
    ///
    /// Parameters
    /// ----------
    /// min_x: f64
    ///     lower bound of the x axis (longitude) (value fitted from the data)
    /// max_x: f64
    ///     upper bound of the x axis (longitude)
    /// min_y: f64
    ///     lower bound of the y axis (latitude)
    /// max_y: f64
    ///     upper bound of the y axis (latitude)
    /// delta: f64
    ///     grid step (degrees), 0.0015 on the UniTE basis
    #[new]
    fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64, delta: f64) -> Self {
        Self(UniformDegreeGridSystem::new(
            min_x, max_x, min_y, max_y, delta,
        ))
    }

    #[staticmethod]
    fn _internal_from_py(min_x: f64, max_x: f64, min_y: f64, max_y: f64, delta: f64) -> Self {
        Self(UniformDegreeGridSystem::new(
            min_x, max_x, min_y, max_y, delta,
        ))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let args = (
            self.0.min_x,
            self.0.max_x,
            self.0.min_y,
            self.0.max_y,
            self.0.delta,
        )
            .into_py_any(py)?
            .into_bound(py);
        let constructor = py.get_type::<Self>().getattr("_internal_from_py")?;
        Ok(PyTuple::new(py, [constructor, args])?)
    }

    #[getter]
    fn get_delta(&self) -> f64 {
        self.0.delta
    }

    #[getter]
    fn get_min_x(&self) -> f64 {
        self.0.min_x
    }

    #[getter]
    fn get_max_x(&self) -> f64 {
        self.0.max_x
    }

    #[getter]
    fn get_min_y(&self) -> f64 {
        self.0.min_y
    }

    #[getter]
    fn get_max_y(&self) -> f64 {
        self.0.max_y
    }

    #[getter]
    fn get_num_x_grids(&self) -> i64 {
        self.0.num_x_grids
    }

    #[getter]
    fn get_num_y_grids(&self) -> i64 {
        self.0.num_y_grids
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PyUniformDegreeGridSystem(delta={}, x=({}, {}), y=({}, {}), num_x={}, num_y={})",
            self.0.delta,
            self.0.min_x,
            self.0.max_x,
            self.0.min_y,
            self.0.max_y,
            self.0.num_x_grids,
            self.0.num_y_grids
        ))
    }

    /// Check whether a raw coordinate lies inside the data bounds
    pub fn in_bounds(&self, x: f64, y: f64) -> bool {
        self.0.in_bounds(x, y)
    }

    /// Convert a raw coordinate into a grid id
    ///
    /// Parameters
    /// ----------
    /// x: f64
    ///     longitude
    /// y: f64
    ///     latitude
    ///
    /// Returns
    /// ----------
    /// int
    ///     cell id (= grid_y * num_x_grids + grid_x, bit-for-bit identical to UniTE)
    pub fn locate_in_grid(&self, x: f64, y: f64) -> i64 {
        self.0.locate_in_grid(x, y)
    }

    /// Batch locating: equal-length longitude/latitude arrays → cell id array (sequential implementation)
    ///
    /// Parameters
    /// ----------
    /// xs: List[float]
    ///     longitude array
    /// ys: List[float]
    ///     latitude array
    ///
    /// Returns
    /// ----------
    /// List[int]
    ///     cell id array
    pub fn locate_in_grid_batch(&self, xs: Vec<f64>, ys: Vec<f64>) -> PyResult<Vec<i64>> {
        Ok(self.0.locate_in_grid_batch(&xs, &ys)?)
    }

    /// Convert a raw coordinate into a grid token symbol (format "G_{grid_x}_{grid_y}")
    pub fn coord_to_grid_symbol(&self, x: f64, y: f64) -> String {
        self.0
            .transform_coord_into_grid_coord(x, y)
            .get_grid_symbol()
    }

    /// Convert a cell id into a grid coordinate
    pub fn to_grid_coordinate(&self, cell_id: i64) -> PyResult<GridCoord> {
        Ok(self.0.to_grid_coordinate(cell_id)?)
    }

    /// Convert a grid coordinate into the centroid coordinate
    pub fn grid_coord_to_centroid_point(&self, grid_coord: &GridCoord) -> Coordinate {
        self.0.grid_coord_to_centroid_point(grid_coord)
    }

    /// Enumerate all cells: ``List[Tuple[str, int]]``, row-major with contiguous ids,
    /// so ``UniteGridTokenizer`` can feed the vocabulary directly.
    pub fn enumerate_grid_symbols(&self) -> Vec<(String, i64)> {
        self.0.enumerate_grid_symbols()
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
}
