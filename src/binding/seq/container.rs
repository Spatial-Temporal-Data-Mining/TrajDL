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

use numpy::{
    PyArray, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray2, PyUntypedArray,
    PyUntypedArrayMethods, dtype,
};
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyIndexError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyList, PyListMethods, PyType},
};
use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::{
    err::TrajDLError,
    sequence::{
        base::SequenceMeta,
        loc::container::LocSeq,
        point::container::Trajectory,
        traits::{LocSequence, ReadSequenceMeta, WriteSequenceMeta},
    },
    spatial::coord::CoordinateSystem,
};

#[gen_stub_pyclass]
#[pyclass(eq, name = "LocSeq", module = "trajdl.trajdl_rust")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyLocSeq(pub LocSeq);

#[gen_stub_pyclass]
#[pyclass(eq, name = "Trajectory", module = "trajdl.trajdl_rust")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyTrajectory(pub Trajectory);

impl From<LocSeq> for PyLocSeq {
    fn from(value: LocSeq) -> Self {
        Self(value)
    }
}

impl From<Trajectory> for PyTrajectory {
    fn from(value: Trajectory) -> Self {
        Self(value)
    }
}

macro_rules! impl_py_methods_for_container {
    // 宏匹配：类型名，和一个包含额外方法的代码块 {}
    ($struct_name:ty, { $($extra_methods:tt)* }) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl $struct_name {
            /// Length of the sequence
            fn __len__(&self) -> usize {
                self.0.seq.len()
            }

            #[getter]
            fn get_entity_id(&self) -> Option<&str> {
                self.0.get_entity_id()
            }

            #[setter]
            fn set_entity_id(&mut self, entity_id: Option<String>) {
                self.0.set_entity_id(entity_id)
            }

            #[getter]
            fn get_start_ts(&self) -> Option<i64> {
                self.0.get_start_ts()
            }

            #[setter]
            fn set_start_ts(&mut self, start_ts: Option<i64>) {
                self.0.set_start_ts(start_ts)
            }

            #[getter]
            fn get_ts_seq(&self) -> PyResult<Option<Vec<i64>>> {
                Ok(self.0.get_ts_seq()?)
            }

            #[setter]
            fn set_ts_seq(&mut self, ts_seq: Option<Vec<i64>>) {
                self.0.set_ts_seq(ts_seq)
            }

            // 插入额外的方法
            $($extra_methods)*
        }
    };
}

impl_py_methods_for_container!(PyLocSeq, {
    /// Constructor
    ///
    /// Returns
    /// ----------
    /// An empty LocSeq
    #[new]
    fn new() -> Self {
        Self(Default::default())
    }

    /// TODO: fix this comment. Build from a Python `List[str]`
    ///
    /// Parameters
    /// ----------
    /// seq: List[str]
    ///     Location sequence data, each element is the string representation of a location
    ///
    /// entity_id: Optional[str]
    ///     Trajectory id
    ///
    /// Returns
    /// ----------
    /// Sequence
    ///     Location sequence
    #[classmethod]
    #[pyo3(signature = (seq, entity_id=None, ts_seq=None))]
    fn from_list(
        _cls: &Bound<'_, PyType>,
        seq: &Bound<'_, PyList>,
        entity_id: Option<String>,
        ts_seq: Option<Vec<i64>>,
    ) -> PyResult<Self> {
        let mut loc_vec = Vec::with_capacity(seq.len());
        for loc in seq {
            loc_vec.push(loc.extract::<Option<String>>()?);
        }

        if let Some(ref ts_seq_vec) = ts_seq {
            if ts_seq_vec.len() != loc_vec.len() {
                return Err(TrajDLError::InvalidParams(
                    "'seq' and 'ts_seq' must have same length".to_string(),
                )
                .into());
            }
            for window in ts_seq_vec.windows(2) {
                if window[0] > window[1] {
                    return Err(TrajDLError::InvalidParams(
                        "'ts_seq' should be a monotonically increasing List[int].".to_string(),
                    )
                    .into());
                }
            }
        }

        let r = LocSeq {
            meta: SequenceMeta {
                entity_id: entity_id,
                start_ts: None,
                ts_seq: ts_seq,
            },
            seq: loc_vec,
        };
        Ok(Self(r))
    }

    // TODO: to be optimized
    fn __repr__(&self) -> PyResult<String> {
        let length = self.__len__();
        let size_repr = format!("size={}", length);
        let entity_id_expr = if let Some(entity_id) = self.get_entity_id() {
            format!(", entity_id='{}'", entity_id)
        } else {
            "".to_string()
        };
        let loc_seq_expr = if length > 0 {
            let elements = self
                .get_seq()?
                .iter()
                .take(3)
                .map(|loc| match loc {
                    Some(loc) => format!("'{}'", loc),
                    None => "null".to_string(),
                })
                .collect::<Vec<String>>()
                .join(", ");
            format!(
                ", loc_seq={}{}",
                elements,
                if length > 3 {
                    ", ...".to_string()
                } else {
                    "".to_string()
                }
            )
        } else {
            "".to_string()
        };

        let ts_seq_expr = if let Some(ts_seq) = self.get_ts_seq()? {
            let elements = ts_seq
                .iter()
                .take(3)
                .map(|ts| format!("{}", ts))
                .collect::<Vec<String>>()
                .join(", ");
            format!(
                ", ts_seq={}{}",
                elements,
                if length > 3 {
                    ", ...".to_string()
                } else {
                    "".to_string()
                }
            )
        } else {
            "".to_string()
        };

        Ok(format!(
            "LocSeq({}{}{}{})",
            size_repr, entity_id_expr, loc_seq_expr, ts_seq_expr
        ))
    }

    #[getter]
    fn get_seq(&self) -> PyResult<Vec<Option<String>>> {
        Ok(self.0.seq.clone())
    }

    #[setter]
    fn set_seq(&mut self, seq: Vec<Option<String>>) {
        self.0.seq = seq;
    }

    fn __getitem__(&self, idx: i64) -> PyResult<Option<String>> {
        let len = self.0.seq.len() as i64;

        // 处理负数索引：-1 是最后一个，-len 是第一个，超出则报错
        let index = if idx < 0 { len + idx } else { idx };

        if index < 0 || index >= len {
            Err(PyIndexError::new_err("index out of range"))
        } else {
            Ok(self.0.seq[index as usize].clone())
        }
    }

    /// Get the start point of the location sequence
    #[getter]
    fn o(&self) -> PyResult<Option<String>> {
        if self.0.len() > 0 {
            self.__getitem__(0)
        } else {
            Ok(None)
        }
    }

    /// Get the end point of the location sequence
    #[getter]
    fn d(&self) -> PyResult<Option<String>> {
        if self.0.len() > 0 {
            self.__getitem__(-1)
        } else {
            Ok(None)
        }
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
});

pub fn pylist_to_locseq(seq: &Bound<'_, PyList>) -> PyResult<LocSeq> {
    let mut loc_vec = Vec::with_capacity(seq.len());
    for loc in seq {
        loc_vec.push(loc.extract::<Option<String>>()?);
    }
    Ok(LocSeq::from_vec(loc_vec))
}

/// Convert a longitude/latitude sequence expressed as a Python `List[List[float]]` into a Trajectory
pub fn pylist_to_trajectory(
    seq: &Bound<'_, PyList>,
    coord_system: CoordinateSystem,
) -> PyResult<Trajectory> {
    let mut loc_vec = Vec::with_capacity(seq.len());
    for (idx, coord) in seq.iter().enumerate() {
        let coord = coord.extract::<[f64; 2]>().map_err(|_| {
            PyErr::from(TrajDLError::DataConvertionError(format!("Failed to convert element at index {} to [f64; 2]. Expected a sequence of two numbers.", idx)))
        })?;

        loc_vec.push(coord);
    }
    Ok(Trajectory::from_vec(loc_vec, coord_system))
}

/// Convert a longitude/latitude sequence expressed as a Python `numpy.ndarray` into a `Vec<[f64; 2]>`
fn traj_np_to_vec(array: PyReadonlyArray2<'_, f64>) -> Result<Vec<[f64; 2]>, TrajDLError> {
    let shape = array.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(TrajDLError::DataConvertionError(format!(
            "shape of array must be (n, 2), instead of {:?}",
            shape
        )));
    }

    let array_view = array.as_array();
    // 尝试获取一个扁平化的slice。这只有在numpy数组的内存是C-contiguous时才有效。
    // Polars to_numpy() 默认就是C-contiguous，所以这个路径会被命中。
    if let Some(slice) = array_view.as_slice() {
        // slice 是一个 &[f64]，形如 [lon1, lat1, lon2, lat2, ...]
        // 我们可以用 chunks_exact 高效地把它转换成 Vec<[f64; 2]>
        // 这个操作非常快，几乎等同于一次 memcpy。
        let results = slice
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect::<Vec<[f64; 2]>>();
        Ok(results)
    } else {
        // 如果数组不是内存连续的（例如，通过高级索引创建的视图），
        // as_slice()会失败，我们回退到逐行拷贝，但也用更高效的方式。
        let mut results = Vec::<[f64; 2]>::with_capacity(shape[0]);
        for row in array_view.rows().into_iter() {
            // 直接索引，避免了 collect 和 try_into 的开销
            results.push([row[0], row[1]]);
        }
        Ok(results)
    }
}

impl_py_methods_for_container!(PyTrajectory, {
    #[new]
    fn new() -> Self {
        Self(Default::default())
    }

    /// Build from a Python `List[List[float]]`
    ///
    /// Parameters
    /// ----------
    /// seq: List[List[float]]
    ///     Trajectory data, each element is a list of length 2
    ///
    /// coord_system: CoordinateSystem
    ///     Coordinate system
    ///
    /// entity_id: Optional[str]
    ///     Trajectory id
    ///
    /// Returns
    /// ----------
    /// Sequence
    ///     Trajectory sequence
    #[classmethod]
    #[pyo3(signature = (seq, coord_system, entity_id=None))]
    pub fn from_list(
        _cls: &Bound<'_, PyType>,
        seq: &Bound<'_, PyList>,
        coord_system: CoordinateSystem,
        entity_id: Option<String>,
    ) -> PyResult<Self> {
        let mut r = pylist_to_trajectory(seq, coord_system)?;
        r.set_entity_id(entity_id);
        Ok(Self(r))
    }

    /// Build from a numpy.ndarray
    ///
    /// Parameters
    /// ----------
    /// seq : numpy.ndarray
    ///     Trajectory data, shape (N, 2), dtype must be float64
    /// coord_system : CoordinateSystem
    ///     Coordinate system (required, prevents coordinate-system information loss)
    /// entity_id : Optional[str]
    ///     Trajectory id
    #[classmethod]
    #[pyo3(signature = (seq, coord_system, entity_id=None))]
    fn from_numpy(
        _cls: &Bound<'_, PyType>,
        seq: &Bound<'_, PyUntypedArray>,
        coord_system: CoordinateSystem,
        entity_id: Option<String>,
    ) -> PyResult<Self> {
        let py = seq.py();

        // 检查维度是否为2
        if seq.ndim() != 2 {
            return Err(TrajDLError::InvalidParams(format!(
                "Expected a 2D numpy array, but got a {}D array.",
                seq.ndim()
            ))
            .into());
        }

        // 检查 dtype 是否为 f64
        if !seq.dtype().is_equiv_to(&dtype::<f64>(py)) {
            // 使用 dtype.name() 获取 dtype 的字符串表示
            return Err(TrajDLError::InvalidParams(format!(
                "Expected a numpy array with dtype=float64, but got dtype='{}'",
                seq.dtype()
            ))
            .into());
        }

        let r = Trajectory {
            meta: SequenceMeta {
                entity_id,
                start_ts: None,
                ts_seq: None,
            },
            seq: traj_np_to_vec(seq.extract()?)?,
            coord_system,
        };
        Ok(Self(r))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Trajectory(seq={:?})", self.0.seq))
    }

    fn __getitem__(&self, idx: i64) -> PyResult<[f64; 2]> {
        let len = self.0.seq.len() as i64;

        // 处理负数索引：-1 是最后一个，-len 是第一个，超出则报错
        let index = if idx < 0 { len + idx } else { idx };

        if index < 0 || index >= len {
            Err(PyIndexError::new_err("index out of range"))
        } else {
            Ok(self.0.seq[index as usize].clone())
        }
    }

    #[getter]
    fn get_seq(&self) -> PyResult<Vec<[f64; 2]>> {
        Ok(self.0.seq.clone())
    }

    /// Return the trajectory point sequence converted into a (numpy.ndarray, CoordinateSystem) tuple
    ///
    /// The return value carries the coordinate-system information to prevent coordinate-system information loss.
    fn to_numpy(&self, py: Python) -> PyResult<(Py<PyArray2<f64>>, CoordinateSystem)> {
        let seq = &self.0.seq;
        let num_points = seq.len();
        let x = seq.as_slice().as_flattened();
        let array = PyArray::from_slice(py, x);
        Ok((array.reshape([num_points, 2])?.into(), self.0.coord_system))
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

    /// Get the coordinate system of the trajectory
    #[getter]
    fn get_coord_system(&self) -> CoordinateSystem {
        self.0.coord_system
    }

    /// Convert to the Web Mercator coordinate system
    ///
    /// Returns a new Trajectory object with coordinates converted to Web Mercator
    fn to_web_mercator(&self) -> Self {
        PyTrajectory(self.0.to_web_mercator())
    }

    /// Convert to the WGS84 coordinate system
    ///
    /// Returns a new Trajectory object with coordinates converted to WGS84
    fn to_wgs84(&self) -> Self {
        PyTrajectory(self.0.to_wgs84())
    }
});
