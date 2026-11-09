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

use arrow::array::Float64Array;
use pyo3::{
    Bound, FromPyObject, PyAny,
    types::{PyAnyMethods, PyList, PyListMethods},
};

use crate::{
    binding::seq::{
        container::{PyLocSeq, PyTrajectory, pylist_to_locseq, pylist_to_trajectory},
        view::{PyLocSeqView, PyTrajectoryView},
    },
    err::TrajDLError,
    sequence::{
        loc::container::LocSeq,
        point::container::Trajectory,
        traits::{LocSequence, PointSequence},
    },
    spatial::coord::{Coordinate, CoordinateSystem},
};

#[derive(FromPyObject)]
pub enum PyLocSeqType<'a> {
    /// Container type
    Container(PyLocSeq),

    /// View type
    View(PyLocSeqView),

    /// Python List[str]
    Py(Bound<'a, PyList>),
}

#[derive(Clone, Debug, FromPyObject)]
pub enum PyTrajectoryType<'a> {
    /// Container type
    Container(PyTrajectory),

    /// View type
    View(PyTrajectoryView),

    /// Python List[List[float]]
    Py(Bound<'a, PyList>),
}

/// Build a PyLocSeqType from a PyAny
impl<'a> TryFrom<&Bound<'a, PyAny>> for PyLocSeqType<'a> {
    type Error = TrajDLError;

    fn try_from(seq: &Bound<'a, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(seq) = seq.extract::<PyLocSeq>() {
            Ok(Self::Container(seq))
        } else if let Ok(seq) = seq.extract::<PyLocSeqView>() {
            Ok(Self::View(seq))
        } else if let Ok(loc_list) = seq.downcast::<PyList>() {
            Ok(Self::Py(loc_list.clone()))
        } else {
            Err(TrajDLError::DataConvertionError(
                "cannot convert seq into PyLocSeqType".to_string(),
            ))
        }
    }
}

impl<'a> PyLocSeqType<'a> {
    /// Iterate over the locations in the location sequence
    pub fn iter_locs(&self) -> Box<dyn Iterator<Item = Result<Option<String>, TrajDLError>> + '_> {
        match self {
            PyLocSeqType::Container(seq) => Box::new(seq.0.iter_locs().map(|loc| Ok(loc))),
            PyLocSeqType::View(seq) => Box::new(seq.iter_locs().map(|loc| Ok(loc))),
            PyLocSeqType::Py(seq) => {
                let iter = seq.iter().map(|loc| {
                    if loc.is_none() {
                        Ok(None)
                    } else {
                        let x = loc.extract::<&str>();
                        match x {
                            Ok(t) => Ok(Some(t.to_string())),
                            Err(_) => Err(TrajDLError::DataConvertionError("x".to_string())),
                        }
                    }
                });
                Box::new(iter)
            }
        }
    }
}

impl<'a> LocSequence for PyLocSeqType<'a> {
    fn len(&self) -> usize {
        match self {
            PyLocSeqType::Container(seq) => seq.0.len(),
            PyLocSeqType::View(seq) => seq.view.len(),
            PyLocSeqType::Py(seq) => seq.len(),
        }
    }

    fn to_container(&self) -> Result<LocSeq, TrajDLError> {
        match self {
            PyLocSeqType::Container(py_loc_seq) => py_loc_seq.0.to_container(),
            PyLocSeqType::View(py_loc_seq_view) => py_loc_seq_view.view.to_container(),
            PyLocSeqType::Py(seq) => {
                let locseq = pylist_to_locseq(seq).map_err(|e| {
                    TrajDLError::DataConvertionError(format!(
                        "cannot convert pylist into LocSeq with PyErr {:?}",
                        e
                    ))
                })?;
                Ok(locseq)
            }
        }
    }
}

/// Build a PyTrajectoryType from a PyAny
impl<'a> TryFrom<&Bound<'a, PyAny>> for PyTrajectoryType<'a> {
    type Error = TrajDLError;

    fn try_from(seq: &Bound<'a, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(seq) = seq.extract::<PyTrajectory>() {
            Ok(Self::Container(seq))
        } else if let Ok(seq) = seq.extract::<PyTrajectoryView>() {
            Ok(Self::View(seq))
        } else if let Ok(loc_list) = seq.downcast::<PyList>() {
            Ok(Self::Py(loc_list.clone()))
        } else {
            Err(TrajDLError::DataConvertionError(
                "cannot convert seq into PyTrajectoryType".to_string(),
            ))
        }
    }
}

impl<'a> PointSequence for PyTrajectoryType<'a> {
    fn len(&self) -> usize {
        match self {
            PyTrajectoryType::Container(seq) => seq.0.len(),
            PyTrajectoryType::View(seq) => seq.view.len(),
            PyTrajectoryType::Py(seq) => seq.len(),
        }
    }

    fn get(&self, idx: usize) -> [f64; 2] {
        match self {
            PyTrajectoryType::Container(seq) => seq.0.seq[idx],
            PyTrajectoryType::View(seq) => {
                let v = seq.view.seq.value(idx);
                let arr = v
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("cannot cast value in pyarrow array into Float64Array");
                [arr.value(0), arr.value(1)]
            }
            PyTrajectoryType::Py(seq) => {
                let v = seq
                    .get_item(idx)
                    .expect("cannot get element in python object");
                v.extract::<[f64; 2]>()
                    .expect("cannot cast element in python object into [f64; 2]")
            }
        }
    }

    fn to_vec(&self) -> Vec<[f64; 2]> {
        match self {
            PyTrajectoryType::Container(seq) => seq.0.to_vec(),
            PyTrajectoryType::View(seq) => seq.view.to_vec(),
            PyTrajectoryType::Py(seq) => seq
                .iter()
                .map(|pair| {
                    pair.extract::<[f64; 2]>()
                        .expect("cannot cast PyList into Iterator<[f64; 2]>")
                })
                .collect(),
        }
    }

    fn iter_coords(&self) -> Box<dyn Iterator<Item = Coordinate> + '_> {
        match self {
            PyTrajectoryType::Container(seq) => Box::new(seq.0.iter_coords()),
            PyTrajectoryType::View(seq) => Box::new(seq.view.iter_coords()),
            PyTrajectoryType::Py(seq) => {
                let iter = seq.iter().map(|pair| {
                    let x = pair
                        .extract::<[f64; 2]>()
                        .expect("cannot cast PyList into Iterator<[f64; 2]>");
                    match self.coord_system() {
                        CoordinateSystem::Wgs84 => Coordinate::Wgs84 {
                            lng: x[0],
                            lat: x[1],
                        },
                        CoordinateSystem::WebMercator => {
                            Coordinate::WebMercator { x: x[0], y: x[1] }
                        }
                    }
                });
                Box::new(iter)
            }
        }
    }

    fn to_container(&self) -> Result<Trajectory, TrajDLError> {
        match self {
            PyTrajectoryType::Container(seq) => Ok(seq.0.clone()),
            PyTrajectoryType::View(seq) => seq.view.to_container(),
            PyTrajectoryType::Py(seq) => {
                let traj = pylist_to_trajectory(seq, self.coord_system()).map_err(|e| {
                    TrajDLError::DataConvertionError(format!(
                        "cannot convert pylist into Trajectory with PyErr {:?}",
                        e
                    ))
                })?;
                Ok(traj)
            }
        }
    }

    fn coord_system(&self) -> CoordinateSystem {
        match self {
            PyTrajectoryType::Container(seq) => seq.0.coord_system,
            PyTrajectoryType::View(seq) => seq.view.coord_system,
            PyTrajectoryType::Py(_) => CoordinateSystem::Wgs84,
        }
    }
}
