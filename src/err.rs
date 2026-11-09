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

use pyo3::PyErr;
use pyo3::exceptions::*;

use crate::spatial::coord::CoordinateSystem;

#[derive(Debug, thiserror::Error)]
pub enum TrajDLError {
    #[error("ListArray<i64>的长度必须为1")]
    InvalidSizeOfListArray,

    #[error("InvalidParams")]
    InvalidParams(String),

    #[error("坐标必须是长度为2的数组，但收到了 {0} 个元素")]
    InvalidCoordinate(usize),

    #[error("坐标系错误，需要{0}，但是收到了{1}")]
    InvalidCoordinateSystem(CoordinateSystem, CoordinateSystem),

    #[error("DataConvertionError: {0}")]
    DataConvertionError(String),

    #[error("SeqType错误")]
    InvalidSeqType,

    #[error("converter异常")]
    InvalidConverter,

    #[error("越界")]
    OutofIndex(String),
}

impl From<TrajDLError> for PyErr {
    fn from(err: TrajDLError) -> PyErr {
        match err {
            TrajDLError::InvalidCoordinate(n) => {
                PyValueError::new_err(format!("坐标必须包含两个值（经度、纬度），但收到 {} 个", n))
            }
            TrajDLError::InvalidCoordinateSystem(need, got) => {
                PyValueError::new_err(format!("坐标系错误，需要{}，但是收到了{}", need, got))
            }
            TrajDLError::InvalidParams(s) => PyValueError::new_err(format!("InvalidParams: {}", s)),
            TrajDLError::DataConvertionError(s) => {
                PyValueError::new_err(format!("DataConvertionError: {}", s))
            }
            TrajDLError::InvalidSizeOfListArray => {
                PyValueError::new_err("ListArray<i64>的长度必须为1")
            }
            TrajDLError::InvalidSeqType => PyValueError::new_err("SeqType错误"),
            TrajDLError::InvalidConverter => PyValueError::new_err("converter异常"),
            TrajDLError::OutofIndex(e) => PyValueError::new_err(format!("out of index: {}", e)),
        }
    }
}
