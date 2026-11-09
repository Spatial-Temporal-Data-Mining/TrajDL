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

use arrow::array::Int64Array;
use serde::{Deserialize, Serialize};

use crate::{err::TrajDLError, sequence::traits::IterMetaSeq};

use pyo3::pyclass;
use pyo3_stub_gen_derive::gen_stub_pyclass_enum;

/// The type of a sequence, either a location sequence or a trajectory point sequence
#[gen_stub_pyclass_enum]
#[pyclass(eq, module = "trajdl.trajdl_rust")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeqType {
    LocSeq,
    Trajectory,
}

/// Represents metadata for a time-series or trajectory sequence.
///
/// This struct holds auxiliary information about a sequence of data points,
/// such as its identifier, and start time. All fields are optional to allow
/// for flexibility in representing different kinds of sequence data.
///
/// # Examples
///
/// ```
/// # use serde::{Deserialize, Serialize};
/// # #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// # pub struct SequenceMeta {
/// #     pub entity_id: Option<String>,
/// #     pub start_ts: Option<i64>,
/// #     pub ts_seq: Option<Vec<i64>>,
/// # }
/// let meta = SequenceMeta {
///     entity_id: Some("vehicle-42".to_string()),
///     start_ts: Some(1672531200), // e.g., Jan 1, 2023 00:00:00 UTC
///     ts_seq: Some(vec![1672531200, 1672531205, 1672531215]),
/// };
///
/// assert_eq!(meta.entity_id.as_deref(), Some("vehicle-42"));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceMeta {
    /// The optional unique identifier for the entity this sequence belongs to (e.g., a vehicle ID).
    pub entity_id: Option<String>,

    /// The optional starting timestamp of the sequence, typically a Unix timestamp (in seconds or milliseconds).
    pub start_ts: Option<i64>,

    /// An optional sequence of absolute timestamps for each point in the sequence.
    pub ts_seq: Option<Vec<i64>>,
}

/// A zero-copy view of sequence metadata, typically from a row in an Arrow `RecordBatch`.
///
/// Unlike `SequenceMeta` which owns its data (e.g., using `Vec<f32>`), this struct
/// holds references or slices (`Int64Array`, `Float32Array`) of data residing in an
/// underlying Arrow memory buffer. This design avoids memory allocation and copying
/// when iterating over rows of a columnar dataset, making it highly efficient for
/// data processing and analytics tasks.
///
/// It is conceptually a "row-like" view over "column-like" data.
#[derive(Clone, Debug)]
pub struct SequenceMetaView {
    pub entity_id: Option<String>,
    pub start_ts: Option<i64>,
    pub ts_seq: Int64Array,
}

impl IterMetaSeq for SequenceMeta {
    fn iter_ts_seq(&self) -> Result<Option<impl Iterator<Item = i64>>, TrajDLError> {
        Ok(match &self.ts_seq {
            Some(x) => Some(x.iter().map(|x| x.clone())),
            None => None,
        })
    }
}

impl IterMetaSeq for SequenceMetaView {
    fn iter_ts_seq(&self) -> Result<Option<impl Iterator<Item = i64>>, TrajDLError> {
        let iter = (0..self.ts_seq.len()).map(|idx| self.ts_seq.value(idx));
        Ok(Some(iter))
    }
}

impl Default for SequenceMeta {
    fn default() -> Self {
        Self {
            entity_id: Default::default(),
            start_ts: Default::default(),
            ts_seq: Default::default(),
        }
    }
}
