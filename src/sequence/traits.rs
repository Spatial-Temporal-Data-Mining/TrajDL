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
    err::TrajDLError,
    sequence::{loc::container::LocSeq, point::container::Trajectory},
    spatial::coord::{Coordinate, CoordinateSystem},
};

/// Trait for reading meta information
pub trait ReadSequenceMeta {
    fn get_entity_id(&self) -> Option<&str>;
    fn get_start_ts(&self) -> Option<i64>;
    fn get_ts_seq(&self) -> Result<Option<Vec<i64>>, TrajDLError>;
    fn iter_ts_seq(&self) -> Result<Option<impl Iterator<Item = i64>>, TrajDLError>;
}

/// Trait for writing meta information
pub trait WriteSequenceMeta {
    fn set_entity_id(&mut self, entity_id: Option<String>);
    fn set_start_ts(&mut self, start_ts: Option<i64>);
    fn set_ts_seq(&mut self, ts_seq: Option<Vec<i64>>);
}

/// This trait is used to iterate over the sequence values in the meta information
pub trait IterMetaSeq {
    /// Iterate over ts_seq
    fn iter_ts_seq(&self) -> Result<Option<impl Iterator<Item = i64>>, TrajDLError>;
}

pub trait LocSequence {
    /// Sequence length
    fn len(&self) -> usize;

    /// Convert to the container type
    fn to_container(&self) -> Result<LocSeq, TrajDLError>;
}

pub trait PointSequence {
    /// Sequence length
    fn len(&self) -> usize;

    /// Convert to the container type
    fn to_container(&self) -> Result<Trajectory, TrajDLError>;

    /// Get the idx-th point
    fn get(&self, idx: usize) -> [f64; 2];

    /// Convert the longitude/latitude sequence into a vec
    fn to_vec(&self) -> Vec<[f64; 2]>;

    /// Get an iterator over the longitude/latitude coordinates
    fn iter_coords(&self) -> Box<dyn Iterator<Item = Coordinate> + '_>;

    fn get_locs<F>(&self, mut converter: F) -> Vec<(String, Coordinate)>
    where
        F: FnMut(Coordinate) -> String,
    {
        self.iter_coords()
            .map(|coord| (converter(coord), coord))
            .collect()
    }

    fn coord_system(&self) -> CoordinateSystem;
}
