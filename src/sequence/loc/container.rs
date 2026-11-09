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
    impl_read_sequence_meta_trait, impl_write_sequence_meta_trait,
    sequence::{base::SequenceMeta, traits::LocSequence},
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LocSeq {
    pub meta: SequenceMeta,
    pub seq: Vec<Option<String>>,
}

impl LocSeq {
    pub fn from_vec(vec: Vec<Option<String>>) -> Self {
        LocSeq {
            meta: SequenceMeta::default(),
            seq: vec,
        }
    }

    pub fn iter_locs(&self) -> impl Iterator<Item = Option<String>> {
        self.seq.iter().cloned()
    }
}

impl_read_sequence_meta_trait!(LocSeq);
impl_write_sequence_meta_trait!(LocSeq);

impl LocSequence for LocSeq {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn to_container(&self) -> Result<LocSeq, TrajDLError> {
        Ok(self.clone())
    }
}

impl Default for LocSeq {
    fn default() -> Self {
        Self {
            meta: Default::default(),
            seq: Default::default(),
        }
    }
}
