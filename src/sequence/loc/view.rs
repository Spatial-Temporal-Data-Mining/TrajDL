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

use crate::impl_read_sequence_meta_trait;
use crate::sequence::loc::container::LocSeq;
use crate::sequence::traits::{LocSequence, ReadSequenceMeta};
use crate::sequence::utils::{
    extract_entity_id, extract_large_list_array, extract_start_ts, extract_ts_seq,
};
use crate::{
    err::TrajDLError,
    sequence::base::{SequenceMeta, SequenceMetaView},
};
use arrow::array::RecordBatch;
use arrow::array::{Array, LargeStringArray};

/// Extract the location sequence
fn extract_seq_for_locseq(
    batch: &RecordBatch,
    idx_in_batch: usize,
) -> Result<LargeStringArray, TrajDLError> {
    let col_name = "seq";
    let array = extract_large_list_array(batch, col_name)?;
    let first_elment = array.value(idx_in_batch);
    let arr = first_elment
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| {
            TrajDLError::DataConvertionError(format!(
                "cannot convert first element in column {} into StringArray",
                col_name
            ))
        })?;
    Ok(arr.clone())
}

// --- 视图类型 ---
#[derive(Clone, Debug)]
pub struct LocSeqView {
    pub meta: SequenceMetaView, // 组合元数据视图
    pub seq: LargeStringArray,  // 引用位置数据
}

impl LocSeqView {
    /// Build a view from a pyarrow.Table
    pub fn from_arrow_table(batches: &Vec<RecordBatch>, idx: usize) -> Result<Self, TrajDLError> {
        let mut current = 0;
        for batch in batches {
            let num_rows = batch.num_rows();
            if idx < current + num_rows {
                let row_in_batch = idx - current;

                let view = LocSeqView {
                    meta: SequenceMetaView {
                        entity_id: extract_entity_id(batch, row_in_batch)?,
                        start_ts: extract_start_ts(batch, row_in_batch)?,
                        ts_seq: extract_ts_seq(batch, row_in_batch)?,
                    },
                    seq: extract_seq_for_locseq(batch, row_in_batch)?,
                };

                return Ok(view);
            }
            current += num_rows;
        }
        return Err(TrajDLError::OutofIndex("out of index".to_string()));
    }

    pub fn iter_locs(&self) -> impl Iterator<Item = Option<String>> {
        self.seq.iter().map(|x| match x {
            None => None,
            Some(s) => Some(s.to_string()),
        })
    }

    pub fn get(&self, idx: usize) -> Option<&str> {
        if self.seq.is_null(idx) {
            None
        } else {
            Some(self.seq.value(idx))
        }
    }

    pub fn o(&self) -> Option<&str> {
        if self.seq.len() > 0 {
            self.get(0)
        } else {
            None
        }
    }

    pub fn d(&self) -> Option<&str> {
        if self.seq.len() > 0 {
            self.get(self.len() - 1)
        } else {
            None
        }
    }
}

impl_read_sequence_meta_trait!(LocSeqView);

impl LocSequence for LocSeqView {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn to_container(&self) -> Result<LocSeq, TrajDLError> {
        let entity_id = self.get_entity_id().map(String::from);
        let start_ts = self.get_start_ts();
        let ts_seq: Option<Vec<i64>> = self.iter_ts_seq()?.map(|x| x.collect());
        let seq: Vec<Option<String>> = self.iter_locs().collect();
        let result = LocSeq {
            meta: SequenceMeta {
                entity_id: entity_id,
                start_ts: start_ts,
                ts_seq,
            },
            seq: seq,
        };
        Ok(result)
    }
}
