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

use crate::err::TrajDLError;
use arrow::array::{Array, Int64Array, LargeListArray, LargeStringArray, RecordBatch};

pub fn extract_large_list_array<'a>(
    batch: &'a RecordBatch,
    col_name: &str,
) -> Result<&'a LargeListArray, TrajDLError> {
    let col = batch.column_by_name(col_name).ok_or_else(|| {
        TrajDLError::DataConvertionError(format!("cannot find column named {}", col_name))
    })?;

    Ok(col
        .as_any()
        .downcast_ref::<LargeListArray>()
        .ok_or_else(|| {
            TrajDLError::DataConvertionError(format!(
                "cannot convert column {} into LargeListArray",
                col_name
            ))
        })?)
}

/// Extract the sequence id
pub fn extract_entity_id(
    batch: &RecordBatch,
    idx_in_batch: usize,
) -> Result<Option<String>, TrajDLError> {
    let col_name = "entity_id";
    let col = batch.column_by_name(col_name).ok_or_else(|| {
        TrajDLError::DataConvertionError(format!("cannot find column named {}", col_name))
    })?;

    let array = col
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| {
            TrajDLError::DataConvertionError(format!(
                "cannot convert column {} into LargeListArray",
                col_name
            ))
        })?;

    let value = if array.is_null(idx_in_batch) {
        None
    } else {
        Some(array.value(idx_in_batch).to_string())
    };
    Ok(value)
}

/// Extract the per-point timestamps of the sequence
pub fn extract_ts_seq(batch: &RecordBatch, idx_in_batch: usize) -> Result<Int64Array, TrajDLError> {
    let col_name = "ts_seq";
    let array = extract_large_list_array(batch, col_name)?;
    let first_elment = array.value(idx_in_batch);
    Ok(first_elment
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            TrajDLError::DataConvertionError(format!(
                "cannot convert first element in column {} into Int64Array",
                col_name
            ))
        })?
        .clone())
}

/// Extract the start time of the sequence
pub fn extract_start_ts(
    batch: &RecordBatch,
    idx_in_batch: usize,
) -> Result<Option<i64>, TrajDLError> {
    let col_name = "start_ts";
    let col = batch.column_by_name(col_name).ok_or_else(|| {
        TrajDLError::DataConvertionError(format!("cannot find column named {}", col_name))
    })?;

    let array = col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        TrajDLError::DataConvertionError(format!(
            "cannot convert column {} into LargeListArray",
            col_name
        ))
    })?;

    let value = if array.is_null(idx_in_batch) {
        None
    } else {
        Some(array.value(idx_in_batch))
    };
    Ok(value)
}
