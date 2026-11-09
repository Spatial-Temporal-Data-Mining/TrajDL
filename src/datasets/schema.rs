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

//! Canonical Arrow schemas for `Trajectory` / `LocSeq` datasets.
//!
//! These schemas are the single source of truth: the Python side
//! (`python/trajdl/datasets/arrow/schema.py`) imports them via the
//! PyO3 binding layer instead of redefining them, which guarantees
//! Rust and Python agree on field types and ordering.

use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Column name: per-row sequence payload.
///
/// - `Trajectory`: `LargeList<FixedSizeList<Float64, 2>>` (lon/lat pairs).
/// - `LocSeq`: `LargeList<LargeUtf8>` (nullable string locations).
pub const COL_SEQ: &str = "seq";

/// Column name: optional per-row entity identifier.
pub const COL_ENTITY_ID: &str = "entity_id";

/// Column name: optional per-row sequence start timestamp (epoch).
pub const COL_START_TS: &str = "start_ts";

/// Column name: optional per-row timestamp sequence aligned with `seq`.
pub const COL_TS_SEQ: &str = "ts_seq";

/// Inner field of the `seq` column for `Trajectory` rows.
///
/// Each list element is a 2-`Float64` fixed-size list `(x, y)`.
/// `nullable=true` here matches the Python definition (`pa.list_(pa.float64(), 2)`,
/// whose inner field defaults to nullable). We never actually emit nulls for
/// coordinate values, but the schema flag must agree byte-for-byte.
fn traj_inner_coord_field() -> Field {
    Field::new("item", DataType::Float64, true)
}

/// Inner field of the `seq` column for `LocSeq` rows.
fn loc_seq_inner_field() -> Field {
    Field::new("item", DataType::LargeUtf8, true)
}

/// Inner field of the `ts_seq` column shared by both schemas.
fn ts_seq_inner_field() -> Field {
    Field::new("item", DataType::Int64, true)
}

/// Build the canonical schema for `TrajectoryDataset`.
pub fn traj_schema() -> Schema {
    let coord_inner = traj_inner_coord_field();
    let traj_seq_type = DataType::LargeList(Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(Arc::new(coord_inner), 2),
        true,
    )));

    Schema::new(vec![
        Field::new(COL_SEQ, traj_seq_type, true),
        Field::new(COL_ENTITY_ID, DataType::LargeUtf8, true),
        Field::new(COL_START_TS, DataType::Int64, true),
        Field::new(
            COL_TS_SEQ,
            DataType::LargeList(Arc::new(ts_seq_inner_field())),
            true,
        ),
    ])
}

/// Build the canonical schema for `LocSeqDataset`.
pub fn loc_seq_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            COL_SEQ,
            DataType::LargeList(Arc::new(loc_seq_inner_field())),
            true,
        ),
        Field::new(COL_ENTITY_ID, DataType::LargeUtf8, true),
        Field::new(COL_START_TS, DataType::Int64, true),
        Field::new(
            COL_TS_SEQ,
            DataType::LargeList(Arc::new(ts_seq_inner_field())),
            true,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traj_schema_has_expected_fields() {
        let schema = traj_schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![COL_SEQ, COL_ENTITY_ID, COL_START_TS, COL_TS_SEQ]
        );

        // seq column: LargeList<FixedSizeList<Float64, 2>>
        match schema.field(0).data_type() {
            DataType::LargeList(child) => match child.data_type() {
                DataType::FixedSizeList(inner, 2) => {
                    assert_eq!(inner.data_type(), &DataType::Float64);
                }
                other => panic!("unexpected inner list type: {:?}", other),
            },
            other => panic!("unexpected seq type: {:?}", other),
        }

        assert_eq!(schema.field(1).data_type(), &DataType::LargeUtf8);
        assert_eq!(schema.field(2).data_type(), &DataType::Int64);
    }

    #[test]
    fn loc_seq_schema_has_expected_fields() {
        let schema = loc_seq_schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![COL_SEQ, COL_ENTITY_ID, COL_START_TS, COL_TS_SEQ]
        );

        // seq column: LargeList<LargeUtf8>
        match schema.field(0).data_type() {
            DataType::LargeList(child) => {
                assert_eq!(child.data_type(), &DataType::LargeUtf8);
            }
            other => panic!("unexpected seq type: {:?}", other),
        }

        // ts_seq column: LargeList<Int64>
        match schema.field(3).data_type() {
            DataType::LargeList(child) => {
                assert_eq!(child.data_type(), &DataType::Int64);
            }
            other => panic!("unexpected ts_seq type: {:?}", other),
        }
    }

    #[test]
    fn schemas_are_stable_across_calls() {
        // Ensure no hidden state: building twice yields equal schemas.
        assert_eq!(traj_schema(), traj_schema());
        assert_eq!(loc_seq_schema(), loc_seq_schema());
    }
}
