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

//! High-throughput `Vec<Trajectory|LocSeq> -> RecordBatch` builders.
//!
//! These functions are the Rust-side fast path used by Python's
//! `init_from_seqs` to bypass the per-element Python loop that would
//! otherwise dominate dataset construction time.
//!
//! Both builders preserve the exact column ordering and semantics of
//! the schemas defined in [`crate::datasets::schema`], so the produced
//! `RecordBatch` is byte-equivalent to what the legacy Python path
//! produces.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Float64Builder, Int64Builder, LargeListBuilder,
    LargeStringBuilder, RecordBatch,
};
use arrow::error::ArrowError;

use crate::datasets::schema::{loc_seq_schema, traj_schema};
use crate::sequence::loc::container::LocSeq;
use crate::sequence::point::container::Trajectory;
use crate::sequence::traits::ReadSequenceMeta;

/// Append the four shared scalar / list columns
/// (`entity_id`, `start_ts`, `ts_seq`) to their builders.
///
/// The `seq` column is schema-specific and handled by the caller.
fn append_meta_columns<S: ReadSequenceMeta>(
    seq: &S,
    entity_id_builder: &mut LargeStringBuilder,
    start_ts_builder: &mut Int64Builder,
    ts_seq_builder: &mut LargeListBuilder<Int64Builder>,
) -> Result<(), ArrowError> {
    // entity_id
    match seq.get_entity_id() {
        Some(id) => entity_id_builder.append_value(id),
        None => entity_id_builder.append_null(),
    }

    // start_ts
    match seq.get_start_ts() {
        Some(ts) => start_ts_builder.append_value(ts),
        None => start_ts_builder.append_null(),
    }

    // ts_seq
    match seq
        .get_ts_seq()
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?
    {
        Some(ts_vec) => {
            for ts in ts_vec.iter() {
                ts_seq_builder.values().append_value(*ts);
            }
            ts_seq_builder.append(true);
        }
        None => {
            ts_seq_builder.append(false);
        }
    }

    Ok(())
}

/// Build a `RecordBatch` matching [`traj_schema`] from a slice of
/// `Trajectory` containers.
///
/// Each row's `seq` column is encoded as a `LargeList<FixedSizeList<f64, 2>>`
/// where the inner fixed-size list pairs are the trajectory's `(x, y)`
/// coordinates. Coordinate-system semantics are not encoded in the schema
/// (callers must keep that invariant separately).
pub fn build_traj_record_batch(trajs: &[Trajectory]) -> Result<RecordBatch, ArrowError> {
    let schema = Arc::new(traj_schema());
    let n_rows = trajs.len();

    // seq builder: LargeList<FixedSizeList<f64, 2>>
    // FixedSizeListBuilder needs the per-element capacity hint via `with_capacity`,
    // which is the number of fixed-size lists (= number of points). Items per
    // list is fixed at 2 by the schema.
    let total_points: usize = trajs.iter().map(|t| t.seq.len()).sum();
    let coord_values_builder = Float64Builder::with_capacity(total_points * 2);
    // The inner Float64 field is `nullable=true` to match the Python schema
    // (`pa.list_(pa.float64(), 2)` defaults to a nullable inner field). The
    // builder must produce arrays whose data type matches the schema returned
    // by `traj_schema()` byte-for-byte, otherwise `RecordBatch::try_new`
    // rejects the column.
    let fixed_size_list_builder = FixedSizeListBuilder::with_capacity(
        coord_values_builder,
        2, // (x, y)
        total_points,
    )
    .with_field(Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float64,
        true,
    )));
    let mut seq_builder: LargeListBuilder<FixedSizeListBuilder<Float64Builder>> =
        LargeListBuilder::with_capacity(fixed_size_list_builder, n_rows).with_field(Arc::new(
            arrow::datatypes::Field::new(
                "item",
                arrow::datatypes::DataType::FixedSizeList(
                    Arc::new(arrow::datatypes::Field::new(
                        "item",
                        arrow::datatypes::DataType::Float64,
                        true,
                    )),
                    2,
                ),
                true,
            ),
        ));

    let mut entity_id_builder = LargeStringBuilder::with_capacity(n_rows, n_rows * 8);
    let mut start_ts_builder = Int64Builder::with_capacity(n_rows);
    let ts_inner = Int64Builder::with_capacity(n_rows);
    let mut ts_seq_builder: LargeListBuilder<Int64Builder> =
        LargeListBuilder::with_capacity(ts_inner, n_rows);

    for traj in trajs {
        // seq: append all (x, y) pairs as fixed-size lists, then close the row.
        for coord in &traj.seq {
            let inner = seq_builder.values();
            inner.values().append_value(coord[0]);
            inner.values().append_value(coord[1]);
            inner.append(true);
        }
        seq_builder.append(true);

        append_meta_columns(
            traj,
            &mut entity_id_builder,
            &mut start_ts_builder,
            &mut ts_seq_builder,
        )?;
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(seq_builder.finish()),
        Arc::new(entity_id_builder.finish()),
        Arc::new(start_ts_builder.finish()),
        Arc::new(ts_seq_builder.finish()),
    ];

    RecordBatch::try_new(schema, columns)
}

/// Build a `RecordBatch` matching [`loc_seq_schema`] from a slice of
/// `LocSeq` containers.
///
/// Each row's `seq` column is encoded as a `LargeList<LargeUtf8>` where
/// individual list entries can be null (matching `Vec<Option<String>>`).
pub fn build_loc_seq_record_batch(seqs: &[LocSeq]) -> Result<RecordBatch, ArrowError> {
    let schema = Arc::new(loc_seq_schema());
    let n_rows = seqs.len();

    let total_locs: usize = seqs.iter().map(|s| s.seq.len()).sum();
    let loc_inner = LargeStringBuilder::with_capacity(total_locs, total_locs * 8);
    let mut seq_builder: LargeListBuilder<LargeStringBuilder> =
        LargeListBuilder::with_capacity(loc_inner, n_rows);

    let mut entity_id_builder = LargeStringBuilder::with_capacity(n_rows, n_rows * 8);
    let mut start_ts_builder = Int64Builder::with_capacity(n_rows);
    let ts_inner = Int64Builder::with_capacity(n_rows);
    let mut ts_seq_builder: LargeListBuilder<Int64Builder> =
        LargeListBuilder::with_capacity(ts_inner, n_rows);

    for seq in seqs {
        for loc in &seq.seq {
            match loc {
                Some(s) => seq_builder.values().append_value(s),
                None => seq_builder.values().append_null(),
            }
        }
        seq_builder.append(true);

        append_meta_columns(
            seq,
            &mut entity_id_builder,
            &mut start_ts_builder,
            &mut ts_seq_builder,
        )?;
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(seq_builder.finish()),
        Arc::new(entity_id_builder.finish()),
        Arc::new(start_ts_builder.finish()),
        Arc::new(ts_seq_builder.finish()),
    ];

    RecordBatch::try_new(schema, columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::base::SequenceMeta;
    use crate::spatial::coord::CoordinateSystem;
    // `len`, `is_null`, etc. live on the `Array` trait; bring it into scope.
    use arrow::array::Array;

    fn make_traj(
        coords: Vec<[f64; 2]>,
        entity_id: Option<&str>,
        start_ts: Option<i64>,
        ts_seq: Option<Vec<i64>>,
    ) -> Trajectory {
        Trajectory {
            meta: SequenceMeta {
                entity_id: entity_id.map(String::from),
                start_ts,
                ts_seq,
            },
            seq: coords,
            coord_system: CoordinateSystem::Wgs84,
        }
    }

    fn make_loc_seq(
        locs: Vec<Option<String>>,
        entity_id: Option<&str>,
        start_ts: Option<i64>,
        ts_seq: Option<Vec<i64>>,
    ) -> LocSeq {
        LocSeq {
            meta: SequenceMeta {
                entity_id: entity_id.map(String::from),
                start_ts,
                ts_seq,
            },
            seq: locs,
        }
    }

    #[test]
    fn build_traj_batch_shapes_and_values() {
        let trajs = vec![
            make_traj(
                vec![[1.0, 2.0], [3.0, 4.0]],
                Some("user_a"),
                Some(100),
                Some(vec![100, 101]),
            ),
            make_traj(vec![[5.0, 6.0]], None, None, None),
        ];

        let batch = build_traj_record_batch(&trajs).expect("build ok");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().as_ref(), &traj_schema());

        // entity_id: "user_a", null
        let entity_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
            .expect("entity_id is LargeStringArray");
        assert_eq!(entity_col.value(0), "user_a");
        assert!(entity_col.is_null(1));

        // start_ts: 100, null
        let start_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("start_ts is Int64Array");
        assert_eq!(start_col.value(0), 100);
        assert!(start_col.is_null(1));

        // ts_seq: [100, 101], null list
        let ts_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .expect("ts_seq is LargeListArray");
        let row0 = ts_col.value(0);
        let row0_ints = row0
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("inner Int64Array");
        assert_eq!(row0_ints.values(), &[100, 101]);
        assert!(ts_col.is_null(1));

        // seq: row0 has 2 coordinate pairs, row1 has 1
        let seq_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .expect("seq is LargeListArray");
        let row0 = seq_col.value(0);
        let fixed = row0
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
            .expect("inner FixedSizeListArray");
        assert_eq!(fixed.len(), 2);
        let coords = fixed
            .values()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("Float64Array");
        assert_eq!(coords.values(), &[1.0, 2.0, 3.0, 4.0]);

        let row1 = seq_col.value(1);
        let fixed1 = row1
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
            .expect("inner FixedSizeListArray");
        assert_eq!(fixed1.len(), 1);
    }

    #[test]
    fn build_loc_seq_batch_handles_null_locs() {
        let seqs = vec![
            make_loc_seq(
                vec![Some("a".into()), None, Some("c".into())],
                Some("u1"),
                Some(7),
                Some(vec![10, 20, 30]),
            ),
            make_loc_seq(vec![], None, None, None),
        ];

        let batch = build_loc_seq_record_batch(&seqs).expect("build ok");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().as_ref(), &loc_seq_schema());

        let seq_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .expect("seq is LargeListArray");

        let row0 = seq_col.value(0);
        let strs = row0
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
            .expect("LargeStringArray");
        assert_eq!(strs.len(), 3);
        assert_eq!(strs.value(0), "a");
        assert!(strs.is_null(1));
        assert_eq!(strs.value(2), "c");

        // empty row: list of length 0, not null
        let row1 = seq_col.value(1);
        assert_eq!(row1.len(), 0);
        assert!(!seq_col.is_null(1));
    }

    #[test]
    fn build_traj_batch_empty_input() {
        let trajs: Vec<Trajectory> = vec![];
        let batch = build_traj_record_batch(&trajs).expect("build ok");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 4);
    }

    #[test]
    fn build_loc_seq_batch_empty_input() {
        let seqs: Vec<LocSeq> = vec![];
        let batch = build_loc_seq_record_batch(&seqs).expect("build ok");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 4);
    }
}
