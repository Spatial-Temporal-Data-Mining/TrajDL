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

use crate::sequence::point::container::Trajectory;
use crate::sequence::traits::{ReadSequenceMeta, WriteSequenceMeta};
use crate::sequence::utils::{
    extract_entity_id, extract_large_list_array, extract_start_ts, extract_ts_seq,
};
use crate::spatial::coord::{Coordinate, CoordinateSystem, wgs84};
use crate::{
    err::TrajDLError,
    impl_read_sequence_meta_trait,
    sequence::{
        base::{SequenceMeta, SequenceMetaView},
        traits::PointSequence,
    },
};
use arrow::array::{Array, FixedSizeListArray};
use arrow::array::{Float64Array, RecordBatch};

/// Extract the location sequence
fn extract_seq_for_trajectory(
    batch: &RecordBatch,
    idx_in_batch: usize,
) -> Result<FixedSizeListArray, TrajDLError> {
    let col_name = "seq";
    let array = extract_large_list_array(batch, col_name)?;
    let first_elment = array.value(idx_in_batch);
    let arr = first_elment
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| {
            TrajDLError::DataConvertionError(format!(
                "cannot convert first element in column {} into FixedSizeListArray",
                col_name
            ))
        })?;
    Ok(arr.clone())
}

#[derive(Clone, Debug)]
pub struct TrajectoryView {
    pub meta: SequenceMetaView,         // 组合元数据视图
    pub seq: FixedSizeListArray,        // 引用点数据
    pub coord_system: CoordinateSystem, // 坐标系，默认为 WGS84
}

impl TrajectoryView {
    /// Build a view from a pyarrow.Table
    pub fn from_arrow_table(batches: &Vec<RecordBatch>, idx: usize) -> Result<Self, TrajDLError> {
        Self::from_arrow_table_with_coord(batches, idx, CoordinateSystem::Wgs84)
    }

    /// Build a view from a pyarrow.Table, specifying the coordinate system
    pub fn from_arrow_table_with_coord(
        batches: &Vec<RecordBatch>,
        idx: usize,
        coord_system: CoordinateSystem,
    ) -> Result<Self, TrajDLError> {
        let mut current = 0;
        for batch in batches {
            let num_rows = batch.num_rows();
            if idx < current + num_rows {
                let row_in_batch = idx - current;

                let view = TrajectoryView {
                    meta: SequenceMetaView {
                        entity_id: extract_entity_id(batch, row_in_batch)?,
                        start_ts: extract_start_ts(batch, row_in_batch)?,
                        ts_seq: extract_ts_seq(batch, row_in_batch)?,
                    },
                    seq: extract_seq_for_trajectory(batch, row_in_batch)?,
                    coord_system,
                };

                return Ok(view);
            }
            current += num_rows;
        }
        Err(TrajDLError::OutofIndex("out of index".to_string()))
    }

    pub fn iter_seq(&self) -> impl Iterator<Item = [f64; 2]> {
        let arr = &self.seq;
        (0..arr.len()).map(|idx| {
            let point = arr.value(idx);
            let point_arr = point
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("cannot convert point in seq column into Float64Array");
            [point_arr.value(0), point_arr.value(1)]
        })
    }

    /// Convert the current trajectory data into a Trajectory container in the Web Mercator coordinate system
    pub fn to_web_mercator(&self) -> Result<Trajectory, TrajDLError> {
        match self.coord_system {
            CoordinateSystem::WebMercator => self.to_container(),
            CoordinateSystem::Wgs84 => {
                let v = self.to_vec();
                let web_coords = v.iter().map(|c| wgs84(c[0], c[1]).to_web_mercator());
                let mut traj = Trajectory::from_iter(web_coords, CoordinateSystem::WebMercator);
                traj.set_entity_id(self.get_entity_id().map(|s| s.to_string()));
                traj.set_start_ts(self.get_start_ts());
                traj.set_ts_seq(self.get_ts_seq()?);
                Ok(traj)
            }
        }
    }

    /// Convert the current trajectory data into a Trajectory container in the WGS84 coordinate system
    pub fn to_wgs84(&self) -> Result<Trajectory, TrajDLError> {
        match self.coord_system {
            CoordinateSystem::Wgs84 => self.to_container(),
            CoordinateSystem::WebMercator => {
                let v = self.to_vec();
                let wgs_coords = v
                    .iter()
                    .map(|c| Coordinate::WebMercator { x: c[0], y: c[1] }.to_wgs84());
                let mut traj = Trajectory::from_iter(wgs_coords, CoordinateSystem::Wgs84);
                traj.set_entity_id(self.get_entity_id().map(|s| s.to_string()));
                traj.set_start_ts(self.get_start_ts());
                traj.set_ts_seq(self.get_ts_seq()?);
                Ok(traj)
            }
        }
    }
}

impl_read_sequence_meta_trait!(TrajectoryView);

impl PointSequence for TrajectoryView {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn get(&self, idx: usize) -> [f64; 2] {
        let point = self.seq.value(idx);
        let point_arr = point
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("cannot convert point in seq column into Float64Array");
        [point_arr.value(0), point_arr.value(1)]
    }

    /// Convert the sequence into a Vec<[f64; 2]> with high performance
    fn to_vec(&self) -> Vec<[f64; 2]> {
        let values = self
            .seq
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Expected Float64Array");
        let slice_f64: &[f64] = values.values();

        // 使用迭代器和 chunks_exact 进行转换
        self.seq
            .iter()
            .enumerate() // 获取索引以便检查 null
            .filter_map(|(i, _)| {
                if self.seq.is_valid(i) {
                    // 如果当前 list item 不为 null，则从底层扁平数组中获取数据块
                    let start = i * self.seq.value_length() as usize;
                    let end = start + self.seq.value_length() as usize;
                    let chunk = &slice_f64[start..end];
                    // try_into() 可以安全地将 &[f64] (如果长度正确) 转换为 &[f64; 2]
                    // 然后我们解引用得到 [f64; 2]
                    chunk.try_into().ok()
                } else {
                    // 如果为 null，则跳过
                    None
                }
            })
            .collect()
    }

    fn iter_coords(&self) -> Box<dyn Iterator<Item = Coordinate> + '_> {
        match self.coord_system {
            CoordinateSystem::Wgs84 => Box::new(self.iter_seq().map(|x| Coordinate::Wgs84 {
                lng: x[0],
                lat: x[1],
            })),
            CoordinateSystem::WebMercator => Box::new(
                self.iter_seq()
                    .map(|x| Coordinate::WebMercator { x: x[0], y: x[1] }),
            ),
        }
    }

    fn to_container(&self) -> Result<Trajectory, TrajDLError> {
        let entity_id = self.get_entity_id().map(String::from);
        let start_ts = self.get_start_ts();
        let ts_seq: Option<Vec<i64>> = self.iter_ts_seq()?.map(|x| x.collect());
        let seq: Vec<[f64; 2]> = self.to_vec();
        let result = Trajectory {
            meta: SequenceMeta {
                entity_id: entity_id,
                start_ts: start_ts,
                ts_seq,
            },
            seq: seq,
            coord_system: self.coord_system,
        };
        Ok(result)
    }

    /// Return the current coordinate system
    fn coord_system(&self) -> CoordinateSystem {
        self.coord_system
    }
}
