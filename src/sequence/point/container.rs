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
    sequence::{base::SequenceMeta, traits::PointSequence},
    spatial::coord::{Coordinate, CoordinateSystem},
};
use traj_dist_rs::traits::AsCoord;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Trajectory {
    pub meta: SequenceMeta,
    // TODO: 这里使用numpy的性能更好，因为polars的Series可以通过零拷贝转换为numpy.ndarray。
    pub seq: Vec<[f64; 2]>,
    /// Coordinate system, defaults to WGS84
    #[serde(default)]
    pub coord_system: CoordinateSystem,
}

impl Default for Trajectory {
    fn default() -> Self {
        Self {
            meta: Default::default(),
            seq: Default::default(),
            coord_system: Default::default(),
        }
    }
}

impl Trajectory {
    pub fn from_vec(vec: Vec<[f64; 2]>, coord_system: CoordinateSystem) -> Self {
        Trajectory {
            meta: SequenceMeta {
                entity_id: None,
                start_ts: None,
                ts_seq: None,
            },
            seq: vec,
            coord_system,
        }
    }

    pub fn from_iter<I: Iterator<Item = T>, T: AsCoord>(
        coords: I,
        coord_system: CoordinateSystem,
    ) -> Self {
        let loc_vec: Vec<[f64; 2]> = coords.map(|c| [c.x(), c.y()]).collect();
        Self::from_vec(loc_vec, coord_system)
    }

    /// Convert to the Web Mercator coordinate system (returns a new Trajectory)
    pub fn to_web_mercator(&self) -> Self {
        match self.coord_system {
            CoordinateSystem::WebMercator => self.clone(),
            CoordinateSystem::Wgs84 => {
                let web_coords: Vec<[f64; 2]> = self
                    .seq
                    .iter()
                    .map(|c| {
                        Coordinate::Wgs84 {
                            lng: c[0],
                            lat: c[1],
                        }
                        .to_web_mercator()
                        .to_pair()
                    })
                    .collect();
                Self {
                    meta: self.meta.clone(),
                    seq: web_coords,
                    coord_system: CoordinateSystem::WebMercator,
                }
            }
        }
    }

    /// Convert to the WGS84 coordinate system (returns a new Trajectory)
    pub fn to_wgs84(&self) -> Self {
        match self.coord_system {
            CoordinateSystem::Wgs84 => self.clone(),
            CoordinateSystem::WebMercator => {
                let wgs_coords: Vec<[f64; 2]> = self
                    .seq
                    .iter()
                    .map(|c| {
                        Coordinate::WebMercator { x: c[0], y: c[1] }
                            .to_wgs84()
                            .to_pair()
                    })
                    .collect();
                Self {
                    meta: self.meta.clone(),
                    seq: wgs_coords,
                    coord_system: CoordinateSystem::Wgs84,
                }
            }
        }
    }
}

impl_read_sequence_meta_trait!(Trajectory);
impl_write_sequence_meta_trait!(Trajectory);

impl PointSequence for Trajectory {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn get(&self, idx: usize) -> [f64; 2] {
        self.seq[idx]
    }

    fn to_vec(&self) -> Vec<[f64; 2]> {
        self.seq.clone()
    }

    fn iter_coords(&self) -> Box<dyn Iterator<Item = Coordinate> + '_> {
        match self.coord_system {
            CoordinateSystem::Wgs84 => Box::new(self.seq.iter().map(|c| Coordinate::Wgs84 {
                lng: c[0],
                lat: c[1],
            })),
            CoordinateSystem::WebMercator => Box::new(
                self.seq
                    .iter()
                    .map(|c| Coordinate::WebMercator { x: c[0], y: c[1] }),
            ),
        }
    }

    fn to_container(&self) -> Result<Trajectory, TrajDLError> {
        Ok(self.clone())
    }

    /// Return the current coordinate system
    fn coord_system(&self) -> CoordinateSystem {
        self.coord_system
    }
}
