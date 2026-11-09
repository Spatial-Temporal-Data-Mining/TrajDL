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

use serde::Deserialize;
use serde::Serialize;

use crate::err::TrajDLError;
use crate::sequence::traits::PointSequence;
use crate::spatial::coord::Coordinate;
use crate::spatial::coord::GridCoord;
use crate::spatial::region::RectangleBoundary;
use traj_dist_rs::traits::AsCoord;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimpleGridSystem {
    pub boundary: RectangleBoundary,
    pub step_x: f64,
    pub step_y: f64,
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub num_x_grids: i64,
    pub num_y_grids: i64,
    /// If true, use column-major order for cell_id: cell_id = grid_x * num_y_grids + grid_y.
    /// If false (default), use row-major order: cell_id = grid_y * num_x_grids + grid_x.
    pub column_major: bool,
}

impl SimpleGridSystem {
    pub fn new(boundary: RectangleBoundary, step_x: f64, step_y: f64) -> Self {
        Self::with_column_major(boundary, step_x, step_y, false)
    }

    pub fn with_column_major(
        boundary: RectangleBoundary,
        step_x: f64,
        step_y: f64,
        column_major: bool,
    ) -> Self {
        let min_x = boundary.min_point.x();
        let min_y = boundary.min_point.y();
        let max_x = boundary.max_point.x();
        let max_y = boundary.max_point.y();

        let num_x_grids = ((max_x - min_x) / step_x).ceil() as i64;
        let num_y_grids = ((max_y - min_y) / step_y).ceil() as i64;

        Self {
            boundary,
            step_x,
            step_y,
            min_x,
            min_y,
            max_x,
            max_y,
            num_x_grids,
            num_y_grids,
            column_major,
        }
    }

    pub fn len(&self) -> usize {
        (self.num_x_grids * self.num_y_grids) as usize
    }

    pub fn in_boundary<C: AsCoord>(&self, coord: &C) -> bool {
        self.boundary.in_boundary(coord)
    }

    pub fn in_boundary_seq<T: PointSequence>(&self, seq: &T) -> Vec<bool> {
        self.boundary.in_boundary_seq(seq)
    }

    /// Locate the grid id by grid coordinate
    ///
    /// Parameters
    /// ----------
    /// grid_coord: GridCoord
    ///     grid coordinate
    ///
    /// Returns
    /// ----------
    /// int
    ///     grid id
    pub fn locate_by_grid_coordinate(&self, grid_coord: &GridCoord) -> i64 {
        if self.column_major {
            grid_coord.grid_x * self.num_y_grids + grid_coord.grid_y
        } else {
            grid_coord.grid_y * self.num_x_grids + grid_coord.grid_x
        }
    }

    /// Convert a longitude/latitude coordinate into a grid coordinate
    ///
    /// Parameters
    /// ----------
    /// coord: Coordinate
    ///     coordinate
    ///
    /// Returns
    /// ----------
    /// GridCoord
    ///     grid coordinate
    pub fn transform_coord_into_grid_coord<C: AsCoord>(&self, coord: &C) -> GridCoord {
        // TODO: 这里需要检查坐标系是否一致，否则报错
        let xoffset = (coord.x() - self.boundary.min_point.x()) / self.step_x;
        let yoffset = (coord.y() - self.boundary.min_point.y()) / self.step_y;
        GridCoord {
            grid_x: xoffset.floor() as i64,
            grid_y: yoffset.floor() as i64,
        }
    }

    /// Convert a longitude/latitude sequence into a grid-coordinate sequence
    /// TODO: complete the docstring
    pub fn transform_coord_into_grid_coord_seq<T: PointSequence>(
        &self,
        seq: &T,
    ) -> Result<Vec<GridCoord>, TrajDLError> {
        Ok(seq
            .iter_coords()
            .map(|coord| self.transform_coord_into_grid_coord(&coord))
            .collect())
    }

    /// Convert a longitude/latitude coordinate into a grid id
    ///
    /// Parameters
    /// ----------
    /// coord: Coordinate
    ///     coordinate
    ///
    /// Returns
    /// ----------
    /// int
    ///     grid id
    pub fn locate_in_grid<C: AsCoord>(&self, coord: &C) -> i64 {
        let grid_coord = self.transform_coord_into_grid_coord(coord);
        self.locate_by_grid_coordinate(&grid_coord)
    }

    pub fn locate_in_grid_seq<T: PointSequence>(&self, seq: &T, unk_loc: Option<i64>) -> Vec<i64> {
        let result: Vec<i64> = seq
            .iter_coords()
            .map(|coord| {
                let mut loc = self.locate_in_grid(&coord);
                if !self.in_boundary(&coord) {
                    if let Some(unknown_id) = unk_loc {
                        loc = unknown_id;
                    }
                }
                loc
            })
            .collect();

        result
    }

    /// Check whether a grid coordinate lies inside the region
    /// TODO: complete the docstring
    pub fn in_boundary_by_grid_coordinate(&self, grid_coord: &GridCoord) -> bool {
        let grid_x = grid_coord.grid_x;
        let grid_y = grid_coord.grid_y;
        0 <= grid_x && grid_x < self.num_x_grids && 0 <= grid_y && grid_y < self.num_y_grids
    }

    /// Convert a grid id into a grid coordinate
    pub fn to_grid_coordinate(&self, cell_id: i64) -> Result<GridCoord, TrajDLError> {
        // TODO: fix this bug later(cell_id should not be converted into usize)
        if cell_id < 0 || cell_id as usize >= self.len() {
            return Err(TrajDLError::OutofIndex(format!(
                "The given loc {} does not belong this grid.",
                cell_id
            )));
        }

        if self.column_major {
            Ok(GridCoord {
                grid_x: cell_id / self.num_y_grids,
                grid_y: cell_id % self.num_y_grids,
            })
        } else {
            Ok(GridCoord {
                grid_x: cell_id % self.num_x_grids,
                grid_y: cell_id / self.num_x_grids,
            })
        }
    }

    /// Convert a grid coordinate into the centroid coordinate of that grid
    ///
    /// Parameters
    /// ----------
    /// grid_coord: GridCoord
    ///     grid coordinate
    ///
    /// Returns
    /// ----------
    /// Coordinate
    ///     coordinate of the grid centroid
    pub fn grid_coord_to_centroid_point(&self, grid_coord: &GridCoord) -> Coordinate {
        let x = self.min_x + (grid_coord.grid_x as f64 + 0.5) * self.step_x;
        let y = self.min_y + (grid_coord.grid_y as f64 + 0.5) * self.step_y;
        // TODO: 这里默认是WGS84，实际上应该是根据boundary的坐标系来判断
        Coordinate::Wgs84 { lng: x, lat: y }
    }

    /// Convert a group of grid symbols into their grid centroid coordinates
    ///
    /// A grid symbol has the format "G_{grid_x}_{grid_y}"; this method parses it into a grid coordinate,
    /// then computes and returns the centroid coordinate of that grid.
    ///
    /// Parameters
    /// ----------
    /// grid_symbols: &[String]
    ///     list of grid symbols, each element in the format "G_{grid_x}_{grid_y}"
    ///
    /// Returns
    /// ----------
    /// Vec<Coordinate>
    ///     list of corresponding grid centroid coordinates
    pub fn grid_symbols_to_centroid_coords(&self, grid_symbols: &[String]) -> Vec<Coordinate> {
        grid_symbols
            .iter()
            .map(|grid_symbol| GridCoord::from_grid_symbol(grid_symbol.clone()))
            .map(|grid_coord| self.grid_coord_to_centroid_point(&grid_coord))
            .collect()
    }

    /// Convert one grid symbol into its grid centroid coordinate
    ///
    /// A grid symbol has the format "G_{grid_x}_{grid_y}"; this method parses it into a grid coordinate,
    /// then computes and returns the centroid coordinate of that grid.
    ///
    /// Parameters
    /// ----------
    /// grid_symbol: String
    ///     grid symbol, format "G_{grid_x}_{grid_y}"
    ///
    /// Returns
    /// ----------
    /// Coordinate
    ///     the corresponding grid centroid coordinate
    pub fn grid_symbol_to_centroid_coord(&self, grid_symbol: String) -> Coordinate {
        self.grid_coord_to_centroid_point(&GridCoord::from_grid_symbol(grid_symbol))
    }

    /// Enumerate all grid symbols in the grid system, producing a token id sequence
    /// exactly identical to `locate_by_grid_coordinate` according to the current `column_major` setting.
    ///
    /// The return value is a `Vec<(String, i64)>` in enumeration order:
    ///   - `column_major=true` (column-major): outer `i_x ∈ [0, num_x)`, inner `i_y ∈ [0, num_y)`,
    ///      the id of the `k`-th element equals `i_x * num_y + i_y == k`.
    ///   - `column_major=false` (row-major): outer `i_y ∈ [0, num_y)`, inner `i_x ∈ [0, num_x)`,
    ///      the id of the `k`-th element equals `i_y * num_x + i_x == k`.
    ///
    /// This sequence guarantees:
    ///   1. length == `num_x_grids * num_y_grids`;
    ///   2. ids take the values `0..num_x_grids * num_y_grids`, contiguous with no gaps;
    ///   3. ids match `locate_by_grid_coordinate(GridCoord{grid_x, grid_y})` exactly;
    ///   4. symbols and ids are aligned in both directions, so upper layers (e.g. TrajCLTokenizer.fit) can feed the vocabulary directly.
    pub fn enumerate_grid_symbols(&self) -> Vec<(String, i64)> {
        let num_x = self.num_x_grids;
        let num_y = self.num_y_grids;
        let total = (num_x * num_y) as usize;
        let mut out: Vec<(String, i64)> = Vec::with_capacity(total);
        if self.column_major {
            for i_x in 0..num_x {
                for i_y in 0..num_y {
                    let coord = GridCoord {
                        grid_x: i_x,
                        grid_y: i_y,
                    };
                    out.push((
                        coord.get_grid_symbol(),
                        self.locate_by_grid_coordinate(&coord),
                    ));
                }
            }
        } else {
            for i_y in 0..num_y {
                for i_x in 0..num_x {
                    let coord = GridCoord {
                        grid_x: i_x,
                        grid_y: i_y,
                    };
                    out.push((
                        coord.get_grid_symbol(),
                        self.locate_by_grid_coordinate(&coord),
                    ));
                }
            }
        }
        out
    }
}

/// Uniform degree-space grid system (the UniTE basis).
///
/// Two deterministic differences from [`SimpleGridSystem`] (from the exact semantics of UniTE
/// `data.py::project_to_grid`):
///
/// 1. **Binning formula**: `t = (x - min) / (max - min) * num`, `idx = ceil(t) - 1` then
///    clipped to `[0, num-1]`. For non-boundary points this equals floor; **points exactly on a
///    boundary go to the lower cell** (SimpleGrid's floor semantics put them in the upper cell),
///    and at `x == max` the value is clipped into the last cell (floor would go out of range).
/// 2. **Data-driven fit**: the bounds come from the data's own min/max (`UniteGridTokenizer.fit`)
///    rather than being specified by hand; `num = ceil(span / delta)`.
///
/// Token composition matches UniTE: `cell_id = grid_y * num_x_grids + grid_x` (row-major,
/// y outer). The expression order `(x - min) / span * num` mirrors UniTE token for token; under
/// f64 it is bit-for-bit identical to the numpy implementation (locked by a parity fixture).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UniformDegreeGridSystem {
    pub delta: f64,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub num_x_grids: i64,
    pub num_y_grids: i64,
}

impl UniformDegreeGridSystem {
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64, delta: f64) -> Self {
        if !delta.is_finite() || delta <= 0.0 {
            panic!("delta must be a positive finite number, got {}", delta);
        }
        if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
            panic!("bounds must be finite");
        }
        if max_x <= min_x || max_y <= min_y {
            panic!(
                "invalid bounds: require max > min on both axes, got x=({}, {}), y=({}, {})",
                min_x, max_x, min_y, max_y
            );
        }
        Self {
            delta,
            min_x,
            max_x,
            min_y,
            max_y,
            num_x_grids: ((max_x - min_x) / delta).ceil() as i64,
            num_y_grids: ((max_y - min_y) / delta).ceil() as i64,
        }
    }

    pub fn len(&self) -> usize {
        (self.num_x_grids * self.num_y_grids) as usize
    }

    /// UniTE exact formula: `idx = clip(ceil((v - min) / span * num) - 1, 0, num - 1)`.
    fn axis_index(&self, v: f64, min: f64, max: f64, num: i64) -> i64 {
        let t = (v - min) / (max - min) * num as f64;
        let idx = (t.ceil() - 1.0) as i64;
        idx.clamp(0, num - 1)
    }

    pub fn in_bounds(&self, x: f64, y: f64) -> bool {
        (self.min_x..=self.max_x).contains(&x) && (self.min_y..=self.max_y).contains(&y)
    }

    pub fn locate_by_grid_coordinate(&self, grid_coord: &GridCoord) -> i64 {
        grid_coord.grid_y * self.num_x_grids + grid_coord.grid_x
    }

    pub fn transform_coord_into_grid_coord(&self, x: f64, y: f64) -> GridCoord {
        GridCoord {
            grid_x: self.axis_index(x, self.min_x, self.max_x, self.num_x_grids),
            grid_y: self.axis_index(y, self.min_y, self.max_y, self.num_y_grids),
        }
    }

    pub fn locate_in_grid(&self, x: f64, y: f64) -> i64 {
        let grid_coord = self.transform_coord_into_grid_coord(x, y);
        self.locate_by_grid_coordinate(&grid_coord)
    }

    /// Batch locating: equal-length `(xs, ys)` coordinate arrays → cell id array. Sequential implementation (see D3).
    pub fn locate_in_grid_batch(&self, xs: &[f64], ys: &[f64]) -> Result<Vec<i64>, TrajDLError> {
        if xs.len() != ys.len() {
            return Err(TrajDLError::DataConvertionError(format!(
                "xs.len() ({}) != ys.len() ({})",
                xs.len(),
                ys.len()
            )));
        }
        Ok(xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| self.locate_in_grid(x, y))
            .collect())
    }

    pub fn to_grid_coordinate(&self, cell_id: i64) -> Result<GridCoord, TrajDLError> {
        if cell_id < 0 || cell_id as usize >= self.len() {
            return Err(TrajDLError::OutofIndex(format!(
                "The given loc {} does not belong this grid.",
                cell_id
            )));
        }
        Ok(GridCoord {
            grid_x: cell_id % self.num_x_grids,
            grid_y: cell_id / self.num_x_grids,
        })
    }

    /// Grid centroid. The cell width is `span / num`; under ceil binning, it equals delta exactly
    /// when span is an integer multiple of delta.
    pub fn grid_coord_to_centroid_point(&self, grid_coord: &GridCoord) -> Coordinate {
        let x = self.min_x
            + (grid_coord.grid_x as f64 + 0.5) * (self.max_x - self.min_x)
                / self.num_x_grids as f64;
        let y = self.min_y
            + (grid_coord.grid_y as f64 + 0.5) * (self.max_y - self.min_y)
                / self.num_y_grids as f64;
        Coordinate::Wgs84 { lng: x, lat: y }
    }

    /// Enumerate all cells: `(grid_symbol, cell_id)`, row-major with contiguous ids,
    /// consistent with `locate_by_grid_coordinate` (so UniteGridTokenizer can feed the vocabulary directly).
    pub fn enumerate_grid_symbols(&self) -> Vec<(String, i64)> {
        let mut out: Vec<(String, i64)> = Vec::with_capacity(self.len());
        for i_y in 0..self.num_y_grids {
            for i_x in 0..self.num_x_grids {
                let coord = GridCoord {
                    grid_x: i_x,
                    grid_y: i_y,
                };
                out.push((
                    coord.get_grid_symbol(),
                    self.locate_by_grid_coordinate(&coord),
                ));
            }
        }
        out
    }
}

#[cfg(test)]
mod uniform_degree_tests {
    use super::*;

    // [0, 4] x [10, 12], delta = 1 → num_x = 4, num_y = 2, 共 8 格
    fn make_grid() -> UniformDegreeGridSystem {
        UniformDegreeGridSystem::new(0.0, 4.0, 10.0, 12.0, 1.0)
    }

    #[test]
    fn num_grids_from_delta() {
        let g = make_grid();
        assert_eq!(g.num_x_grids, 4);
        assert_eq!(g.num_y_grids, 2);
        assert_eq!(g.len(), 8);
    }

    #[test]
    fn interior_points_follow_floor_semantics() {
        let g = make_grid();
        // t = 0.5 → ceil(0.5)-1 = 0；t = 2.5 → ceil(2.5)-1 = 2
        assert_eq!(g.locate_in_grid(0.5, 10.5), 0); // (grid_x=0, grid_y=0) → 0*4+0
        assert_eq!(g.locate_in_grid(2.5, 11.5), 6); // (grid_x=2, grid_y=1) → 1*4+2
    }

    #[test]
    fn boundary_points_go_to_lower_cell() {
        let g = make_grid();
        // x=2.0 是 cell 1/2 的精确边界：UniTE ceil(2)-1 = 1（下格），
        // 而 SimpleGrid 的 floor 语义会给 2（上格）——两者确定性不同
        assert_eq!(g.locate_in_grid(2.0, 10.5), 1);
        assert_eq!(g.locate_in_grid(0.0, 10.5), 0); // t=0 → clip(-1) = 0
        assert_eq!(g.locate_in_grid(4.0, 10.5), 3); // x=max：t=4 → ceil-1=3，无需越界处理
        assert_eq!(g.locate_in_grid(4.5, 10.5), 3); // 超界 → clip 进最后一格
    }

    #[test]
    fn cell_id_is_row_major_y_outer() {
        let g = make_grid();
        assert_eq!(
            g.locate_by_grid_coordinate(&GridCoord {
                grid_x: 0,
                grid_y: 0
            }),
            0
        );
        assert_eq!(
            g.locate_by_grid_coordinate(&GridCoord {
                grid_x: 3,
                grid_y: 0
            }),
            3
        );
        assert_eq!(
            g.locate_by_grid_coordinate(&GridCoord {
                grid_x: 0,
                grid_y: 1
            }),
            4
        );
        assert_eq!(
            g.locate_by_grid_coordinate(&GridCoord {
                grid_x: 2,
                grid_y: 1
            }),
            6
        );
    }

    #[test]
    fn round_trip_to_grid_coordinate() {
        let g = make_grid();
        for cell_id in 0..g.len() as i64 {
            let coord = g.to_grid_coordinate(cell_id).unwrap();
            assert_eq!(g.locate_by_grid_coordinate(&coord), cell_id);
        }
        assert!(g.to_grid_coordinate(8).is_err());
        assert!(g.to_grid_coordinate(-1).is_err());
    }

    #[test]
    fn batch_matches_single() {
        let g = make_grid();
        let xs = vec![0.0, 0.5, 2.0, 2.5, 4.0, 4.5];
        let ys = vec![10.0, 10.5, 11.0, 11.5, 12.0, 10.1];
        let batch = g.locate_in_grid_batch(&xs, &ys).unwrap();
        for ((&x, &y), &b) in xs.iter().zip(ys.iter()).zip(batch.iter()) {
            assert_eq!(b, g.locate_in_grid(x, y));
        }
        assert!(g.locate_in_grid_batch(&xs, &ys[..3]).is_err());
    }

    #[test]
    fn enumerate_grid_symbols_continuous() {
        let g = make_grid();
        let pairs = g.enumerate_grid_symbols();
        assert_eq!(pairs.len(), 8);
        let ids: Vec<i64> = pairs.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids, (0..8).collect::<Vec<i64>>());
        // 行主序：第一个 G_0_0，第五个（第二行开头）G_0_1
        assert_eq!(pairs[0], ("G_0_0".to_string(), 0));
        assert_eq!(pairs[4], ("G_0_1".to_string(), 4));
        assert_eq!(pairs[6], ("G_2_1".to_string(), 6));
    }

    #[test]
    fn non_multiple_span_uses_ceil_num() {
        // span=3.5, delta=1 → num=4，最后一格只覆盖 [2.625, 3.5]
        let g = UniformDegreeGridSystem::new(0.0, 3.5, 0.0, 1.0, 1.0);
        assert_eq!(g.num_x_grids, 4);
        assert_eq!(g.locate_in_grid(3.5, 0.5), 3);
        assert_eq!(g.locate_in_grid(2.9, 0.5), 3); // 2.9 > 2.625（第 4 格下界）
        assert_eq!(g.locate_in_grid(2.6, 0.5), 2); // t=2.6/3.5*4=2.97 → ceil-1=2
    }

    #[test]
    #[should_panic(expected = "delta must be")]
    fn rejects_bad_delta() {
        UniformDegreeGridSystem::new(0.0, 1.0, 0.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "invalid bounds")]
    fn rejects_inverted_bounds() {
        UniformDegreeGridSystem::new(1.0, 0.0, 0.0, 1.0, 0.001);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::coord::Coordinate;
    use crate::spatial::region::RectangleBoundary;

    fn make_grid(column_major: bool) -> SimpleGridSystem {
        // 4 x 3 网格（num_x_grids=4, num_y_grids=3，共 12 个 cell）
        let boundary = RectangleBoundary::new(
            Coordinate::WebMercator { x: 0.0, y: 0.0 },
            Coordinate::WebMercator { x: 400.0, y: 300.0 },
        );
        SimpleGridSystem::with_column_major(boundary, 100.0, 100.0, column_major)
    }

    #[test]
    fn enumerate_grid_symbols_column_major_invariants() {
        let grid = make_grid(true);
        let pairs = grid.enumerate_grid_symbols();

        // 1. 长度等于 num_x * num_y
        assert_eq!(pairs.len(), (grid.num_x_grids * grid.num_y_grids) as usize);

        // 2. id 严格 0..N 连续
        let ids: Vec<i64> = pairs.iter().map(|(_, id)| *id).collect();
        let expected_ids: Vec<i64> = (0..pairs.len() as i64).collect();
        assert_eq!(ids, expected_ids);

        // 3. id 必须与 locate_by_grid_coordinate 一致；symbol 必须与 GridCoord 双向对齐
        for (sym, id) in &pairs {
            let coord = GridCoord::from_grid_symbol(sym.clone());
            assert_eq!(grid.locate_by_grid_coordinate(&coord), *id);
            assert_eq!(coord.get_grid_symbol(), *sym);
        }

        // 4. 列主序下，第一个一定是 G_0_0:0，最后一个一定是 G_{nx-1}_{ny-1}:N-1
        assert_eq!(pairs.first().unwrap(), &("G_0_0".to_string(), 0));
        let last_sym = format!("G_{}_{}", grid.num_x_grids - 1, grid.num_y_grids - 1);
        assert_eq!(
            pairs.last().unwrap(),
            &(last_sym, (grid.num_x_grids * grid.num_y_grids - 1))
        );
    }

    #[test]
    fn enumerate_grid_symbols_row_major_invariants() {
        let grid = make_grid(false);
        let pairs = grid.enumerate_grid_symbols();

        assert_eq!(pairs.len(), (grid.num_x_grids * grid.num_y_grids) as usize);

        // 行主序下 id 同样应该 0..N 连续
        let ids: Vec<i64> = pairs.iter().map(|(_, id)| *id).collect();
        let expected_ids: Vec<i64> = (0..pairs.len() as i64).collect();
        assert_eq!(ids, expected_ids);

        // id 必须等于 locate_by_grid_coordinate 在行主序下的输出
        for (sym, id) in &pairs {
            let coord = GridCoord::from_grid_symbol(sym.clone());
            assert_eq!(grid.locate_by_grid_coordinate(&coord), *id);
        }

        // 行主序下，第一个仍是 G_0_0:0；遍历顺序变成 i_y 外层，所以第二个应是 G_1_0:1
        assert_eq!(pairs.first().unwrap(), &("G_0_0".to_string(), 0));
        assert_eq!(pairs[1], ("G_1_0".to_string(), 1));
    }

    #[test]
    fn enumerate_grid_symbols_unique_keys() {
        // 不同 cell 必须产出不同的 symbol，保证后续灌入 HashMap/dict 时不会覆盖
        let grid = make_grid(true);
        let pairs = grid.enumerate_grid_symbols();
        let unique_syms: std::collections::HashSet<&String> =
            pairs.iter().map(|(s, _)| s).collect();
        assert_eq!(unique_syms.len(), pairs.len());
    }
}
