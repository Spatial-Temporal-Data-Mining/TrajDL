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

#[cfg(test)]
mod tests {
    use traj_dist_rs::traits::AsCoord;

    #[test]
    fn test_array_impl() {
        let point = [1.5, -2.7];
        assert_eq!(point.x(), 1.5);
        assert_eq!(point.y(), -2.7);
    }

    #[test]
    fn test_array_ref_impl() {
        let point = [0.0, 3.14];
        let point_ref: &[f64; 2] = &point;
        assert_eq!(point_ref.x(), 0.0);
        assert_eq!(point_ref.y(), 3.14);
    }

    // 确保可以用于泛型函数
    #[test]
    fn test_generic_usage() {
        fn distance<T: AsCoord>(p: &T) -> f64 {
            (p.x().powi(2) + p.y().powi(2)).sqrt()
        }

        let point = [3.0, 4.0];
        assert_eq!(distance(&point), 5.0);
    }
}
