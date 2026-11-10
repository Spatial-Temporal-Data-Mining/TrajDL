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

//! PyO3 bindings for `crate::datasets`.
//!
//! Exposes:
//! - `get_traj_schema()` / `get_loc_seq_schema()`: zero-copy `pa.Schema`.
//! - `build_traj_record_batch(seqs)` / `build_loc_seq_record_batch(seqs)`:
//!   construct a `pa.RecordBatch` directly in Rust, returned to Python via
//!   the Arrow C Data Interface (no copy; Arrow buffers stay in their
//!   original allocation, lifetime managed by `Arc` refcounting on both
//!   sides of the FFI boundary).

pub mod schema;
