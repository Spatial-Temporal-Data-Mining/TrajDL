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

//! Arrow-backed dataset utilities.
//!
//! This module owns the canonical Arrow schema definitions for
//! `Trajectory` / `LocSeq` datasets and provides high-throughput
//! `Vec<Trajectory|LocSeq> -> RecordBatch` builders. Python-side
//! `init_from_seqs` calls into these builders via the PyO3 binding
//! layer (`crate::binding::datasets`) to avoid the per-element
//! Python-level Arrow append loop that bottlenecks large dataset
//! construction.

pub mod builder;
pub mod schema;
