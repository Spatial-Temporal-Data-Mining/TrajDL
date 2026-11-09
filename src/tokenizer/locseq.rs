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

use pyo3::{Bound, types::PyList};

use std::collections::HashMap;

use pyo3::{PyResult, pyfunction};
use pyo3_stub_gen_derive::gen_stub_pyfunction;

use crate::binding::seq::types::PyLocSeqType;

/// Count the locations that occur in seqs
///
/// Parameters
/// ----------
/// seqs: List[LocSeq] | List[List[str]]
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (seqs, bos_token = None, eos_token = None))]
pub fn count_locations(
    seqs: &Bound<'_, PyList>,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> PyResult<HashMap<Option<String>, i64>> {
    // 统计每个位置的出现频数
    let mut freq = HashMap::<Option<String>, i64>::new();

    for seq in seqs {
        let loc_seq = PyLocSeqType::try_from(&seq)?;
        for loc in loc_seq.iter_locs() {
            *freq.entry(loc?).or_insert(0) += 1;
        }
        if let Some(bos) = bos_token {
            *freq.entry(Some(bos.to_string())).or_insert(0) += 1;
        }
        if let Some(eos) = eos_token {
            *freq.entry(Some(eos.to_string())).or_insert(0) += 1;
        }
    }
    Ok(freq)
}
