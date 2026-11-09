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

// 定义一个宏来为所有结构体实现Sequence trait
#[macro_export]
macro_rules! impl_read_sequence_meta_trait {
    // 宏匹配任何名为 $type 的类型
    ($type:ty) => {
        impl $crate::sequence::traits::ReadSequenceMeta for $type {
            fn get_entity_id(&self) -> Option<&str> {
                self.meta.entity_id.as_deref()
            }

            fn get_start_ts(&self) -> Option<i64> {
                self.meta.start_ts
            }

            fn get_ts_seq(&self) -> Result<Option<Vec<i64>>, TrajDLError> {
                Ok(self
                    .iter_ts_seq()?
                    .map(|iter| iter.collect())
                    .and_then(|vec: Vec<i64>| (!vec.is_empty()).then_some(vec)))
            }

            fn iter_ts_seq(&self) -> Result<Option<impl Iterator<Item = i64>>, TrajDLError> {
                $crate::sequence::traits::IterMetaSeq::iter_ts_seq(&self.meta)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_write_sequence_meta_trait {
    // 宏匹配任何名为 $type 的类型
    ($type:ty) => {
        impl $crate::sequence::traits::WriteSequenceMeta for $type {
            fn set_entity_id(&mut self, entity_id: Option<String>) {
                self.meta.entity_id = entity_id;
            }

            fn set_start_ts(&mut self, start_ts: Option<i64>) {
                self.meta.start_ts = start_ts
            }

            fn set_ts_seq(&mut self, ts_seq: Option<Vec<i64>>) {
                self.meta.ts_seq = ts_seq
            }
        }
    };
}
