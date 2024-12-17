# Copyright 2024 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import threading

import requests
from tqdm import tqdm


class FileDownloader:
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.file_size = 0
        self.chunk_size = 0
        self.supports_range = False

    def check_range_support(self, url):
        response = requests.head(url)
        # 检查 Accept-Ranges 头
        if "Accept-Ranges" in response.headers:
            if response.headers["Accept-Ranges"] == "bytes":
                self.supports_range = True
        else:
            self.supports_range = False

        # 获取文件大小
        self.file_size = int(response.headers.get("content-length", 0))
        self.chunk_size = (
            self.file_size // self.num_threads
            if self.supports_range
            else self.file_size
        )

    def download_chunk(self, url, start, end, thread_id, progress_bar, temp_dir):
        headers = {"Range": f"bytes={start}-{end}"} if self.supports_range else {}
        response = requests.get(url, headers=headers, stream=True)

        # 写入临时文件
        with open(os.path.join(temp_dir, f"part_{thread_id}"), "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                progress_bar.update(len(chunk))

    def download(self, url, output_path):
        self.check_range_support(url)

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()

        try:
            if self.supports_range:
                print("服务器支持 Range 请求，使用多线程下载...")
                threads = []
                progress_bar = tqdm(
                    total=self.file_size, unit="B", unit_scale=True, desc="Downloading"
                )

                for i in range(self.num_threads):
                    start = i * self.chunk_size
                    end = (
                        start + self.chunk_size - 1 if i < self.num_threads - 1 else ""
                    )
                    thread = threading.Thread(
                        target=self.download_chunk,
                        args=(url, start, end, i, progress_bar, temp_dir),
                    )
                    threads.append(thread)
                    thread.start()

                # 等待所有线程完成
                for thread in threads:
                    thread.join()
                progress_bar.close()
                self.merge_files(temp_dir, output_path)
            else:
                print("服务器不支持 Range 请求，使用单线程下载...")
                progress_bar = tqdm(
                    total=self.file_size, unit="B", unit_scale=True, desc="Downloading"
                )
                self.download_chunk(
                    url, 0, self.file_size - 1, 0, progress_bar, temp_dir
                )
                progress_bar.close()
                self.merge_files(temp_dir, output_path)
        finally:
            # 删除临时目录
            shutil.rmtree(temp_dir)

    def merge_files(self, temp_dir, output_path):
        with open(output_path, "wb") as output_file:
            for i in range(self.num_threads):
                part_file_path = os.path.join(temp_dir, f"part_{i}")
                if os.path.exists(part_file_path):
                    with open(part_file_path, "rb") as input_file:
                        output_file.write(input_file.read())
