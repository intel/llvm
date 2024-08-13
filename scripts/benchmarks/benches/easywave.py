# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import Benchmark
from .result import Result
from .velocity import VelocityBase, VelocityBench
from utils.utils import run
from .options import options
import re
import os

class Easywave(VelocityBase):
    def __init__(self, vb: VelocityBench):
        super().__init__("easywave", "easyWave_sycl", vb)

    def download_deps(self):
        self.download_untar("easywave", "https://git.gfz-potsdam.de/id2/geoperil/easyWave/-/raw/master/data/examples.tar.gz", "examples.tar.gz")

    def name(self):
        return "Velocity-Bench Easywave"

    def unit(self):
        return "ms"

    def bin_args(self) -> list[str]:
        return ["-grid", f"{self.data_path}/examples/e2Asean.grd",
                "-source", f"{self.data_path}/examples/BengkuluSept2007.flt",
                "-time", "120"]

    # easywave doesn't output a useful single perf value. Instead, we parse the
    # output logs looking for the very last line containing the elapsed time of the
    # application.
    def get_last_elapsed_time(self, log_file_path) -> float:
        elapsed_time_pattern = re.compile(r'Model time = (\d{2}:\d{2}:\d{2}),\s+elapsed: (\d+) msec')
        last_elapsed_time = None

        try:
            with open(log_file_path, 'r') as file:
                for line in file:
                    match = elapsed_time_pattern.search(line)
                    if match:
                        last_elapsed_time = int(match.group(2))
            
            if last_elapsed_time is not None:
                return last_elapsed_time
            else:
                raise ValueError("No elapsed time found in the log file.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {log_file_path} does not exist.")
        except Exception as e:
            raise e

    def parse_output(self, stdout: str) -> float:
        return self.get_last_elapsed_time(os.path.join(options.benchmark_cwd, "easywave.log"))
