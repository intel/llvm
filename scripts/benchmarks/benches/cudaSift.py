# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import Benchmark
from .result import Result
from .velocity import VelocityBase, VelocityBench
from utils.utils import run
import os
import re
import shutil

class CudaSift(VelocityBase):
    def __init__(self, vb: VelocityBench):
        super().__init__("cudaSift", "cudaSift", vb)

    def download_deps(self):
        images = os.path.join(self.vb.repo_path, self.bench_name, 'inputData')
        dest = os.path.join(self.directory, 'inputData')
        if not os.path.exists(dest):
            shutil.copytree(images, dest)

    def name(self):
        return "Velocity-Bench CudaSift"

    def unit(self):
        return "ms"

    def parse_output(self, stdout: str) -> float:
        match = re.search(r'Avg workload time = (\d+\.\d+) ms', stdout)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Failed to parse benchmark output.")
