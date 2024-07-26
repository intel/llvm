# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import Benchmark
from .result import Result
from .velocity import VelocityBase, VelocityBench
from utils.utils import run
import re

class SobelFilter(VelocityBase):
    def __init__(self, vb: VelocityBench):
        super().__init__("sobel_filter", "sobel_filter", vb)

    def download_deps(self):
        self.download_untar("sobel_filter", "https://github.com/oneapi-src/Velocity-Bench/raw/main/sobel_filter/res/sobel_filter_data.tgz?download=", "sobel_filter_data.tgz")
        return

    def name(self):
        return "Velocity-Bench Sobel Filter"

    def unit(self):
        return "ms"

    def bin_args(self) -> list[str]:
        return ["-i", f"{self.data_path}/sobel_filter_data/silverfalls_32Kx32K.png",
                "-n", "5"]

    def extra_env_vars(self) -> dict:
        return {"OPENCV_IO_MAX_IMAGE_PIXELS" : "1677721600"}

    def parse_output(self, stdout: str) -> float:
        match = re.search(r'sobelfilter - total time for whole calculation: (\d+\.\d+) s', stdout)
        if match:
            return round(float(match.group(1)) * 1000, 3)
        else:
            raise ValueError("Failed to parse benchmark output.")

