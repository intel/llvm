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

class Hashtable(VelocityBase):
    def __init__(self, vb: VelocityBench):
        super().__init__("hashtable", "hashtable_sycl", vb)

    def name(self):
        return "Velocity-Bench Hashtable"

    def unit(self):
        return "M keys/sec"

    def bin_args(self) -> list[str]:
        return ["--no-verify"]

    def lower_is_better(self):
        return False

    def parse_output(self, stdout: str) -> float:
        match = re.search(r'(\d+\.\d+) million keys/second', stdout)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Failed to parse keys per second from benchmark output.")
