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

class Bitcracker(VelocityBase):
    def __init__(self, vb: VelocityBench):
        super().__init__("bitcracker", "bitcracker", vb)
        self.data_path = os.path.join(vb.repo_path, "bitcracker", "hash_pass")

    def name(self):
        return "Velocity-Bench Bitcracker"

    def unit(self):
        return "s"

    def bin_args(self) -> list[str]:
        return ["-f", f"{self.data_path}/img_win8_user_hash.txt",
                "-d", f"{self.data_path}/user_passwords_60000.txt",
                "-b", "60000"]

    def parse_output(self, stdout: str) -> float:
        match = re.search(r'bitcracker - total time for whole calculation: (\d+\.\d+) s', stdout)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("{self.__class__.__name__}: Failed to parse benchmark output.")
