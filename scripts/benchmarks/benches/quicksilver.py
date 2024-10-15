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

class QuickSilver(VelocityBase):
    def __init__(self, vb: VelocityBench):
        super().__init__("QuickSilver", "qs", vb)
        self.data_path = os.path.join(vb.repo_path, "QuickSilver", "Examples", "AllScattering")

    def run(self, env_vars) -> list[Result]:
        # TODO: fix the crash in QuickSilver when UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
        if 'UR_L0_USE_IMMEDIATE_COMMANDLISTS' in env_vars and env_vars['UR_L0_USE_IMMEDIATE_COMMANDLISTS'] == '0':
            return None

        return super().run(env_vars)

    def name(self):
        return "Velocity-Bench QuickSilver"

    def unit(self):
        return "MMS/CTT"

    def lower_is_better(self):
        return False

    def bin_args(self) -> list[str]:
        return ["-i", f"{self.data_path}/scatteringOnly.inp"]

    def extra_env_vars(self) -> dict:
        return {"QS_DEVICE" : "GPU"}

    def parse_output(self, stdout: str) -> float:
        match = re.search(r'Figure Of Merit\s+(\d+\.\d+)', stdout)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("{self.__class__.__name__}: Failed to parse benchmark output.")
