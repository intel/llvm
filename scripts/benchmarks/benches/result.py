# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Result:
    label: str
    value: float
    command: str
    env: str
    stdout: str
    passed: bool = True
    unit: str = ""
    name: str = ""
    lower_is_better: bool = True
