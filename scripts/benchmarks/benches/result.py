# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Optional
from dataclasses_json import dataclass_json
from datetime import datetime

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
    # stddev can be optionally set by the benchmark,
    # if not set, it will be calculated automatically.
    stddev: float = 0.0
    # values below should not be set by the benchmark
    name: str = ""
    lower_is_better: bool = True
    git_hash: str = ''
    date: Optional[datetime] = None

@dataclass_json
@dataclass
class BenchmarkRun:
    results: list[Result]
    name: str = 'This PR'
    git_hash: str = ''
    date: datetime = None
