# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass, field
from typing import Optional
from dataclasses_json import config, dataclass_json
from datetime import datetime


@dataclass_json
@dataclass
class Result:
    label: str
    value: float
    command: list[str]
    env: dict[str, str]
    stdout: str
    passed: bool = True
    unit: str = ""
    explicit_group: str = ""
    # stddev can be optionally set by the benchmark,
    # if not set, it will be calculated automatically.
    stddev: float = 0.0
    # values below should not be set by the benchmark
    name: str = ""
    lower_is_better: bool = True
    suite: str = "Unknown"
    description: str = "No description provided."


@dataclass_json
@dataclass
class BenchmarkRun:
    results: list[Result]
    name: str = "This PR"
    hostname: str = "Unknown"
    git_hash: str = ""
    github_repo: str = None
    date: datetime = field(
        default=None,
        metadata=config(encoder=datetime.isoformat, decoder=datetime.fromisoformat),
    )
