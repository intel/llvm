# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass, field
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
    # stddev can be optionally set by the benchmark,
    # if not set, it will be calculated automatically.
    stddev: float = 0.0
    git_url: str = ""
    git_hash: str = ""
    # values below should not be set by the benchmark
    name: str = ""
    lower_is_better: bool = True
    suite: str = "Unknown"


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
    compute_runtime: str = "Unknown"


@dataclass_json
@dataclass
class BenchmarkTag:
    name: str
    description: str = ""


@dataclass_json
@dataclass
class BenchmarkMetadata:
    type: str = "benchmark"  # or 'group'
    description: str = None
    notes: str = None
    unstable: str = None
    tags: list[str] = field(default_factory=list)
    range_min: float = None
    range_max: float = None
    display_name: str = None
    explicit_group: str = None


@dataclass_json
@dataclass
class BenchmarkOutput:
    runs: list[BenchmarkRun]
    metadata: dict[str, BenchmarkMetadata]
    tags: dict[str, BenchmarkTag]
    default_compare_names: list[str] = field(default_factory=list)
