# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass, field
from enum import Enum
import multiprocessing
import os


class Compare(Enum):
    LATEST = "latest"
    AVERAGE = "average"
    MEDIAN = "median"


class MarkdownSize(Enum):
    SHORT = "short"
    FULL = "full"


@dataclass
class DetectVersionsOptions:
    """
    Options for automatic version detection
    """

    # Components to detect versions for:
    sycl: bool = False
    compute_runtime: bool = False
    # umf: bool = False
    # level_zero: bool = False

    # Placeholder text, should automatic version detection fail: This text will
    # only be used if automatic version detection for x component is explicitly
    # specified.
    not_found_placeholder = "unknown"  # None

    # TODO unauthenticated users only get 60 API calls per hour: this will not
    # work if we enable benchmark CI in precommit.
    compute_runtime_tag_api: str = (
        "https://api.github.com/repos/intel/compute-runtime/tags"
    )
    # Max amount of api calls permitted on each run of the benchmark scripts
    max_api_calls = 4


@dataclass
class Options:
    TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
    workdir: str = None
    sycl: str = None
    ur: str = None
    ur_adapter: str = None
    umf: str = None
    redownload: bool = False
    benchmark_cwd: str = "INVALID"
    timeout: float = 600
    iterations: int = 3
    save_name: str = None
    compare: Compare = Compare.LATEST
    compare_max: int = 10  # average/median over how many results
    output_markdown: MarkdownSize = MarkdownSize.SHORT
    output_html: str = "local"
    output_directory: str = None
    dry_run: bool = False
    stddev_threshold: float = 0.02
    iterations_stddev: int = 5
    build_compute_runtime: bool = False
    extra_ld_libraries: list[str] = field(default_factory=list)
    extra_env_vars: dict = field(default_factory=dict)
    compute_runtime_tag: str = "25.27.34303.5"
    build_igc: bool = False
    current_run_name: str = "This PR"
    preset: str = "Full"
    build_jobs: int = len(os.sched_getaffinity(0))  # Cores available for the process.
    exit_on_failure: bool = False
    flamegraph: bool = False
    unitrace: bool = False

    # Options intended for CI:

    regression_threshold: float = 0.05
    # It's necessary in CI to compare or redo benchmark runs. Instead of
    # generating a new timestamp each run by default, specify a single timestamp
    # to use across the entire CI run.
    timestamp_override: str = None
    # The default directory to fetch results from is args.benchmark_directory,
    # hence a default value of "None" as the value is decided during runtime.
    #
    # However, sometimes you may want to fetch results from a different
    # directory, i.e. in CI when you clone the results directory elsewhere.
    results_directory_override: str = None
    # By default, we fetch SYCL commit info from the folder where main.py is
    # located. This doesn't work right when CI uses different commits for e.g.
    # CI scripts vs SYCl build source.
    github_repo_override: str = None
    git_commit_override: str = None
    # Filename used to store Github summary files:
    github_summary_filename: str = "github_summary.md"
    # Archiving settings
    # Archived runs are stored separately from the main dataset but are still accessible
    # via the HTML UI when "Include archived runs" is enabled.
    # Archived runs older than 3 times the specified days are not included in the dashboard,
    # ie. when archiving data older than 7 days, runs older than 21 days are not included.
    archive_baseline_days: int = 30  # Archive Baseline_* runs after 30 days
    archive_pr_days: int = 7  # Archive other (PR/dev) runs after 7 days

    # EWMA Options:

    # The smoothing factor is alpha in the EWMA equation. Generally, a higher
    # smoothing factor results in newer data having more weight, and a lower
    # smoothing factor results in older data having more weight.
    #
    # Valid values for this smoothing factor ranges from (0, 1). Note that no
    # value of smothing factor will result in older elements having more weight
    # than newer elements.
    EWMA_smoothing_factor: float = 0.15

    detect_versions: DetectVersionsOptions = field(
        default_factory=DetectVersionsOptions
    )


options = Options()
