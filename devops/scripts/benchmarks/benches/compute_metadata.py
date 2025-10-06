# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Metadata generator for Compute Benchmarks.

This module provides centralized metadata generation for Compute Benchmark groups,
ensuring consistency between benchmark group membership and group metadata definitions.
"""

from typing import Dict, List

from utils.result import BenchmarkMetadata

from .base import Benchmark


class ComputeMetadataGenerator:
    """
    Generates metadata for Compute Benchmark groups.

    This class keeps the logic for creating group metadata, ensuring that
    all possible benchmark group configurations have corresponding metadata entries.
    """

    def __init__(self):
        # Base metadata for core groups
        self._base_group_metadata = {
            "SubmitKernel": {
                "description": "Measures CPU time overhead of submitting kernels through different APIs.",
                "notes": (
                    "Each layer builds on top of the previous layer, adding functionality and overhead.\n"
                    "The first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\n"
                    "The UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\n"
                    "Work is ongoing to reduce the overhead of the SYCL API\n"
                ),
                "tags": ["submit", "micro", "SYCL", "UR", "L0"],
                "range_min": 0.0,
            },
            "SinKernelGraph": {
                "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
                "tags": ["submit", "memory", "proxy", "SYCL", "UR", "L0", "graph"],
            },
            "SubmitGraph": {"tags": ["submit", "micro", "SYCL", "UR", "L0", "graph"]},
            "FinalizeGraph": {"tags": ["finalize", "micro", "SYCL", "graph"]},
        }

    def generate_metadata_from_benchmarks(
        self, benchmarks: List[Benchmark]
    ) -> Dict[str, BenchmarkMetadata]:
        """
        Generate group metadata based on actual benchmark configurations.

        Args:
            benchmarks: List of benchmark instances to analyze

        Returns:
            Dictionary mapping group names to their metadata
        """
        metadata = {}
        # Discover all group names from actual benchmarks
        for benchmark in benchmarks:
            if hasattr(benchmark, "explicit_group") and callable(
                benchmark.explicit_group
            ):
                group_name = benchmark.explicit_group()
                if group_name:
                    self._generate_metadata(metadata, group_name)

        return metadata

    def _generate_metadata(
        self, metadata: Dict[str, BenchmarkMetadata], group_name: str
    ):
        base_metadata = self._base_group_metadata.get(group_name.split()[0], {})
        metadata[group_name] = BenchmarkMetadata(
            type="group",
            description=base_metadata.get("description"),
            notes=base_metadata.get("notes"),
            unstable=base_metadata.get("unstable"),
            tags=base_metadata.get("tags", []),
            range_min=base_metadata.get("range_min"),
            range_max=base_metadata.get("range_max"),
            explicit_group=group_name,
        )
