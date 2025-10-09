# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Metadata generator for Compute Benchmarks.

This module provides centralized metadata generation for Compute Benchmark groups,
ensuring consistency between benchmark group membership and group metadata definitions.
"""

from collections import namedtuple
from typing import Dict, List

from utils.result import BenchmarkMetadata

from .base import Benchmark


def string_consts(cls):
    """Decorator to convert string-annotated class attributes to string constants."""
    for key, value in cls.__annotations__.items():
        if value is str:
            setattr(cls, key, key)
    return cls


@string_consts
class Tags:
    """String constants for benchmark tags to prevent typos."""

    submit: str
    micro: str
    SYCL: str
    UR: str
    L0: str
    graph: str
    memory: str
    proxy: str
    finalize: str


BaseGroupMetadata = namedtuple(
    "BaseGroupMetadata",
    [
        "description",
        "notes",
        "unstable",
        "tags",
        "range_min",
        "range_max",
    ],
    defaults=(None, None, None, [], None, None),
)


class ComputeMetadataGenerator:
    """
    Generates metadata for Compute Benchmark groups.

    This class keeps the logic for creating group metadata, ensuring that
    all possible benchmark group configurations have corresponding metadata entries.
    """

    def __init__(self):
        # Base metadata for core groups
        self._base_group_metadata = {
            "SubmitKernel": BaseGroupMetadata(
                description="Measures CPU time overhead of submitting kernels through different APIs.",
                notes=(
                    "Each layer builds on top of the previous layer, adding functionality and overhead.\n"
                    "The first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\n"
                    "The UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\n"
                    "Work is ongoing to reduce the overhead of the SYCL API\n"
                ),
                tags=[Tags.submit, Tags.micro, Tags.SYCL, Tags.UR, Tags.L0],
                range_min=0.0,
            ),
            "SinKernelGraph": BaseGroupMetadata(
                unstable="This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
                tags=[
                    Tags.submit,
                    Tags.memory,
                    Tags.proxy,
                    Tags.SYCL,
                    Tags.UR,
                    Tags.L0,
                    Tags.graph,
                ],
            ),
            "SubmitGraph": BaseGroupMetadata(
                tags=[Tags.submit, Tags.micro, Tags.SYCL, Tags.UR, Tags.L0, Tags.graph]
            ),
            "FinalizeGraph": BaseGroupMetadata(
                tags=[Tags.finalize, Tags.micro, Tags.SYCL, Tags.graph]
            ),
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
            group_name = benchmark.explicit_group()
            if group_name and group_name not in metadata:
                metadata[group_name] = self._generate_metadata(group_name)

        return metadata

    def _generate_metadata(self, group_name: str) -> BenchmarkMetadata:
        """
        Generate metadata for a specific benchmark group.

        Args:
            group_name: Name of the benchmark group

        Returns:
            BenchmarkMetadata: Metadata object describing the specified benchmark group.
        """
        base_group_name = self._extract_base_group_name(group_name)
        base_metadata = self._base_group_metadata.get(
            base_group_name, BaseGroupMetadata()
        )
        return BenchmarkMetadata(
            type="group",
            description=base_metadata.description,
            notes=base_metadata.notes,
            unstable=base_metadata.unstable,
            tags=base_metadata.tags,
            range_min=base_metadata.range_min,
            range_max=base_metadata.range_max,
            explicit_group=group_name,
        )

    def _extract_base_group_name(self, group_name: str) -> str:
        """
        Extracts the base group name from a group name string.
        Assumes group names are in the format 'BaseGroupName [Variant]'.
        If the format changes, this method should be updated accordingly.
        """
        return group_name.split()[0]
