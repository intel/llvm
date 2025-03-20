# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import os
import shutil
from pathlib import Path
from utils.result import BenchmarkMetadata, BenchmarkTag, Result
from options import options
from utils.utils import download, run

benchmark_tags = [BenchmarkTag('sycl', 'Benchmark uses SYCL RT'),
                  BenchmarkTag('ur', 'Benchmark uses Unified Runtime'),
                  BenchmarkTag('L0', 'Benchmark uses L0 directly'),
                  BenchmarkTag('umf', 'Benchmark uses UMF directly'),
                  BenchmarkTag('micro', 'Microbenchmark focusing on a specific niche'),
                  BenchmarkTag('application', 'Real application-based performance test'),
                  BenchmarkTag('proxy', 'Benchmark that tries to implement a real application use-case'),
                  BenchmarkTag('submit', 'Benchmark tests the kernel submit path'),
                  BenchmarkTag('math', 'Benchmark tests math compute performance'),
                  BenchmarkTag('memory', 'Benchmark tests memory transfer performance'),
                  BenchmarkTag('allocation', 'Benchmark tests memory allocation performance'),
                  BenchmarkTag('graph', 'Benchmark tests graph performance'),]

def translate_tags(tag_names: list[str]) -> list[BenchmarkTag]:
    tags = []
    for tag_name in tag_names:
        for tag in benchmark_tags:
            if tag.name == tag_name:
                tags.append(tag)
                break

    return tags

class Benchmark:
    def __init__(self, directory, suite):
        self.directory = directory
        self.suite = suite

    @staticmethod
    def get_adapter_full_path():
        for libs_dir_name in ["lib", "lib64"]:
            adapter_path = os.path.join(
                options.ur, libs_dir_name, f"libur_adapter_{options.ur_adapter}.so"
            )
            if os.path.isfile(adapter_path):
                return adapter_path
        assert (
            False
        ), f"could not find adapter file {adapter_path} (and in similar lib paths)"

    def run_bench(self, command, env_vars, ld_library=[], add_sycl=True):
        env_vars = env_vars.copy()
        if options.ur is not None:
            env_vars.update(
                {"UR_ADAPTERS_FORCE_LOAD": Benchmark.get_adapter_full_path()}
            )

        env_vars.update(options.extra_env_vars)

        ld_libraries = options.extra_ld_libraries.copy()
        ld_libraries.extend(ld_library)

        return run(
            command=command,
            env_vars=env_vars,
            add_sycl=add_sycl,
            cwd=options.benchmark_cwd,
            ld_library=ld_libraries,
        ).stdout.decode()

    def create_data_path(self, name, skip_data_dir=False):
        if skip_data_dir:
            data_path = os.path.join(self.directory, name)
        else:
            data_path = os.path.join(self.directory, "data", name)
            if options.redownload and Path(data_path).exists():
                shutil.rmtree(data_path)

        Path(data_path).mkdir(parents=True, exist_ok=True)

        return data_path

    def download(
        self,
        name,
        url,
        file,
        untar=False,
        unzip=False,
        skip_data_dir=False,
        checksum="",
    ):
        self.data_path = self.create_data_path(name, skip_data_dir)
        return download(self.data_path, url, file, untar, unzip, checksum)

    def lower_is_better(self):
        return True

    def setup(self):
        raise NotImplementedError()

    def run(self, env_vars) -> list[Result]:
        raise NotImplementedError()

    def teardown(self):
        raise NotImplementedError()

    def stddev_threshold(self):
        return None

    def get_suite_name(self) -> str:
        return self.suite.name()

    def name(self):
        raise NotImplementedError()

    def description(self):
        return "No description provided."

    def notes(self) -> str:
        return None

    def unstable(self) -> str:
        return None

    def get_tags(self) -> list[str]:
        return []

    def get_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            type="benchmark",
            description=self.description(),
            notes=self.notes(),
            unstable=self.unstable(),
            tags=translate_tags(self.get_tags())
        )

class Suite:
    def benchmarks(self) -> list[Benchmark]:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def setup(self):
        return

    def additionalMetadata(self) -> dict[str, BenchmarkMetadata]:
        return {}
