# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
from pathlib import Path
from .result import Result
from .options import options
from utils.utils import download, run
import urllib.request
import tarfile

class Benchmark:
    def __init__(self, directory, suite):
        self.directory = directory
        self.suite = suite

    @staticmethod
    def get_adapter_full_path():
        for libs_dir_name in ['lib', 'lib64']:
            adapter_path = os.path.join(
                options.ur, libs_dir_name, f"libur_adapter_{options.ur_adapter}.so")
            if os.path.isfile(adapter_path):
                return adapter_path
        assert False, \
            f"could not find adapter file {adapter_path} (and in similar lib paths)"

    def run_bench(self, command, env_vars, ld_library=[], add_sycl=True):
        env_vars_with_forced_adapter = env_vars.copy()
        if options.ur is not None:
            env_vars_with_forced_adapter.update(
                {'UR_ADAPTERS_FORCE_LOAD': Benchmark.get_adapter_full_path()})

        return run(
            command=command,
            env_vars=env_vars_with_forced_adapter,
            add_sycl=add_sycl,
            cwd=options.benchmark_cwd,
            ld_library=ld_library
        ).stdout.decode()

    def create_data_path(self, name, skip_data_dir = False):
        if skip_data_dir:
            data_path = os.path.join(self.directory, name)
        else:
            data_path = os.path.join(self.directory, 'data', name)
            if options.rebuild and Path(data_path).exists():
                shutil.rmtree(data_path)

        Path(data_path).mkdir(parents=True, exist_ok=True)

        return data_path

    def download(self, name, url, file, untar = False, unzip = False, skip_data_dir = False):
        self.data_path = self.create_data_path(name, skip_data_dir)
        return download(self.data_path, url, file, untar, unzip)

    def name(self):
        raise NotImplementedError()

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

class Suite:
    def benchmarks(self) -> list[Benchmark]:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def setup(self):
        return
