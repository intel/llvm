# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
from pathlib import Path
from .result import Result
from .options import options
from utils.utils import run
import urllib.request
import tarfile

class Benchmark:
    def __init__(self, directory):
        self.directory = directory

    @staticmethod
    def get_adapter_full_path():
        for libs_dir_name in ['lib', 'lib64']:
            adapter_path = os.path.join(
                options.ur, libs_dir_name, f"libur_adapter_{options.ur_adapter}.so")
            if os.path.isfile(adapter_path):
                return adapter_path
        assert False, \
            f"could not find adapter file {adapter_path} (and in similar lib paths)"

    def run_bench(self, command, env_vars):
        env_vars_with_forced_adapter = env_vars.copy()
        if options.ur is not None:
            env_vars_with_forced_adapter.update(
                {'UR_ADAPTERS_FORCE_LOAD': Benchmark.get_adapter_full_path()})

        return run(
            command=command,
            env_vars=env_vars_with_forced_adapter,
            add_sycl=True,
            cwd=options.benchmark_cwd
        ).stdout.decode()

    def create_data_path(self, name):
        data_path = os.path.join(self.directory, "data", name)

        if options.rebuild and Path(data_path).exists():
           shutil.rmtree(data_path)

        Path(data_path).mkdir(parents=True, exist_ok=True)

        return data_path

    def download_untar(self, name, url, file):
        self.data_path = self.create_data_path(name)
        data_file = os.path.join(self.data_path, file)
        if not Path(data_file).exists():
            print(f"{data_file} does not exist, downloading")
            urllib.request.urlretrieve(url, data_file)
            file = tarfile.open(data_file)
            file.extractall(self.data_path)
            file.close()
        else:
            print(f"{data_file} exists, skipping...")

    def name(self):
        raise NotImplementedError()

    def unit(self):
        raise NotImplementedError()

    def lower_is_better(self):
        return True

    def setup(self):
        raise NotImplementedError()

    def run(self, env_vars) -> list[Result]:
        raise NotImplementedError()

    def teardown(self):
        raise NotImplementedError()

class Suite:
    def benchmarks(self) -> list[Benchmark]:
        raise NotImplementedError()

    def setup(self):
        return
