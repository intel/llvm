# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

from pathlib import Path
from .utils import *
from options import options

def replace_in_file(file_path, search_pattern, replacement):
    with open(file_path, 'r') as file:
        content = file.read()
    
    modified_content = re.sub(search_pattern, replacement, content)
    
    with open(file_path, 'w') as file:
        file.write(modified_content)

class ComputeRuntime:
    def __init__(self):
        self.gmmlib = self.build_gmmlib()
        self.level_zero = self.build_level_zero()
        self.compute_runtime = self.build_compute_runtime(self.gmmlib, self.level_zero)

        return

    def ld_libraries(self) -> list[str]:
        return [
            os.path.join(self.gmmlib, "lib64"),
            os.path.join(self.level_zero, "lib64"),
            os.path.join(self.compute_runtime, "bin"),
        ]

    def env_vars(self) -> dict:
        return {"ZE_ENABLE_ALT_DRIVERS" : os.path.join(self.compute_runtime, "bin", "libze_intel_gpu.so"),
                "OCL_ICD_FILENAMES" : os.path.join(self.compute_runtime, "bin", "libigdrcl.so")}

    def build_gmmlib(self):
        self.gmmlib_repo = git_clone(options.workdir, "gmmlib-repo", "https://github.com/intel/gmmlib.git", "9104c2090158b35d440afdf8ec940d89cc7b3c6a")
        self.gmmlib_build = os.path.join(options.workdir, "gmmlib-build")
        self.gmmlib_install = os.path.join(options.workdir, "gmmlib-install")
        configure_command = [
            "cmake",
            f"-B {self.gmmlib_build}",
            f"-S {self.gmmlib_repo}",
            f"-DCMAKE_INSTALL_PREFIX={self.gmmlib_install}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        run(configure_command)
        run(f"cmake --build {self.gmmlib_build} -j")
        run(f"cmake --install {self.gmmlib_build}")
        return self.gmmlib_install

    def build_level_zero(self):
        self.level_zero_repo = git_clone(options.workdir, "level-zero-repo", "https://github.com/oneapi-src/level-zero.git", "3969f34c16a843b943b948f8fe7081ef87deb369")
        self.level_zero_build = os.path.join(options.workdir, "level-zero-build")
        self.level_zero_install = os.path.join(options.workdir, "level-zero-install")

        cmakelists_path = os.path.join(self.level_zero_repo, "CMakeLists.txt")
        # there's a bug in level-zero CMakeLists.txt that makes it install headers into incorrect location.
        replace_in_file(cmakelists_path, r'DESTINATION \./include/', 'DESTINATION include/')

        configure_command = [
            "cmake",
            f"-B {self.level_zero_build}",
            f"-S {self.level_zero_repo}",
            f"-DCMAKE_INSTALL_PREFIX={self.level_zero_install}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        run(configure_command)
        run(f"cmake --build {self.level_zero_build} -j")
        run(f"cmake --install {self.level_zero_build}")
        return self.level_zero_install

    def build_compute_runtime(self, gmmlib, level_zero):
        self.compute_runtime_repo = git_clone(options.workdir, "compute-runtime-repo", "https://github.com/intel/compute-runtime.git", options.compute_runtime_tag)
        self.compute_runtime_build = os.path.join(options.workdir, "compute-runtime-build")

        cmakelists_path = os.path.join(self.compute_runtime_repo, "level_zero", "cmake", "FindLevelZero.cmake")
        # specifying custom L0 is problematic...
        replace_in_file(cmakelists_path, r'(\$\{LEVEL_ZERO_ROOT\}\s*)', r'\1NO_DEFAULT_PATH\n')

        cmakelists_path = os.path.join(self.compute_runtime_repo, "CMakeLists.txt")
        # Remove -Werror...
        replace_in_file(cmakelists_path, r'\s-Werror(?:=[a-zA-Z]*)?', '')

        configure_command = [
            "cmake",
            f"-B {self.compute_runtime_build}",
            f"-S {self.compute_runtime_repo}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DNEO_ENABLE_i915_PRELIM_DETECTION=1",
            "-DNEO_ENABLE_I915_PRELIM_DETECTION=1",
            "-DNEO_SKIP_UNIT_TESTS=1",
            f"-DGMM_DIR={gmmlib}",
            f"-DLEVEL_ZERO_ROOT={level_zero}"
        ]
        run(configure_command)
        run(f"cmake --build {self.compute_runtime_build} -j")
        return self.compute_runtime_build

def get_compute_runtime() -> ComputeRuntime: # ComputeRuntime singleton
    if not hasattr(get_compute_runtime, "instance"):
        get_compute_runtime.instance = ComputeRuntime()
    return get_compute_runtime.instance
