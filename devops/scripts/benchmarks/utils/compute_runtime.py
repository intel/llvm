# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import yaml

from pathlib import Path
from .utils import *
from options import options


def replace_in_file(file_path, search_pattern, replacement):
    with open(file_path, "r") as file:
        content = file.read()

    modified_content = re.sub(search_pattern, replacement, content)

    with open(file_path, "w") as file:
        file.write(modified_content)


class ComputeRuntime:
    def __init__(self):
        self.compute_runtime = self.build_compute_runtime()

        return

    def ld_libraries(self) -> list[str]:
        paths = [
            os.path.join(self.gmmlib, "lib64"),
            os.path.join(self.level_zero, "lib64"),
            os.path.join(self.compute_runtime, "bin"),
        ]

        if options.build_igc:
            paths.append(os.path.join(self.igc, "lib"))

        return paths

    def env_vars(self) -> dict:
        return {
            "ZE_ENABLE_ALT_DRIVERS": os.path.join(
                self.compute_runtime, "bin", "libze_intel_gpu.so"
            ),
            "OCL_ICD_FILENAMES": os.path.join(
                self.compute_runtime, "bin", "libigdrcl.so"
            ),
        }

    def build_gmmlib(self, repo, commit):
        self.gmmlib_repo = git_clone(options.workdir, "gmmlib-repo", repo, commit)
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
        run(f"cmake --build {self.gmmlib_build} -j {options.build_jobs}")
        run(f"cmake --install {self.gmmlib_build}")
        return self.gmmlib_install

    def build_level_zero(self, repo, commit):
        self.level_zero_repo = git_clone(
            options.workdir, "level-zero-repo", repo, commit
        )
        self.level_zero_build = os.path.join(options.workdir, "level-zero-build")
        self.level_zero_install = os.path.join(options.workdir, "level-zero-install")

        cmakelists_path = os.path.join(self.level_zero_repo, "CMakeLists.txt")
        # there's a bug in level-zero CMakeLists.txt that makes it install headers into incorrect location.
        replace_in_file(
            cmakelists_path, r"DESTINATION \./include/", "DESTINATION include/"
        )

        configure_command = [
            "cmake",
            f"-B {self.level_zero_build}",
            f"-S {self.level_zero_repo}",
            f"-DCMAKE_INSTALL_PREFIX={self.level_zero_install}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        run(configure_command)
        run(f"cmake --build {self.level_zero_build} -j {options.build_jobs}")
        run(f"cmake --install {self.level_zero_build}")
        return self.level_zero_install

    def build_igc(self, repo, commit):
        self.igc_repo = git_clone(options.workdir, "igc", repo, commit)
        self.vc_intr = git_clone(
            options.workdir,
            "vc-intrinsics",
            "https://github.com/intel/vc-intrinsics",
            "9d255266e1df8f1dc5d11e1fbb03213acfaa4fc7",
        )
        self.llvm_project = git_clone(
            options.workdir,
            "llvm-project",
            "https://github.com/llvm/llvm-project",
            "llvmorg-15.0.7",
        )
        llvm_projects = os.path.join(self.llvm_project, "llvm", "projects")
        self.ocl = git_clone(
            llvm_projects,
            "opencl-clang",
            "https://github.com/intel/opencl-clang",
            "ocl-open-150",
        )
        self.translator = git_clone(
            llvm_projects,
            "llvm-spirv",
            "https://github.com/KhronosGroup/SPIRV-LLVM-Translator",
            "llvm_release_150",
        )
        self.spirv_tools = git_clone(
            options.workdir,
            "SPIRV-Tools",
            "https://github.com/KhronosGroup/SPIRV-Tools.git",
            "f289d047f49fb60488301ec62bafab85573668cc",
        )
        self.spirv_headers = git_clone(
            options.workdir,
            "SPIRV-Headers",
            "https://github.com/KhronosGroup/SPIRV-Headers.git",
            "0e710677989b4326ac974fd80c5308191ed80965",
        )

        self.igc_build = os.path.join(options.workdir, "igc-build")
        self.igc_install = os.path.join(options.workdir, "igc-install")
        configure_command = [
            "cmake",
            "-DCMAKE_C_FLAGS=-Wno-error",
            "-DCMAKE_CXX_FLAGS=-Wno-error",
            f"-B {self.igc_build}",
            f"-S {self.igc_repo}",
            f"-DCMAKE_INSTALL_PREFIX={self.igc_install}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        run(configure_command)

        # set timeout to 2h. IGC takes A LONG time to build if building from scratch.
        run(
            f"cmake --build {self.igc_build} -j {options.build_jobs}",
            timeout=60 * 60 * 2,
        )
        # cmake --install doesn't work...
        run("make install", cwd=self.igc_build)
        return self.igc_install

    def read_manifest(self, manifest_path):
        with open(manifest_path, "r") as file:
            manifest = yaml.safe_load(file)
        return manifest

    def get_repo_info(self, manifest, component_name):
        component = manifest["components"].get(component_name)
        if component:
            repo = component.get("repository")
            revision = component.get("revision")
            return repo, revision
        return None, None

    def build_compute_runtime(self):
        self.compute_runtime_repo = git_clone(
            options.workdir,
            "compute-runtime-repo",
            "https://github.com/intel/compute-runtime.git",
            options.compute_runtime_tag,
        )
        self.compute_runtime_build = os.path.join(
            options.workdir, "compute-runtime-build"
        )

        manifest_path = os.path.join(
            self.compute_runtime_repo, "manifests", "manifest.yml"
        )
        manifest = self.read_manifest(manifest_path)

        level_zero_repo, level_zero_commit = self.get_repo_info(manifest, "level_zero")
        self.level_zero = self.build_level_zero(level_zero_repo, level_zero_commit)

        gmmlib_repo, gmmlib_commit = self.get_repo_info(manifest, "gmmlib")
        self.gmmlib = self.build_gmmlib(gmmlib_repo, gmmlib_commit)

        if options.build_igc:
            igc_repo, igc_commit = self.get_repo_info(manifest, "igc")
            self.igc = self.build_igc(igc_repo, igc_commit)

        cmakelists_path = os.path.join(
            self.compute_runtime_repo, "level_zero", "cmake", "FindLevelZero.cmake"
        )
        # specifying custom L0 is problematic...
        replace_in_file(
            cmakelists_path, r"(\$\{LEVEL_ZERO_ROOT\}\s*)", r"\1NO_DEFAULT_PATH\n"
        )

        cmakelists_path = os.path.join(self.compute_runtime_repo, "CMakeLists.txt")
        # Remove -Werror...
        replace_in_file(cmakelists_path, r"\s-Werror(?:=[a-zA-Z]*)?", "")

        configure_command = [
            "cmake",
            f"-B {self.compute_runtime_build}",
            f"-S {self.compute_runtime_repo}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DNEO_ENABLE_i915_PRELIM_DETECTION=1",
            "-DNEO_ENABLE_I915_PRELIM_DETECTION=1",
            "-DNEO_SKIP_UNIT_TESTS=1",
            f"-DGMM_DIR={self.gmmlib}",
            f"-DLEVEL_ZERO_ROOT={self.level_zero}",
        ]
        if options.build_igc:
            configure_command.append(f"-DIGC_DIR={self.igc}")

        run(configure_command)
        run(f"cmake --build {self.compute_runtime_build} -j {options.build_jobs}")
        return self.compute_runtime_build


def get_compute_runtime() -> ComputeRuntime:  # ComputeRuntime singleton
    if not hasattr(get_compute_runtime, "instance"):
        get_compute_runtime.instance = ComputeRuntime()
    return get_compute_runtime.instance
