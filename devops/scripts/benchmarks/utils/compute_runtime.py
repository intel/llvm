# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import yaml

from .logger import log
from .utils import *
from options import options
from git_project import GitProject


def replace_in_file(file_path: Path, search_pattern: str, replacement: str) -> None:
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
            os.path.join(self.gmmlib, "lib"),
            os.path.join(self.level_zero, "lib"),
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

    def build_gmmlib(self, repo, commit) -> tuple[Path, bool]:
        log.info("Building GMMLib...")
        project = GitProject(repo, commit, Path(options.workdir), "gmmlib")
        rebuilt = False
        if project.needs_rebuild():
            project.configure()
            project.build()
            project.install()
            rebuilt = True
            log.info("GMMLib build complete.")
        else:
            log.info("GMMLib build skipped, already built.")
        return project.install_dir, rebuilt

    def build_level_zero(self, repo, commit) -> tuple[Path, bool]:
        log.info("Building Level Zero...")
        project = GitProject(repo, commit, Path(options.workdir), "level-zero")
        cmakelists_path = project.src_dir / "CMakeLists.txt"
        # there's a bug in level-zero CMakeLists.txt that makes it install headers into incorrect location.
        replace_in_file(
            cmakelists_path, r"DESTINATION \./include/", "DESTINATION include/"
        )

        rebuilt = False
        if project.needs_rebuild():
            project.configure()
            project.build()
            project.install()
            rebuilt = True
            log.info("Level Zero build complete.")
        else:
            log.info("Level Zero build skipped, already built.")
        return project.install_dir, rebuilt

    def build_igc(self, repo, commit) -> tuple[Path, bool]:
        log.info("Building IGC...")
        igc_project = GitProject(repo, commit, Path(options.workdir), "igc")
        rebuilt = False
        if igc_project.needs_rebuild():
            # Clone igc dependencies by creating a GitProject instance for each dependency.
            # Repos with commit hashes as refs can't be cloned shallowly.
            GitProject(
                "https://github.com/intel/vc-intrinsics",
                "9d255266e1df8f1dc5d11e1fbb03213acfaa4fc7",
                Path(options.workdir),
                "vc-intrinsics",
                no_suffix_src=True,
                shallow_clone=False,
            )
            llvm_project = GitProject(
                "https://github.com/llvm/llvm-project",
                "llvmorg-15.0.7",
                Path(options.workdir),
                "llvm-project",
                no_suffix_src=True,
            )
            llvm_projects = llvm_project.src_dir / "llvm" / "projects"
            GitProject(
                "https://github.com/intel/opencl-clang",
                "ocl-open-150",
                llvm_projects,
                "opencl-clang",
                no_suffix_src=True,
            )
            GitProject(
                "https://github.com/KhronosGroup/SPIRV-LLVM-Translator",
                "llvm_release_150",
                llvm_projects,
                "llvm-spirv",
                no_suffix_src=True,
            )
            GitProject(
                "https://github.com/KhronosGroup/SPIRV-Tools.git",
                "f289d047f49fb60488301ec62bafab85573668cc",
                Path(options.workdir),
                "SPIRV-Tools",
                no_suffix_src=True,
                shallow_clone=False,
            )
            GitProject(
                "https://github.com/KhronosGroup/SPIRV-Headers.git",
                "0e710677989b4326ac974fd80c5308191ed80965",
                Path(options.workdir),
                "SPIRV-Headers",
                no_suffix_src=True,
                shallow_clone=False,
            )

            configure_args = [
                "-DCMAKE_C_FLAGS=-Wno-error",
                "-DCMAKE_CXX_FLAGS=-Wno-error",
            ]
            igc_project.configure(extra_args=configure_args)
            # set timeout to 2h. IGC takes A LONG time to build if building from scratch.
            igc_project.build(timeout=60 * 60 * 2)
            # cmake --install doesn't work...
            run("make install", cwd=igc_project.build_dir)
            rebuilt = True
            log.info("IGC build complete.")
        else:
            log.info("IGC build skipped, already built.")
        return igc_project.install_dir, rebuilt

    def read_manifest(self, manifest_path: Path) -> dict:
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
        project = GitProject(
            "https://github.com/intel/compute-runtime.git",
            options.compute_runtime_tag,
            Path(options.workdir),
            "compute-runtime",
            use_installdir=False,
        )

        manifest_path = project.src_dir / "manifests" / "manifest.yml"
        manifest = self.read_manifest(manifest_path)

        level_zero_repo, level_zero_commit = self.get_repo_info(manifest, "level_zero")
        self.level_zero, self.level_zero_rebuilt = self.build_level_zero(
            level_zero_repo, level_zero_commit
        )

        gmmlib_repo, gmmlib_commit = self.get_repo_info(manifest, "gmmlib")
        self.gmmlib, self.gmmlib_rebuilt = self.build_gmmlib(gmmlib_repo, gmmlib_commit)

        if options.build_igc:
            igc_repo, igc_commit = self.get_repo_info(manifest, "igc")
            self.igc, self.igc_rebuilt = self.build_igc(igc_repo, igc_commit)

        if (
            project.needs_rebuild()
            or self.level_zero_rebuilt
            or self.gmmlib_rebuilt
            or (options.build_igc and self.igc_rebuilt)
        ):
            cmakelists_path = (
                project.src_dir / "level_zero" / "cmake" / "FindLevelZero.cmake"
            )
            # specifying custom L0 is problematic...
            replace_in_file(
                cmakelists_path, r"(\$\{LEVEL_ZERO_ROOT\}\s*)", r"\1NO_DEFAULT_PATH\n"
            )

            cmakelists_path = project.src_dir / "CMakeLists.txt"
            # Remove -Werror...
            replace_in_file(cmakelists_path, r"\s-Werror(?:=[a-zA-Z]*)?", "")

            log.info("Building Compute Runtime...")
            extra_config_args = [
                "-DNEO_ENABLE_i915_PRELIM_DETECTION=1",
                "-DNEO_ENABLE_I915_PRELIM_DETECTION=1",
                "-DNEO_SKIP_UNIT_TESTS=1",
                f"-DGMM_DIR={self.gmmlib}",
                f"-DLEVEL_ZERO_ROOT={self.level_zero}",
            ]
            if options.build_igc:
                extra_config_args.append(f"-DIGC_DIR={self.igc}")

            project.configure(extra_args=extra_config_args)
            project.build()
            log.info("Compute Runtime build complete.")
        else:
            log.info("Compute Runtime build skipped, already built.")
        return project.build_dir


def get_compute_runtime() -> ComputeRuntime:  # ComputeRuntime singleton
    if not hasattr(get_compute_runtime, "instance"):
        get_compute_runtime.instance = ComputeRuntime()
    return get_compute_runtime.instance
