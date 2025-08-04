# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import shutil

from utils.logger import log
from utils.utils import run
from options import options


class GitProject:
    def __init__(
        self,
        url: str,
        ref: str,
        directory: Path,
        name: str,
        force_rebuild: bool = False,
        no_suffix_src: bool = False,
    ) -> None:
        self._url = url
        self._ref = ref
        self._directory = directory
        self._name = name
        self._force_rebuild = force_rebuild
        self._no_suffix_src = no_suffix_src
        self._rebuild_needed = self._git_clone()

    @property
    def src_dir(self) -> Path:
        suffix = "" if self._no_suffix_src else "-src"
        return self._directory / f"{self._name}{suffix}"

    @property
    def build_dir(self) -> Path:
        return self._directory / f"{self._name}-build"

    @property
    def install_dir(self) -> Path:
        return self._directory / f"{self._name}-install"

    def needs_rebuild(self, check_build=False, check_install=False) -> bool:
        """Checks if the project needs to be rebuilt.

        Args:
            check_build (bool): If True, checks if the build directory exists and has some files.
            check_install (bool): If True, checks if the install directory exists and has some files.

        Returns:
            bool: True if the project needs to be rebuilt, False otherwise.
        """
        log.debug(f"Checking if project {self._name} needs rebuild.")
        if self._force_rebuild:
            log.debug(
                f"Force rebuild is enabled for project {self._name}, rebuild needed."
            )
            if Path(self.build_dir).exists():
                shutil.rmtree(self.build_dir)
            return True
        elif self._rebuild_needed:
            return True
        if check_build:
            if self.build_dir.exists() and any(
                path.is_file() for path in self.build_dir.glob("**/*")
            ):
                log.debug(
                    f"Build directory {self.build_dir} exists and is not empty, no rebuild needed."
                )
            else:
                log.debug(
                    f"Build directory {self.build_dir} does not exist or does not contain any file, rebuild needed."
                )
                return True
        if check_install:
            if self.install_dir.exists() and any(
                path.is_file() for path in self.install_dir.glob("**/*")
            ):
                log.debug(
                    f"Install directory {self.install_dir} exists and is not empty, no rebuild needed."
                )
            else:
                log.debug(
                    f"Install directory {self.install_dir} does not exist or does not contain any file, rebuild needed."
                )
                return True
        return False

    def configure(
        self,
        extra_args: list | None = None,
        install_prefix=True,
        add_sycl: bool = False,
    ) -> None:
        """Configures the project."""
        cmd = [
            "cmake",
            f"-S {self.src_dir}",
            f"-B {self.build_dir}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        if install_prefix:
            cmd.append(f"-DCMAKE_INSTALL_PREFIX={self.install_dir}")
        if extra_args:
            cmd.extend(extra_args)

        run(cmd, add_sycl=add_sycl)

    def build(
        self,
        target: str = "",
        add_sycl: bool = False,
        ld_library: list = [],
        timeout: int | None = None,
    ) -> None:
        """Builds the project."""
        target_arg = f"--target {target}" if target else ""
        run(
            f"cmake --build {self.build_dir} {target_arg} -j {options.build_jobs}",
            add_sycl=add_sycl,
            ld_library=ld_library,
            timeout=timeout,
        )

    def install(self) -> None:
        """Installs the project."""
        run(f"cmake --install {self.build_dir}")

    def _git_clone(self) -> bool:
        """Clone a git repository into a specified directory at a specific commit.
        Returns:
            bool: True if the repository was cloned or updated, False if it was already up-to-date.
        """
        log.debug(f"Cloning {self._url} into {self.src_dir} at commit {self._ref}")
        if self.src_dir.exists() and Path(self.src_dir, ".git").exists():
            log.debug(
                f"Repository {self._url} already exists at {self.src_dir}, checking for updates."
            )
            run("git fetch", cwd=self.src_dir)
            target_commit = (
                run(f"git rev-parse {self._ref}", cwd=self.src_dir)
                .stdout.decode()
                .strip()
            )
            current_commit = (
                run("git rev-parse HEAD", cwd=self.src_dir).stdout.decode().strip()
            )
            if current_commit != target_commit:
                log.debug(
                    f"Current commit {current_commit} does not match target {target_commit}, checking out {self._ref}."
                )
                run("git reset --hard", cwd=self.src_dir)
                run(f"git checkout {self._ref}", cwd=self.src_dir)
            else:
                log.debug(
                    f"Current commit {current_commit} matches target {target_commit}, no update needed."
                )
                return False
        elif not self.src_dir.exists():
            run(f"git clone --recursive {self._url} {self.src_dir}")
            run(f"git checkout {self._ref}", cwd=self.src_dir)
        else:
            raise Exception(
                f"The directory {self.src_dir} exists but is not a git repository."
            )
        log.debug(f"Cloned {self._url} into {self.src_dir} at commit {self._ref}")
        return True
