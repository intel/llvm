# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from pathlib import Path

from utils.utils import git_clone, run

class GitProject(ABC):
    def __init__(self, directory: Path, src_dir: Path, build_dir: Path, install_dir: Path):
        self.directory = directory
        self.src_dir = src_dir
        self.build_dir = build_dir
        self.install_dir = install_dir
        self.git_url = git_url
        self.git_hash = git_hash

    @abstractmethod
    def setup(self) -> None:
        """Sets up the project environment: checks if the project needs to be cloned
        or updated, and prepares the build.
        """
        repo_updated = git_clone(self.src_dir, self.git_url, self.git_hash)
        repo_installed = self.check_install()
        if repo_updated or not repo_installed:
            self.rebuild()

    def rebuild(self) -> None:
        """Rebuilds the project."""
        self.configure()
        self.build()
        self.install()

    def configure(self) -> None:
        """Configures the project."""
        run(f"cmake -S {self.src_dir} -B build", cwd=self.src_dir)

    def build(self) -> None:
        """Builds the project."""
        run(f"cmake --build {self.build_dir}", cwd=self.src_dir)

    def install(self) -> None:
        """Installs the project."""
        run(f"cmake --install {self.install_dir}", cwd=self.src_dir)

    def check_install(self) -> bool:
        """Checks if the project is already installed
        by searching for files in the install directory.
        """
        return self.install_dir.exists() and any(item.is_file() for item in self.install_dir.iterdir())
