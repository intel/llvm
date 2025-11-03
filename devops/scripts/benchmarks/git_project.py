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
        use_installdir: bool = True,
        no_suffix_src: bool = False,
        shallow_clone: bool = True,
    ) -> None:
        self._url = url
        self._ref = ref
        self._directory = directory
        self._name = name
        self._use_installdir = use_installdir
        self._no_suffix_src = no_suffix_src
        self._shallow_clone = shallow_clone
        self._rebuild_needed = self._setup_repo()

    @property
    def name(self):
        return self._name

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

    def needs_rebuild(self) -> bool:
        if self._rebuild_needed:
            log.debug(
                f"Rebuild needed because new sources were detected for project {self._name}."
            )
            return True

        dir_to_check = self.install_dir if self._use_installdir else self.build_dir

        if not (
            dir_to_check.exists()
            and any(path.is_file() for path in dir_to_check.glob("**/*"))
        ):
            log.debug(
                f"{dir_to_check} does not exist or does not contain any file, rebuild needed."
            )
            return True
        log.debug(f"{dir_to_check} exists and is not empty, no rebuild needed.")
        return False

    def configure(
        self,
        extra_args: list | None = None,
        add_sycl: bool = False,
    ) -> None:
        """Configures the project."""
        cmd = [
            "cmake",
            f"-S {self.src_dir}",
            f"-B {self.build_dir}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        if self._use_installdir:
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

    def cherry_pick(self, commit_hash: str) -> None:
        """Cherry-pick a specific commit."""
        try:
            log.debug(f"Cherry-picking commit {commit_hash} in {self.src_dir}")
            run(f"git cherry-pick {commit_hash}", cwd=self.src_dir)
            log.debug(f"Successfully cherry-picked commit {commit_hash}")
        except Exception as e:
            log.error(f"Failed to cherry-pick commit {commit_hash}: {e}")
            raise

    def _can_shallow_clone_ref(self, ref: str) -> bool:
        """Check if we can do a shallow clone with this ref using git ls-remote."""
        try:
            result = run(f"git ls-remote --heads --tags {self._url} {ref}")
            output = result.stdout.decode().strip()

            if output:
                # Found the ref as a branch or tag
                log.debug(
                    f"Ref {ref} found as branch/tag via ls-remote, can shallow clone"
                )
                return True
            else:
                # Not found as branch/tag, likely a SHA commit
                log.debug(
                    f"Ref {ref} not found as branch/tag via ls-remote, likely SHA commit"
                )
                return False
        except Exception as e:
            log.debug(
                f"Could not check ref {ref} via ls-remote: {e}, assuming SHA commit"
            )
            return False

    def _git_clone(self) -> None:
        """Clone the git repository."""
        try:
            log.debug(f"Cloning {self._url} into {self.src_dir} at commit {self._ref}")
            git_clone_cmd = f"git clone --recursive {self._url} {self.src_dir}"
            if self._shallow_clone:
                if self._can_shallow_clone_ref(self._ref):
                    # Shallow clone for branches and tags only
                    git_clone_cmd = f"git clone --recursive --depth 1 --branch {self._ref} {self._url} {self.src_dir}"
                else:
                    log.debug(f"Cannot shallow clone SHA {self._ref}, using full clone")

            run(git_clone_cmd)
            run(f"git checkout {self._ref}", cwd=self.src_dir)
            log.debug(f"Cloned {self._url} into {self.src_dir} at commit {self._ref}")
        except Exception as e:
            log.error(f"Failed to clone repository {self._url}: {e}")
            raise

    def _git_fetch(self) -> None:
        """Fetch the latest changes from the remote repository."""
        try:
            log.debug(f"Fetching latest changes for {self._url} in {self.src_dir}")
            run("git fetch", cwd=self.src_dir)
            run("git reset --hard", cwd=self.src_dir)
            run(f"git checkout {self._ref}", cwd=self.src_dir)
            log.debug(f"Fetched latest changes for {self._url} in {self.src_dir}")
        except Exception as e:
            log.error(f"Failed to fetch updates for repository {self._url}: {e}")
            raise

    def _setup_repo(self) -> bool:
        """Clone a git repository into a specified directory at a specific commit.
        Returns:
            bool: True if the repository was cloned or updated, False if it was already up-to-date.
        """
        if not self.src_dir.exists():
            self._git_clone()
            return True
        elif Path(self.src_dir, ".git").exists():
            log.debug(
                f"Repository {self._url} already exists at {self.src_dir}, checking for updates."
            )
            current_commit = (
                run("git rev-parse HEAD^{commit}", cwd=self.src_dir)
                .stdout.decode()
                .strip()
            )
            try:
                target_commit = (
                    run(f"git rev-parse {self._ref}^{{commit}}", cwd=self.src_dir)
                    .stdout.decode()
                    .strip()
                )
                if current_commit != target_commit:
                    log.debug(
                        f"Current commit {current_commit} does not match target {target_commit}, checking out {self._ref}."
                    )
                    run("git reset --hard", cwd=self.src_dir)
                    run(f"git checkout {self._ref}", cwd=self.src_dir)
                    return True
            except Exception:
                log.error(
                    f"Failed to resolve target commit {self._ref}. Fetching updates."
                )
                if self._shallow_clone:
                    log.debug(f"Cloning a clean shallow copy.")
                    shutil.rmtree(self.src_dir)
                    self._git_clone()
                    return True
                else:
                    self._git_fetch()
                    return True
            else:
                log.debug(
                    f"Current commit {current_commit} matches target {target_commit}, no update needed."
                )
                return False
        else:
            raise Exception(
                f"The directory {self.src_dir} exists but is not a git repository."
            )
