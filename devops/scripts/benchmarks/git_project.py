# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from pathlib import Path
import shutil

from utils.logger import log
from utils.utils import run
from options import options


# Marker file written into build_dir after a successful build. Records a
# fingerprint of the source tree so reruns can skip rebuilding.
BUILD_COMPLETE_MARKER = "benchmark_build_complete.json"


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
        src_dir_override: Path | None = None,
    ) -> None:
        self._url = url
        self._ref = ref
        self._directory = directory
        self._name = name
        self._use_installdir = use_installdir
        self._no_suffix_src = no_suffix_src
        self._shallow_clone = shallow_clone
        self._src_dir_override = src_dir_override
        self._rebuild_needed = self._setup_repo()

    @property
    def name(self):
        return self._name

    @property
    def src_dir(self) -> Path:
        if self._src_dir_override is not None:
            return self._src_dir_override
        suffix = "" if self._no_suffix_src else "-src"
        return self._directory / f"{self._name}{suffix}"

    @property
    def build_dir(self) -> Path:
        return self._directory / f"{self._name}-build"

    @property
    def install_dir(self) -> Path:
        return self._directory / f"{self._name}-install"

    @property
    def _build_marker_path(self) -> Path:
        return self.build_dir / BUILD_COMPLETE_MARKER

    def _source_fingerprint(self) -> dict | None:
        """Fingerprint the source tree for build-completion tracking.

        Returns a dict with the current commit and `git status --porcelain`
        output, or None if src_dir is not a git repository (in which case the
        caller should always rebuild).
        """
        try:
            commit = (
                run("git rev-parse HEAD", cwd=self.src_dir).stdout.decode().strip()
            )
            status = (
                run("git status --porcelain", cwd=self.src_dir).stdout.decode().strip()
            )
        except Exception as e:
            log.debug(
                f"Could not fingerprint source for {self._name} at {self.src_dir}: {e}"
            )
            return None
        return {"commit": commit, "status": status}

    def mark_build_complete(self) -> None:
        """Record a build-complete marker so reruns can skip rebuilding.

        No-op when the source is not a git repository (nothing to fingerprint).
        """
        fingerprint = self._source_fingerprint()
        if fingerprint is None:
            return
        try:
            self.build_dir.mkdir(parents=True, exist_ok=True)
            with open(self._build_marker_path, "w") as f:
                json.dump(fingerprint, f)
            log.debug(f"Wrote build-complete marker to {self._build_marker_path}")
        except Exception as e:
            log.debug(f"Failed to write build-complete marker for {self._name}: {e}")

    def _read_build_marker(self) -> dict | None:
        try:
            with open(self._build_marker_path) as f:
                return json.load(f)
        except Exception:
            return None

    def needs_rebuild(self) -> bool:
        if options.offline:
            log.debug("Rebuild is disabled due to --offline option.")
            return False

        dir_to_check = self.install_dir if self._use_installdir else self.build_dir

        if not (
            dir_to_check.exists()
            and any(path.is_file() for path in dir_to_check.glob("**/*"))
        ):
            log.debug(
                f"{dir_to_check} does not exist or does not contain any file, rebuild needed."
            )
            return True

        fingerprint = self._source_fingerprint()
        if fingerprint is None:
            log.debug(
                f"Source for {self._name} is not a git repository, rebuild needed."
            )
            return True

        marker = self._read_build_marker()
        if marker == fingerprint:
            log.debug(
                f"Build-complete marker matches current source for {self._name}, no rebuild needed."
            )
            return False

        log.debug(
            f"Build-complete marker missing or stale for {self._name}, rebuild needed."
        )
        return True

    def configure(
        self,
        extra_args: list | None = None,
        add_sycl: bool = False,
    ) -> None:
        """Configures the project."""

        is_gdb_mode = os.environ.get("LLVM_BENCHMARKS_USE_GDB", "") == "1"
        build_type = "RelWithDebInfo" if is_gdb_mode else "Release"

        cmd = [
            "cmake",
            f"-S {self.src_dir}",
            f"-B {self.build_dir}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
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
        # run() raises on non-zero exit, so reaching here means the build
        # succeeded. Record a marker so identical reruns can skip rebuilding.
        self.mark_build_complete()

    def install(self) -> None:
        """Installs the project."""
        run(f"cmake --install {self.build_dir}")

    def _can_shallow_clone_ref(self, ref: str) -> bool:
        """Check if we can do a shallow clone with this ref using git ls-remote."""
        try:
            result = run(f"git ls-remote --heads --tags {self._url} {ref}")
            output = result.stdout.decode().strip()

            if output:
                # Found the ref as a branch or tag
                log.debug(
                    f"Ref '{ref}' found as branch/tag via ls-remote, can shallow clone"
                )
                return True
            else:
                # Not found as branch/tag, likely a SHA commit or a special ref
                log.debug(
                    f"Ref '{ref}' not found as branch/tag via ls-remote, likely SHA commit or a special ref"
                )
                return False
        except Exception as e:
            log.debug(
                f"Could not check ref '{ref}' via ls-remote: {e}, assuming SHA commit"
            )
            return False

    def _git_clone(self) -> None:
        """Clone the git repository."""
        try:
            log.debug(f"Cloning {self._url} into {self.src_dir} at ref {self._ref}")
            git_clone_cmd = (
                f"git clone --recursive --depth 1 {self._url} {self.src_dir}"
            )
            if self._shallow_clone:
                if self._can_shallow_clone_ref(self._ref):
                    # Shallow clone for branches and tags only
                    git_clone_cmd = f"git clone --recursive --depth 1 --branch {self._ref} {self._url} {self.src_dir}"
                else:
                    log.debug(
                        f"Cannot shallow clone ref '{self._ref}', clone default branch"
                    )

            run(git_clone_cmd)
            run(f"git fetch {self._url} {self._ref}", cwd=self.src_dir)
            run(f"git checkout FETCH_HEAD", cwd=self.src_dir)
            log.debug(f"Cloned {self._url} into {self.src_dir} at ref {self._ref}")
        except Exception as e:
            log.error(f"Failed to clone repository {self._url}: {e}")
            raise

    def _git_fetch(self) -> None:
        """Fetch the ref from the remote repository."""
        try:
            log.debug(f"Fetching ref '{self._ref}' for {self._url} in {self.src_dir}")
            run("git reset --hard", cwd=self.src_dir)
            run(f"git fetch {self._url} {self._ref}", cwd=self.src_dir)
            run(f"git checkout FETCH_HEAD", cwd=self.src_dir)
            log.debug(f"Fetched changes for {self._url} in {self.src_dir}")
        except Exception as e:
            log.error(f"Failed to fetch updates for repository {self._url}: {e}")
            raise

    def _setup_repo(self) -> bool:
        """Clone a git repository into a specified directory at a specific ref.
        Returns:
            bool: True if the repository was cloned or updated, False if it was already up-to-date.
        """
        if self._src_dir_override is not None:
            log.debug(
                f"Using provided source directory {self.src_dir} for {self._name}, skipping git operations."
            )
            return False
        if os.environ.get("LLVM_BENCHMARKS_UNIT_TESTING") == "1":
            log.debug(
                f"Skipping git operations during unit testing of {self._name} (LLVM_BENCHMARKS_UNIT_TESTING=1)."
            )
            return False
        if options.offline:
            log.debug(
                f"Skipping git operations for {self._name} due to --offline option."
            )
            return False
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
