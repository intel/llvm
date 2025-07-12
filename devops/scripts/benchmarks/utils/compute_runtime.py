# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import json
import yaml
import shutil
from abc import ABC, abstractmethod

from .utils import git_clone, run
from .logger import log
from options import options


def _replace_in_file(file_path: str, search_pattern: str, replacement: str) -> None:
    """Replace a pattern in a file with a given replacement."""

    with open(file_path, "r") as file:
        content = file.read()

    modified_content = re.sub(search_pattern, replacement, content)

    with open(file_path, "w") as file:
        file.write(modified_content)


def _remove_directory(path: str) -> None:
    """Clean up a directory if it exists"""
    if path and os.path.exists(path):
        log.info(f"Cleaning directory: {path}")
        try:
            shutil.rmtree(path)
        except OSError as e:
            log.warning(f"Failed to remove directory {path}: {e}")


class Component(ABC):
    """Base class for components of Compute Runtime."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the component."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def build_dir(self) -> str:
        """Returns the build directory for the component."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def install_dir(self) -> str:
        """Returns the installation directory for the component."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abstractmethod
    def src_dir(self) -> str:
        """Returns the source directory for the component."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def configure_cmd(self) -> list[str]:
        """Returns the configure command for the component."""
        return [
            "cmake",
            f"-B {self.build_dir}",
            f"-S {self.src_dir}",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

    def clone_source(self, repo: str, commit: str) -> str:
        """
        Clones the source repository for the component.

        Args:
            repo: Repository URL
            commit: Commit or tag to build

        Returns:
            Path to the cloned source
        """
        return git_clone(options.workdir, self.src_dir, repo, commit)

    def run_build_cmd(self, **kwargs) -> None:
        """Returns the build command for the component."""
        run(f"cmake --build {self.build_dir} -j {options.build_jobs}", **kwargs)

    def run_install_cmd(self) -> None:
        """Returns the install command for the component."""
        run(f"cmake --install {self.build_dir}")

    def build(self, repo: str, commit: str):
        """
        Builds the component.

        Args:
            repo: Repository URL
            commit: Commit or tag to build
        """
        run(self.configure_cmd)
        self.run_build_cmd()
        self.run_install_cmd()

    def get_library_path(self) -> str:
        """Returns the library path for LD_LIBRARY_PATH."""
        return os.path.join(self.install_dir, "lib")

    def clean(self) -> None:
        """Cleans the component build and install directories."""
        _remove_directory(self.build_dir)
        _remove_directory(self.install_dir)


class LevelZero(Component):
    """Handles the build and setup of the Intel Level Zero runtime."""

    @property
    def name(self) -> str:
        return "level_zero"

    @property
    def build_dir(self) -> str:
        return os.path.join(options.workdir, "level-zero-build")

    @property
    def install_dir(self) -> str:
        return os.path.join(options.workdir, "level-zero-install")

    @property
    def src_dir(self) -> str:
        return os.path.join(options.workdir, "level-zero-src")

    def build(self, repo: str, commit: str):
        log.info("Building Level Zero...")
        level_zero_src = self.clone_source(repo, commit)

        # there's a bug in level-zero CMakeLists.txt that makes it install headers into incorrect location.
        cmakelists_path = os.path.join(level_zero_src, "CMakeLists.txt")
        _replace_in_file(
            cmakelists_path, r"DESTINATION \./include/", "DESTINATION include/"
        )

        super().build(repo, commit)
        log.info("Level Zero build complete.")


class Gmmlib(Component):
    """Handles the build and setup of the Intel GMM library."""

    @property
    def name(self) -> str:
        return "gmmlib"

    @property
    def build_dir(self) -> str:
        return os.path.join(options.workdir, "gmmlib-build")

    @property
    def install_dir(self) -> str:
        return os.path.join(options.workdir, "gmmlib-install")

    @property
    def src_dir(self) -> str:
        return os.path.join(options.workdir, "gmmlib-src")

    def build(self, repo: str, commit: str):
        log.info("Building GMM library...")
        self.clone_source(repo, commit)
        super().build(repo, commit)
        log.info("GMM library build complete.")


class Igc(Component):
    """Handles the build and setup of the Intel IGC (Intel Graphics Compiler)."""

    @property
    def name(self) -> str:
        return "igc"

    @property
    def build_dir(self) -> str:
        return os.path.join(options.workdir, "igc-build")

    @property
    def install_dir(self) -> str:
        return os.path.join(options.workdir, "igc-install")

    @property
    def src_dir(self) -> str:
        return os.path.join(options.workdir, "igc-src")

    @property
    def configure_cmd(self) -> list[str]:
        return super().configure_cmd + [
            "-DCMAKE_C_FLAGS=-Wno-error",
            "-DCMAKE_CXX_FLAGS=-Wno-error",
        ]

    def run_build_cmd(self, **kwargs) -> None:
        # set timeout to 2h. IGC takes A LONG time to build if building from scratch.
        super().run_build_cmd(timeout=60 * 60 * 2)

    def run_install_cmd(self) -> None:
        # cmake --install doesn't work...
        run("make install", cwd=self.build_dir)

    def build(self, repo: str, commit: str):
        """
        Builds the IGC component.

        Args:
            repo: Repository URL
            commit: Commit or tag to build
        """
        log.info(f"Building IGC...")
        self.clone_source(repo, commit)
        # Clone all igc dependencies
        git_clone(
            options.workdir,
            "vc-intrinsics",
            "https://github.com/intel/vc-intrinsics",
            "9d255266e1df8f1dc5d11e1fbb03213acfaa4fc7",
        )
        llvm_project_repo = git_clone(
            options.workdir,
            "llvm-project",
            "https://github.com/llvm/llvm-project",
            "llvmorg-15.0.7",
        )
        llvm_projects_path = os.path.join(llvm_project_repo, "llvm", "projects")
        git_clone(
            llvm_projects_path,
            "opencl-clang",
            "https://github.com/intel/opencl-clang",
            "ocl-open-150",
        )
        git_clone(
            llvm_projects_path,
            "llvm-spirv",
            "https://github.com/KhronosGroup/SPIRV-LLVM-Translator",
            "llvm_release_150",
        )
        git_clone(
            options.workdir,
            "SPIRV-Tools",
            "https://github.com/KhronosGroup/SPIRV-Tools.git",
            "f289d047f49fb60488301ec62bafab85573668cc",
        )
        git_clone(
            options.workdir,
            "SPIRV-Headers",
            "https://github.com/KhronosGroup/SPIRV-Headers.git",
            "0e710677989b4326ac974fd80c5308191ed80965",
        )

        super().build(repo, commit)


class ManifestReader:
    """Reads and parses manifest files."""

    def __init__(self, manifest_path: str):
        self._manifest: dict = self._load_manifest(manifest_path)

    def get_component_info(self, component_name: str) -> dict:
        """
        Gets repository URL and commit from manifest.

        Args:
            manifest: Dictionary containing manifest content
            component_name: Name of the component to look for

        Returns:
            Dictionary with 'repository' and 'revision'
        """
        log.debug(f"Getting component info for {component_name} from manifest")
        components_dict = self._manifest.get("components")
        component = components_dict.get(component_name) if components_dict else None
        if not component:
            raise RuntimeError(f"Component {component_name} not found in manifest")

        repo_url = component.get("repository")
        commit = component.get("revision")
        if not repo_url or not commit:
            raise RuntimeError(f"Repository or revision not found for {component_name}")
        log.debug(
            f"Found repository: {repo_url}, revision: {commit} for component {component_name}"
        )
        return {"repository": repo_url, "revision": commit}

    def _load_manifest(self, manifest_path: str):
        """Loads the manifest file."""
        if not os.path.exists(manifest_path):
            raise RuntimeError(f"Manifest file not found: {manifest_path}")
        with open(manifest_path, "r") as file:
            manifest = yaml.safe_load(file)
        if not manifest:
            raise RuntimeError(f"Failed to parse manifest file: {manifest_path}")
        return manifest


class BuildState:
    """Tracks the state of component builds to avoid unnecessary rebuilds."""

    def __init__(self):
        self.state_file = os.path.join(options.workdir, "component_versions.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Loads the build state from disk or creates an empty state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log.warning(
                    f"Failed to load build state file: {e}. Creating new state."
                )
                return {}
        return {}

    def _save_state(self) -> None:
        """Saves the current build state to disk."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except IOError as e:
            log.warning(f"Failed to save build state file: {e}")

    def get_components_to_rebuild(
        self, components: dict[str, Component], manifest_reader: ManifestReader
    ) -> list[Component]:
        """
        Identifies which components have changed by comparing versions with stored state
        and checking if install dirs exist.

        Args:
            components: dict[str, Component]: Dictionary of components to check
            manifest_reader: ManifestReader instance to get component info

        Returns:
            list[str]: List of components that have changed
        """
        to_rebuild = []
        for component in components.values():
            component_info = manifest_reader.get_component_info(component.name)
            version = component_info["revision"]
            install_dir = component.install_dir

            log.debug(
                f"Checking component {component.name} with version {version} in {install_dir}"
            )
            if component.name not in self.state:
                log.info(
                    f"Component {component.name} not found in the Compute Runtime's components state file, will rebuild."
                )
                to_rebuild.append(component)
                continue

            if self.state[component.name] != version:
                log.info(
                    f"Component {component.name} version changed from {self.state.get(component.name, 'unknown')} to {version}, will rebuild."
                )
                to_rebuild.append(component)
                continue

            if not os.path.exists(install_dir):
                log.info(
                    f"Installation directory for {component.name} does not exist at {install_dir}, will rebuild."
                )
                to_rebuild.append(component)
                continue
        log.debug(f"Components to rebuild: {[comp.name for comp in to_rebuild]}")

        return to_rebuild

    def set_component_state(self, component: str, version: str) -> None:
        """Updates the state for a component after a successful build.

        Args:
            component: Name of the component
            version: Version or commit hash of the component
        """
        self.state[component] = version
        self._save_state()


class ComputeRuntime:
    """Handles the build and setup of the Intel Compute Runtime and its dependencies."""

    def __init__(self):
        self._components = {"level_zero": LevelZero(), "gmmlib": Gmmlib()}
        if options.build_igc:
            self._components["igc"] = Igc()
        self._build_dir = os.path.join(options.workdir, "compute-runtime-build")
        self._setup()

    def ld_libraries(self) -> list[str]:
        """
        Get the list of library paths needed for LD_LIBRARY_PATH.

        Returns:
            list[str]: List of library paths
        """
        paths = []

        for component in self._components.values():
            if os.path.exists(component.install_dir):
                paths.append(component.get_library_path())
            else:
                raise RuntimeError(
                    f"Path to {component.name} libraries not found at {component.install_dir}"
                )

        compute_runtime_bin_path = os.path.join(self._build_dir, "bin")
        if os.path.exists(compute_runtime_bin_path):
            paths.append(compute_runtime_bin_path)
        else:
            raise RuntimeError(
                f"Path to Compute Runtime binaries not found at {compute_runtime_bin_path}"
            )

        return paths

    def env_vars(self) -> dict:
        """
        Get environment variables needed for runtime.

        Returns:
            dict: Environment variables to set
        """
        env_vars = {}

        if os.path.exists(self._build_dir):
            libze_path = os.path.join(self._build_dir, "bin", "libze_intel_gpu.so")
            libigdrcl_path = os.path.join(self._build_dir, "bin", "libigdrcl.so")

            if os.path.exists(libze_path):
                env_vars["ZE_ENABLE_ALT_DRIVERS"] = libze_path
            else:
                raise RuntimeError(f"Level Zero driver not found at {libze_path}")

            if os.path.exists(libigdrcl_path):
                env_vars["OCL_ICD_FILENAMES"] = libigdrcl_path
            else:
                raise RuntimeError(f"OpenCL driver not found at {libigdrcl_path}")
        else:
            raise RuntimeError(
                f"Compute Runtime build directory not found at {self._build_dir}"
            )

        return env_vars

    def _setup(self):
        """
        Sets up the Compute Runtime and its dependencies.
        Uses build state to determine if a rebuild is needed.
        Only rebuilds components that have changed and their dependents.
        """
        build_state = BuildState()

        self._compute_runtime_src = git_clone(
            options.workdir,
            "compute-runtime-repo",
            "https://github.com/intel/compute-runtime.git",
            options.compute_runtime_tag,
        )

        # Read the manifest to get component versions
        manifest_reader = ManifestReader(
            os.path.join(self._compute_runtime_src, "manifests", "manifest.yml")
        )

        # Determine which components need to be rebuilt
        rebuild_components = build_state.get_components_to_rebuild(
            self._components, manifest_reader
        )

        # Check if ComputeRuntime itself needs rebuilding
        compute_runtime_changed = self._check_compute_runtime_changed(build_state)

        if not rebuild_components and not compute_runtime_changed:
            log.info(
                "No changes detected in components or ComputeRuntime. Using existing builds."
            )
            return

        components_changed_msg = (
            f"Detected changes in components: {', '.join(comp.name for comp in rebuild_components)}"
            if rebuild_components
            else ""
        )
        runtime_changed_msg = (
            "Detected changes in ComputeRuntime" if compute_runtime_changed else ""
        )
        log.info(" ".join(filter(None, [components_changed_msg, runtime_changed_msg])))

        self._rebuild(
            rebuild_components, manifest_reader, build_state, compute_runtime_changed
        )

    def _check_compute_runtime_changed(self, build_state: BuildState) -> bool:
        """
        Checks if the ComputeRuntime itself needs rebuilding based on version changes
        or if the build directory doesn't exist.

        Args:
            build_state: BuildState instance with saved component states

        Returns:
            bool: True if ComputeRuntime needs rebuilding, False otherwise
        """
        # Check if build directory exists
        if not os.path.exists(self._build_dir):
            log.info(
                f"ComputeRuntime build directory does not exist at {self._build_dir}, will rebuild."
            )
            return True

        # Check if version has changed
        current_version = options.compute_runtime_tag
        previous_version = build_state.state.get("compute_runtime")

        if not previous_version:
            log.info("ComputeRuntime not found in state file, will rebuild.")
            return True

        if previous_version != current_version:
            log.info(
                f"ComputeRuntime version changed from {previous_version} to {current_version}, will rebuild."
            )
            return True

        return False

    def _build(self):
        """
        Builds the Compute Runtime.
        """
        log.info("Building Compute Runtime...")
        configure_command = [
            f"cmake",
            f"-B{self._build_dir}",
            f"-S{self._compute_runtime_src}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DNEO_ENABLE_i915_PRELIM_DETECTION=1",
            f"-DNEO_SKIP_UNIT_TESTS=1",
            f"-DGMM_DIR={self._components['gmmlib'].install_dir}",
            f"-DLEVEL_ZERO_ROOT={self._components['level_zero'].install_dir}",
        ]
        if self._components.get("igc"):
            configure_command.append(f"-DIGC_DIR={self._components['igc'].install_dir}")
        run(configure_command)
        run(f"cmake --build {self._build_dir} -j {options.build_jobs}")
        log.info("Compute Runtime build complete.")

    def _rebuild(
        self,
        rebuild_components: list[Component],
        manifest_reader: ManifestReader,
        build_state: BuildState,
        compute_runtime_changed: bool = False,
    ):
        """
        Rebuilds the specified components and optionally the ComputeRuntime.

        Args:
            rebuild_components: List of components to rebuild
            manifest_reader: ManifestReader instance to get component info
            build_state: BuildState instance to update component states
            compute_runtime_changed: Whether to force rebuild the ComputeRuntime even if no components changed
        """
        if rebuild_components:
            log.info(
                f"Clean rebuild of components: {', '.join(comp.name for comp in rebuild_components)}"
            )

            for component in rebuild_components:
                component.clean()
                component.build(
                    manifest_reader.get_component_info(component.name)["repository"],
                    manifest_reader.get_component_info(component.name)["revision"],
                )
                build_state.set_component_state(
                    component.name,
                    manifest_reader.get_component_info(component.name)["revision"],
                )

        # Rebuild compute_runtime if any dependency changed or if marked as changed
        if rebuild_components or compute_runtime_changed:
            _remove_directory(self._build_dir)
            self._build()
            build_state.set_component_state(
                "compute_runtime", options.compute_runtime_tag
            )


# ComputeRuntime singleton instance
_compute_runtime_instance = None


def get_compute_runtime() -> ComputeRuntime:
    """Returns a singleton instance of ComputeRuntime"""
    global _compute_runtime_instance
    if _compute_runtime_instance is None:
        _compute_runtime_instance = ComputeRuntime()
    return _compute_runtime_instance
