# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gzip
import os
import shutil
import subprocess
import re

import tarfile
import hashlib
from pathlib import Path
from urllib.request import urlopen  # nosec B404
from shutil import copyfileobj

from options import options
from utils.logger import log


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use as a filename or directory name.
    Replace invalid characters with underscores.
    Invalid characters: " : < > | * ? \r \n
    """
    # Replace invalid characters with underscores
    # Added space to list to avoid directories with spaces which cause issues in shell commands
    invalid_chars = r'[":;<>|*?\r\n ]'
    sanitized = re.sub(invalid_chars, "_", name)
    return sanitized


def run(
    command,
    env_vars={},
    cwd=None,
    add_sycl=False,
    ld_library=[],
    timeout=None,
    input=None,
):
    try:
        if timeout is None:
            timeout = options.timeout

        if isinstance(command, str):
            command = command.split()

        env = os.environ.copy()

        for ldlib in ld_library:
            if os.path.isdir(ldlib):
                env["LD_LIBRARY_PATH"] = (
                    ldlib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                )
            else:
                log.warning(f"LD_LIBRARY_PATH component does not exist: {ldlib}")

        # order is important, we want provided sycl rt libraries to be first
        if add_sycl:
            sycl_bin_path = os.path.join(options.sycl, "bin")
            env["PATH"] = sycl_bin_path + os.pathsep + env.get("PATH", "")
            sycl_lib_path = os.path.join(options.sycl, "lib")
            env["LD_LIBRARY_PATH"] = (
                sycl_lib_path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )

        env.update(env_vars)

        command_str = " ".join(command)
        env_str = " ".join(f"{key}={value}" for key, value in env_vars.items())
        full_command_str = f"{env_str} {command_str}".strip()
        log.debug(f"Running: {full_command_str}")

        # Normalize input to bytes if it's a str
        if isinstance(input, str):
            input_bytes = input.encode()
        else:
            input_bytes = input

        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
            input=input_bytes,
        )  # nosec B603

        if result.stdout:
            log.debug(result.stdout.decode())
        if result.stderr:
            log.debug(result.stderr.decode())

        return result
    except subprocess.CalledProcessError as e:
        if e.stdout and e.stdout.decode().strip():
            log.error(e.stdout.decode())
        if e.stderr and e.stderr.decode().strip():
            log.error(e.stderr.decode())
        raise


def prepare_bench_cwd(dir):
    # we need 2 deep to workaround a problem with a fixed relative paths in some velocity benchmarks
    options.benchmark_cwd = os.path.join(dir, "bcwd", "bcwd")
    if os.path.exists(options.benchmark_cwd):
        shutil.rmtree(options.benchmark_cwd)
    os.makedirs(options.benchmark_cwd)


def prepare_workdir(dir, version):
    version_file_path = os.path.join(dir, "BENCH_WORKDIR_VERSION")

    if os.path.exists(dir):
        if os.path.isfile(version_file_path):
            with open(version_file_path, "r") as version_file:
                workdir_version = version_file.read().strip()

            if workdir_version == version:
                prepare_bench_cwd(dir)
                return
            else:
                log.warning(f"Version mismatch, cleaning up benchmark directory {dir}")
                shutil.rmtree(dir)
        else:
            raise Exception(
                f"The directory {dir} exists but is not a benchmark work directory. "
                f"A BENCH_WORKDIR_VERSION file is expected with version {version} but not found at {version_file_path}."
            )

    os.makedirs(dir)
    prepare_bench_cwd(dir)

    with open(version_file_path, "w") as version_file:
        version_file.write(version)


def calculate_checksum(file_path):
    sha_hash = hashlib.sha384()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha_hash.update(byte_block)
    return sha_hash.hexdigest()


def download(dir, url, file, untar=False, unzip=False, checksum=""):
    data_file = os.path.join(dir, file)
    if not Path(data_file).exists():
        log.info(f"{data_file} does not exist, downloading")
        with urlopen(url) as in_stream, open(data_file, "wb") as out_file:
            copyfileobj(in_stream, out_file)

        calculated_checksum = calculate_checksum(data_file)
        if calculated_checksum != checksum:
            log.critical(
                f"Checksum mismatch: expected {checksum}, got {calculated_checksum}. Refusing to continue."
            )
            exit(1)

        if untar:
            file = tarfile.open(data_file)
            file.extractall(dir)
            file.close()
        if unzip:
            [stripped_gz, _] = os.path.splitext(data_file)
            with gzip.open(data_file, "rb") as f_in, open(stripped_gz, "wb") as f_out:
                # copyfileobj expects binary file-like objects; type checker may complain about union types
                shutil.copyfileobj(f_in, f_out)  # type: ignore[arg-type]
    else:
        log.debug(f"{data_file} exists, skipping...")
    return data_file


def get_device_architecture(additional_env_vars):
    sycl_ls_output = run(
        ["sycl-ls", "--verbose"], add_sycl=True, env_vars=additional_env_vars
    ).stdout.decode()

    architectures = set()
    for line in sycl_ls_output.splitlines():
        if re.match(r" *Architecture:", line):
            _, architecture = line.strip().split(":", 1)
            architectures.add(architecture.strip())

    if len(architectures) != 1:
        raise ValueError(
            f"Expected exactly one device architecture, but found {len(architectures)}: {architectures}."
            "Set ONEAPI_DEVICE_SELECTOR=backend:device_id to specify a single device."
        )

    return architectures.pop()


def prune_old_files(directory: str, keep_count: int = 10):
    """Keep only the most recent keep_count files in the directory."""
    if not os.path.isdir(directory):
        log.debug(f"Directory {directory} does not exist, skipping pruning")
        return

    # Get all files sorted by modification time (newest first)
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    files.sort(key=os.path.getmtime, reverse=True)

    # Remove files beyond the keep count
    files_to_remove = files[keep_count:]
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            log.debug(f"Deleted file: {file_path}")
        except OSError as e:
            log.debug(f"Failed to remove {file_path}: {e}")


def remove_by_prefix(directory: str, prefix: str):
    """Remove files with names starting with prefix."""
    if not os.path.exists(directory):
        return

    for f in os.listdir(directory):
        if f.startswith(prefix):
            file_path = os.path.join(directory, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                log.debug(f"Deleted file: {file_path}")


def remove_by_extension(directory: str, extension: str):
    """Remove files with specified extension from directory."""
    if not os.path.exists(directory):
        return

    for f in os.listdir(directory):
        if f.endswith(extension):
            file_path = os.path.join(directory, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                log.debug(f"Deleted file: {file_path}")
