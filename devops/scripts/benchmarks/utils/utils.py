# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gzip
import os
import shutil
import subprocess

import tarfile
from options import options
from pathlib import Path
import hashlib
from urllib.request import urlopen  # nosec B404
from shutil import copyfileobj


def run(
    command,
    env_vars={},
    cwd=None,
    add_sycl=False,
    ld_library=[],
    timeout=None,
):
    try:
        if timeout is None:
            timeout = options.timeout

        if isinstance(command, str):
            command = command.split()

        env = os.environ.copy()

        for ldlib in ld_library:
            env["LD_LIBRARY_PATH"] = ldlib + os.pathsep + env.get("LD_LIBRARY_PATH", "")

        # order is important, we want provided sycl rt libraries to be first
        if add_sycl:
            sycl_bin_path = os.path.join(options.sycl, "bin")
            env["PATH"] = sycl_bin_path + os.pathsep + env.get("PATH", "")
            sycl_lib_path = os.path.join(options.sycl, "lib")
            env["LD_LIBRARY_PATH"] = (
                sycl_lib_path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )

        env.update(env_vars)

        if options.verbose:
            command_str = " ".join(command)
            env_str = " ".join(f"{key}={value}" for key, value in env_vars.items())
            full_command_str = f"{env_str} {command_str}".strip()
            print(f"Running: {full_command_str}")

        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
        )  # nosec B603

        if options.verbose:
            print(result.stdout.decode())
            print(result.stderr.decode())

        return result
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode())
        print(e.stderr.decode())
        raise


def git_clone(dir, name, repo, commit):
    repo_path = os.path.join(dir, name)

    if os.path.isdir(repo_path) and os.path.isdir(os.path.join(repo_path, ".git")):
        run("git fetch", cwd=repo_path)
        run("git reset --hard", cwd=repo_path)
        run(f"git checkout {commit}", cwd=repo_path)
    elif not os.path.exists(repo_path):
        run(f"git clone --recursive {repo} {repo_path}")
        run(f"git checkout {commit}", cwd=repo_path)
    else:
        raise Exception(
            f"The directory {repo_path} exists but is not a git repository."
        )
    return repo_path


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
                print(f"Version mismatch, cleaning up benchmark directory {dir}")
                shutil.rmtree(dir)
        else:
            raise Exception(
                f"The directory {dir} exists but is not a benchmark work directory."
            )

    os.makedirs(dir)
    prepare_bench_cwd(dir)

    with open(version_file_path, "w") as version_file:
        version_file.write(version)


def create_build_path(directory, name):
    build_path = os.path.join(directory, name)

    if options.rebuild and Path(build_path).exists():
        shutil.rmtree(build_path)

    Path(build_path).mkdir(parents=True, exist_ok=True)

    return build_path


def calculate_checksum(file_path):
    sha_hash = hashlib.sha384()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha_hash.update(byte_block)
    return sha_hash.hexdigest()


def download(dir, url, file, untar=False, unzip=False, checksum=""):
    data_file = os.path.join(dir, file)
    if not Path(data_file).exists():
        print(f"{data_file} does not exist, downloading")
        with urlopen(url) as in_stream, open(data_file, "wb") as out_file:
            copyfileobj(in_stream, out_file)

        calculated_checksum = calculate_checksum(data_file)
        if calculated_checksum != checksum:
            print(
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
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"{data_file} exists, skipping...")
    return data_file
