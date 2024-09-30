# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import json
import shutil
import subprocess # nosec B404
from pathlib import Path
from benches.result import Result
from benches.options import options

def run(command, env_vars={}, cwd=None, add_sycl=False):
    try:
        if isinstance(command, str):
            command = command.split()

        env = os.environ.copy()

        if add_sycl:
            sycl_bin_path = os.path.join(options.sycl, 'bin')
            env['PATH'] = sycl_bin_path + os.pathsep + env.get('PATH', '')
            sycl_lib_path = os.path.join(options.sycl, 'lib')
            env['LD_LIBRARY_PATH'] = sycl_lib_path + os.pathsep + env.get('LD_LIBRARY_PATH', '')

        env.update(env_vars)
        result = subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=options.timeout) # nosec B603

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

    if os.path.isdir(repo_path) and os.path.isdir(os.path.join(repo_path, '.git')):
        run("git fetch", cwd=repo_path)
        run(f"git checkout {commit}", cwd=repo_path)
    elif not os.path.exists(repo_path):
        run(f"git clone --recursive {repo} {repo_path}")
        run(f"git checkout {commit}", cwd=repo_path)
    else:
        raise Exception(f"The directory {repo_path} exists but is not a git repository.")
    return repo_path

def save_benchmark_results(dir, save_name, benchmark_data: list[Result]):
    serialized = [res.to_json() for res in benchmark_data]
    results_dir = Path(os.path.join(dir, 'results'))
    os.makedirs(results_dir, exist_ok=True)

    file_path = Path(os.path.join(results_dir, f"{save_name}.json"))
    with file_path.open('w') as file:
        json.dump(serialized, file, indent=4)
    print(f"Benchmark results saved to {file_path}")

def load_benchmark_results(dir, compare_name) -> list[Result]:
    file_path = Path(os.path.join(dir, 'results', f"{compare_name}.json"))
    if file_path.exists():
        with file_path.open('r') as file:
            data = json.load(file)
            return [Result.from_json(item) for item in data]
    else:
        return None

def prepare_bench_cwd(dir):
    # we need 2 deep to workaround a problem with a fixed relative path in cudaSift
    options.benchmark_cwd = os.path.join(dir, 'bcwd', 'bcwd')
    if os.path.exists(options.benchmark_cwd):
        shutil.rmtree(options.benchmark_cwd)
    os.makedirs(options.benchmark_cwd)

def prepare_workdir(dir, version):
    version_file_path = os.path.join(dir, 'BENCH_WORKDIR_VERSION')

    if os.path.exists(dir):
        if os.path.isfile(version_file_path):
            with open(version_file_path, 'r') as version_file:
                workdir_version = version_file.read().strip()

            if workdir_version == version:
                prepare_bench_cwd(dir)
                return
            else:
                print(f"Version mismatch, cleaning up benchmark directory {dir}")
                shutil.rmtree(dir)
        else:
            raise Exception(f"The directory {dir} exists but is a benchmark work directory.")

    os.makedirs(dir)
    prepare_bench_cwd(dir)

    with open(version_file_path, 'w') as version_file:
        version_file.write(version)

def create_build_path(directory, name):
    build_path = os.path.join(directory, name)

    if options.rebuild and Path(build_path).exists():
        shutil.rmtree(build_path)

    Path(build_path).mkdir(parents=True, exist_ok=True)

    return build_path
