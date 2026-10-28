#!/usr/bin/env python3
"""Build a DPC++ toolchain with profile-guided optimization.

The script deliberately composes buildbot/configure.py instead of duplicating
the DPC++ CMake configuration. It produces an instrumented toolchain, exercises
that toolchain with compiler-focused test targets, merges the raw profiles, and
then builds an optimized toolchain from the merged profile.
"""

import argparse
import glob
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


DEFAULT_TRAINING_TARGETS = ("check-clang", "check-llvm", "check-sycl")


def _format_command(command):
    return " ".join(shlex.quote(str(argument)) for argument in command)


class CommandRunner:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def run(self, command, cwd=None):
        location = Path(cwd or Path.cwd())
        print(f"[{location}] {_format_command(command)}")
        if not self.dry_run:
            subprocess.run(command, cwd=location, check=True)


def _configure_command(
    configure_script,
    source_dir,
    build_dir,
    generator,
    cmake_options,
    configure_args,
):
    command = [
        sys.executable,
        str(configure_script),
        "--src-dir",
        str(source_dir),
        "--obj-dir",
        str(build_dir),
        "--cmake-gen",
        generator,
    ]
    command.extend(f"--cmake-opt=-D{option}" for option in cmake_options)
    command.extend(configure_args)
    return command


def _build_command(build_dir, targets, parallelism):
    command = ["cmake", "--build", str(build_dir)]
    if parallelism:
        command.extend(("--parallel", str(parallelism)))
    command.append("--target")
    command.extend(targets)
    return command


def _find_profdata_tool(instrumented_dir, requested_tool):
    if requested_tool:
        return Path(requested_tool).resolve()

    executable = "llvm-profdata.exe" if os.name == "nt" else "llvm-profdata"
    return instrumented_dir / "bin" / executable


def _find_host_compilers(requested_clang, requested_clangxx, dry_run):
    clang = requested_clang or shutil.which("clang")
    clangxx = requested_clangxx or shutil.which("clang++")
    if dry_run:
        return clang or "clang", clangxx or "clang++"
    if not clang or not Path(clang).is_file():
        raise RuntimeError(
            "A host Clang compiler is required. Add clang to PATH or pass "
            "--host-clang."
        )
    if not clangxx or not Path(clangxx).is_file():
        raise RuntimeError(
            "A host Clang++ compiler is required. Add clang++ to PATH or pass "
            "--host-clangxx."
        )
    return str(Path(clang).resolve()), str(Path(clangxx).resolve())


def _merge_profiles(runner, profdata_tool, profile_dir, output_file):
    if runner.dry_run:
        profiles = [str(profile_dir / "*.profraw")]
    else:
        profiles = sorted(glob.glob(str(profile_dir / "*.profraw")))
        if not profiles:
            raise RuntimeError(
                f"No raw profiles were produced in {profile_dir}. "
                "Check that the training targets exercised instrumented tools."
            )
        if not profdata_tool.is_file():
            raise RuntimeError(f"llvm-profdata was not found at {profdata_tool}")

    if not runner.dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    runner.run(
        [str(profdata_tool), "merge", f"-output={output_file}", *profiles]
    )


def _validate_fresh_outputs(
    instrumented_dir, optimized_dir, profile_output, profile_only
):
    outputs = [instrumented_dir, profile_output]
    if not profile_only:
        outputs.append(optimized_dir)
    existing = [path for path in outputs if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"Refusing to reuse PGO output paths: {rendered}. "
            "Choose a fresh --out-dir to avoid stale profile data."
        )


def run_pgo_build(args):
    source_dir = Path(args.src_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    configure_script = source_dir / "buildbot" / "configure.py"
    if not configure_script.is_file():
        raise RuntimeError(
            f"Expected the DPC++ configure script at {configure_script}"
        )

    instrumented_dir = output_dir / "instrumented"
    optimized_dir = output_dir / "optimized"
    profile_dir = instrumented_dir / "profiles"
    profile_output = (
        Path(args.profile_output).resolve()
        if args.profile_output
        else output_dir / "dpcpp.profdata"
    )
    training_targets = args.training_target or list(DEFAULT_TRAINING_TARGETS)
    runner = CommandRunner(args.dry_run)
    if not runner.dry_run:
        _validate_fresh_outputs(
            instrumented_dir, optimized_dir, profile_output, args.profile_only
        )
    host_clang, host_clangxx = _find_host_compilers(
        args.host_clang, args.host_clangxx, args.dry_run
    )

    common_options = [
        f"CMAKE_C_COMPILER={host_clang}",
        f"CMAKE_CXX_COMPILER={host_clangxx}",
    ]

    instrumented_options = common_options + [
        "LLVM_BUILD_INSTRUMENTED=IR",
        f"LLVM_PROFILE_DATA_DIR={profile_dir}",
    ]
    runner.run(
        _configure_command(
            configure_script,
            source_dir,
            instrumented_dir,
            args.cmake_gen,
            instrumented_options,
            args.configure_arg,
        ),
        cwd=source_dir,
    )
    runner.run(
        _build_command(instrumented_dir, [args.build_target], args.parallelism),
        cwd=source_dir,
    )
    runner.run(
        _build_command(instrumented_dir, training_targets, args.parallelism),
        cwd=source_dir,
    )

    profdata_tool = _find_profdata_tool(instrumented_dir, args.llvm_profdata)
    _merge_profiles(runner, profdata_tool, profile_dir, profile_output)

    if args.profile_only:
        print(f"Merged profile: {profile_output}")
        return

    runner.run(
        _configure_command(
            configure_script,
            source_dir,
            optimized_dir,
            args.cmake_gen,
            common_options + [f"LLVM_PROFDATA_FILE={profile_output}"],
            args.configure_arg,
        ),
        cwd=source_dir,
    )
    runner.run(
        _build_command(optimized_dir, [args.build_target], args.parallelism),
        cwd=source_dir,
    )
    print(f"Optimized toolchain: {optimized_dir}")
    print(f"Merged profile: {profile_output}")


def _parse_args(argv=None):
    default_source = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build an instrumented DPC++ toolchain, collect a profile, "
        "and build a PGO-optimized toolchain."
    )
    parser.add_argument(
        "--src-dir",
        default=str(default_source),
        help="Intel LLVM source directory (default: repository root)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(default_source / "build-pgo"),
        help="directory for instrumented and optimized builds",
    )
    parser.add_argument("--cmake-gen", default="Ninja", help="CMake generator")
    parser.add_argument(
        "--host-clang",
        help="host clang executable (default: clang from PATH)",
    )
    parser.add_argument(
        "--host-clangxx",
        help="host clang++ executable (default: clang++ from PATH)",
    )
    parser.add_argument(
        "--configure-arg",
        action="append",
        default=[],
        help="argument forwarded to buildbot/configure.py; may be repeated",
    )
    parser.add_argument(
        "--training-target",
        action="append",
        help="CMake target used to collect profiles; may be repeated",
    )
    parser.add_argument(
        "--build-target",
        default="all",
        help="target used for the instrumented and optimized toolchains",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        help="maximum number of parallel build jobs",
    )
    parser.add_argument(
        "--llvm-profdata",
        help="llvm-profdata executable (default: instrumented build's tool)",
    )
    parser.add_argument("--profile-output", help="merged profile output path")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="collect the profile without creating the optimized build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every command without configuring or building",
    )
    return parser.parse_args(argv)


def main(argv=None):
    try:
        run_pgo_build(_parse_args(argv))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
