#!/usr/bin/env python3
"""Build a DPC++ toolchain with profile-guided optimization.

The script deliberately composes buildbot/configure.py instead of duplicating
the DPC++ CMake configuration. It produces an instrumented toolchain, exercises
that toolchain with compiler-focused test targets and a clean training build,
merges the raw profiles, and then builds an optimized toolchain from the merged
profile.
"""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

DEFAULT_TRAINING_TARGETS = ("check-clang", "check-llvm", "check-sycl")
INSTRUMENTED_PROFILE_FLAGS = "-Xclang -mllvm -Xclang -vp-counters-per-site=6"
MULTI_CONFIG_GENERATORS = ("Ninja Multi-Config", "Xcode")


def _format_command(command):
    return " ".join(shlex.quote(str(argument)) for argument in command)


class CommandRunner:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.environment = os.environ.copy()
        # The profile runtime gives this variable precedence over the output
        # path compiled into instrumented tools. Do not let an ambient value
        # redirect this workflow's profiles outside its isolated directory.
        self.environment.pop("LLVM_PROFILE_FILE", None)

    def run(self, command, cwd=None):
        location = Path(cwd or Path.cwd())
        print(f"[{location}] {_format_command(command)}")
        if not self.dry_run:
            subprocess.run(command, cwd=location, check=True, env=self.environment)


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


def _validate_generator(generator):
    if generator in MULTI_CONFIG_GENERATORS or generator.startswith("Visual Studio "):
        raise RuntimeError(
            f"Multi-configuration generator {generator!r} is not supported. "
            "Use a single-configuration generator such as Ninja."
        )


def _cmake_definition(configure_args, name):
    prefix = "--cmake-opt=-D"
    value = None
    for argument in configure_args:
        if not argument.startswith(prefix):
            continue
        key, separator, candidate = argument[len(prefix) :].partition("=")
        if separator and key.split(":", 1)[0].upper() == name.upper():
            value = candidate
    return value


def _cmake_value_is_true(value):
    normalized = value.strip().upper()
    false_values = {"", "0", "FALSE", "OFF", "NO", "N", "IGNORE", "NOTFOUND"}
    return normalized not in false_values and not normalized.endswith("-NOTFOUND")


def _uses_llvm_dylib(configure_args):
    enabled = "--llvm-link-dylib" in configure_args
    override = _cmake_definition(configure_args, "LLVM_LINK_LLVM_DYLIB")
    return _cmake_value_is_true(override) if override is not None else enabled


def _instrumented_options(common_options, profile_dir, configure_args):
    options = common_options + [
        "LLVM_BUILD_INSTRUMENTED=IR",
        f"LLVM_PROFILE_DATA_DIR={profile_dir}",
        "LLVM_VP_COUNTERS_PER_SITE=6",
    ]
    if not _uses_llvm_dylib(configure_args):
        # HandleLLVMOptions adds this flag itself for dylib builds, using the
        # LLVM_VP_COUNTERS_PER_SITE value above. Static builds need the flag
        # supplied explicitly, but adding it to both paths would duplicate it.
        options.extend(
            (
                f"CMAKE_C_FLAGS={INSTRUMENTED_PROFILE_FLAGS}",
                f"CMAKE_CXX_FLAGS={INSTRUMENTED_PROFILE_FLAGS}",
            )
        )
    return options


def _find_profdata_tool(host_clang, requested_tool, dry_run):
    if requested_tool:
        tool = Path(requested_tool).absolute()
        if not dry_run and not tool.is_file():
            raise RuntimeError(f"llvm-profdata was not found at {tool}")
        return tool

    executable = "llvm-profdata.exe" if os.name == "nt" else "llvm-profdata"
    if dry_run and Path(host_clang).parent == Path("."):
        return Path(executable)

    # Raw profile formats are tied to the compiler-rt profile runtime used by
    # the host compiler. Use llvm-profdata from that same toolchain rather than
    # the newly built (and potentially format-incompatible) LLVM toolchain.
    host_tool = Path(host_clang).resolve().with_name(executable)
    if dry_run or host_tool.is_file():
        return host_tool
    raise RuntimeError(
        f"A host llvm-profdata matching {host_clang} is required. "
        "Install it beside the host Clang or pass --llvm-profdata."
    )


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
    # Do not resolve symlinks here. Clang selects C or C++ driver behavior from
    # argv[0], and distributions commonly make clang++ a symlink to clang.
    return os.path.abspath(clang), os.path.abspath(clangxx)


def _instrumented_compilers(instrumented_dir):
    suffix = ".exe" if os.name == "nt" else ""
    return (
        instrumented_dir / "bin" / f"clang{suffix}",
        instrumented_dir / "bin" / f"clang++{suffix}",
    )


def _clear_raw_profiles(profile_dir):
    for profile in profile_dir.glob("*.profraw"):
        profile.unlink()


def _run_training_targets(runner, build_dir, targets, parallelism):
    for target in targets:
        try:
            runner.run(_build_command(build_dir, [target], parallelism))
        except subprocess.CalledProcessError as error:
            # Instrumentation runtime diagnostics can be observed by FileCheck
            # and make otherwise useful test workloads fail. Keep the coverage,
            # continue with the other suites, and rely on the required clean
            # training build as the success gate.
            print(
                f"warning: training target {target} returned "
                f"{error.returncode}; continuing with remaining workloads",
                file=sys.stderr,
            )


def _merge_profiles(runner, profdata_tool, profile_dir, output_file):
    if runner.dry_run:
        profiles = [str(profile_dir / "*.profraw")]
    else:
        profiles = [str(profile) for profile in sorted(profile_dir.glob("*.profraw"))]
        if not profiles:
            raise RuntimeError(
                f"No raw profiles were produced in {profile_dir}. "
                "Check that the training targets exercised instrumented tools."
            )
        if not profdata_tool.is_file():
            raise RuntimeError(f"llvm-profdata was not found at {profdata_tool}")

    if not runner.dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    runner.run([str(profdata_tool), "merge", f"-output={output_file}", *profiles])


def _validate_fresh_outputs(
    instrumented_dir, training_dir, optimized_dir, profile_output, profile_only
):
    outputs = [instrumented_dir, training_dir, profile_output]
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
        raise RuntimeError(f"Expected the DPC++ configure script at {configure_script}")
    _validate_generator(args.cmake_gen)

    instrumented_dir = output_dir / "instrumented"
    training_dir = output_dir / "training"
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
            instrumented_dir,
            training_dir,
            optimized_dir,
            profile_output,
            args.profile_only,
        )
    host_clang, host_clangxx = _find_host_compilers(
        args.host_clang, args.host_clangxx, args.dry_run
    )

    common_options = [
        f"CMAKE_C_COMPILER={host_clang}",
        f"CMAKE_CXX_COMPILER={host_clangxx}",
    ]

    instrumented_options = _instrumented_options(
        common_options, profile_dir, args.configure_arg
    )
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

    instrumented_clang, instrumented_clangxx = _instrumented_compilers(instrumented_dir)
    runner.run(
        _configure_command(
            configure_script,
            source_dir,
            training_dir,
            args.cmake_gen,
            [
                f"CMAKE_C_COMPILER={instrumented_clang}",
                f"CMAKE_CXX_COMPILER={instrumented_clangxx}",
            ],
            args.configure_arg,
        ),
        cwd=source_dir,
    )
    if not runner.dry_run:
        # Discard profiles emitted while building/configuring the instrumented
        # compiler. Only the explicit workloads below should train PGO.
        _clear_raw_profiles(profile_dir)

    _run_training_targets(runner, instrumented_dir, training_targets, args.parallelism)
    runner.run(
        _build_command(training_dir, [args.build_target], args.parallelism),
        cwd=source_dir,
    )

    profdata_tool = _find_profdata_tool(host_clang, args.llvm_profdata, args.dry_run)
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
        help="llvm-profdata executable (default: host Clang toolchain's tool)",
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
