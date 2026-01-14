# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool


def quote(s):
    return f"'{s}'"


# Configuration file for the 'lit' test runner.

if config.libclc_target is None:
    lit_config.fatal("libclc_target parameter must be set when running directly")

if config.libclc_target not in config.libclc_targets_to_test:
    lit_config.fatal(
        f"libclc_target '{config.libclc_target}' is not built. "
        f"Available targets: {', '.join(quote(s) for s in config.libclc_targets_to_test)}"
    )

# name: The name of this test suite.
config.name = f"LIBCLC-{config.libclc_target.upper()}"


# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".cl", ".cpp"]

# excludes: A list of directories  and fles to exclude from the testsuite.
config.excludes = ["CMakeLists.txt"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(os.path.dirname(__file__), "binding")

libclc_inc = os.path.join(config.libclc_root, "libspirv", "include")

# test_exec_root: The root path where tests should be run. We create a unique
# test directory per libclc target to test to avoid data races when multiple
# targets try and access the the same libclc test files.
config.test_exec_root = os.path.join(
    config.libclc_pertarget_test_dir, config.libclc_target
)

llvm_config.use_default_substitutions()

clang_flags = [
    "-fno-builtin",
    "-I",
    libclc_inc,
    "-target",
    config.libclc_target,
    "-Xclang",
    "-fdeclare-spirv-builtins",
    "-Xclang",
    "-mlink-builtin-bitcode",
    "-Xclang",
    os.path.join(config.libclc_output_dir, f"libspirv-{config.libclc_target}.bc"),
    "-nogpulib",
]

if config.libclc_target == "amdgcn--amdhsa":
    # libclc for amdgcn is currently built for tahiti which doesn't support
    # fp16 so disable the extension for the tests
    clang_flags += ["-Xclang", "-cl-ext=-cl_khr_fp16"]

llvm_config.use_clang(additional_flags=clang_flags)

config.substitutions.append(("%PATH%", config.environment["PATH"]))

tool_dirs = [config.llvm_tools_dir]

tools = ["llvm-dis", "not"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
