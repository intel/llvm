# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'LIBCLC'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.cl', '.cpp']

# excludes: A list of directories  and fles to exclude from the testsuite.
config.excludes = ['CMakeLists.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(os.path.dirname(__file__), 'binding')

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.test_run_dir, 'test')

libclc_inc = os.path.join(config.libclc_root, 'generic', 'include')

target = lit_config.params.get('target', '')
builtins = lit_config.params.get('builtins', '')
cpu = lit_config.params.get('cpu', '')
cpu = [] if cpu == '' else ["-mcpu=" + cpu]

llvm_config.use_default_substitutions()

clang_flags = [
  "-fno-builtin",
  "-I", libclc_inc,
  "-target", target,
  "-Xclang", "-fdeclare-spirv-builtins",
  "-Xclang", "-mlink-builtin-bitcode",
  "-Xclang", os.path.join(config.llvm_libs_dir, "clc", builtins),
  "-nogpulib"
]

if target == 'amdgcn--amdhsa':
    config.available_features.add('amdgcn')

    # libclc for amdgcn is currently built for tahiti which doesn't support
    # fp16 so disable the extension for the tests
    clang_flags += ['-Xclang', '-cl-ext=-cl_khr_fp16']

llvm_config.use_clang(additional_flags=clang_flags + cpu)

config.substitutions.append(('%PATH%', config.environment['PATH']))

tool_dirs = [config.llvm_tools_dir]

tools = ['llvm-dis', 'not']

llvm_config.add_tool_substitutions(tools, tool_dirs)
