# -*- Python -*-

# Configuration file for the 'lit' test runner.
import os
import sys

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = 'libdevice'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.txt']

# excludes: A list of directories to exclude from the testsuite.
config.excludes = ['CMakeLists.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.libdevice_binary_dir, 'test')

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest()

llvm_config.use_default_substitutions()

tool_dirs = [config.clang_tools_dir, config.llvm_tools_dir]

tools = [
    'clang-offload-bundler', 'llvm-dis', 'llvm-spirv'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.has_libsycldevice:
    config.available_features.add("libsycldevice");
config.substitutions.append(
    ('%libsycldevice_obj_dir', config.libdevice_library_dir))
if sys.platform in ['win32']:
    config.substitutions.append(
        ('%libsycldevice_spv_dir', config.libdevice_runtime_dir))
else:
    config.substitutions.append(
        ('%libsycldevice_spv_dir', config.libdevice_library_dir))
