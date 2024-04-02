# Configuration file for LLVM's lit test runner.

import platform

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "SYCL-FUSION"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Define suffixes for test discovery
config.suffixes = [".ll"]

# The root folder for the test suite is right here where this file resides.
config.test_source_root = os.path.dirname(__file__)

# Define substitutions for the tools used in the tests
tools = ["opt", "FileCheck"]
llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)

# Location and file-ending of shared libraries for the passes
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%shlibdir", config.llvm_shlib_dir))

if "NVPTX" in config.llvm_targets_to_build:
    config.available_features.add('cuda')
if "AMDGPU" in config.llvm_targets_to_build:
    config.available_features.add('hip_amd')
