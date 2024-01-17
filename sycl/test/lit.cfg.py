# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile
from distutils.spawn import find_executable

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "SYCL"

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
dump_only_tests = bool(lit_config.params.get("SYCL_LIB_DUMPS_ONLY", False))
if dump_only_tests:
    config.suffixes = [".dump"]  # Only run dump testing
else:
    config.suffixes = [
        ".c",
        ".cpp",
        ".dump",
        ".test",
    ]  # add .spv. Currently not clear what to do with those

# feature tests are considered not so lightweight, so, they are excluded by default
config.excludes = ["Inputs", "feature-tests"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.sycl_obj_root, "test")

# Propagate some variables from the host environment.
llvm_config.with_system_environment(
    ["PATH", "OCL_ICD_FILENAMES", "SYCL_DEVICE_ALLOWLIST", "SYCL_CONFIG_FILE_NAME"]
)

config.substitutions.append(("%python", '"%s"' % (sys.executable)))

# Propagate extra environment variables
if config.extra_environment:
    lit_config.note("Extra environment variables")
    for env_pair in config.extra_environment.split(","):
        [var, val] = env_pair.split("=")
        if val:
            llvm_config.with_environment(var, val)
            lit_config.note("\t" + var + "=" + val)
        else:
            lit_config.note("\tUnset " + var)
            llvm_config.with_environment(var, "")

# If major release preview library is enabled we can enable the feature.
if config.sycl_preview_lib_enabled == "ON":
    config.available_features.add("preview-breaking-changes-supported")

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
# Add 'libcxx' feature to filter out all SYCL abi tests when SYCL runtime
# is built with llvm libcxx. This feature is added for Linux only since MSVC
# CL compiler doesn't support to use llvm libcxx instead of MSVC STL.
if platform.system() == "Linux":
    config.available_features.add("linux")
    if config.sycl_use_libcxx == "ON":
        config.available_features.add("libcxx")
    llvm_config.with_system_environment("LD_LIBRARY_PATH")
    llvm_config.with_environment(
        "LD_LIBRARY_PATH", config.sycl_libs_dir, append_path=True
    )

elif platform.system() == "Windows":
    config.available_features.add("windows")
    llvm_config.with_system_environment("LIB")
    llvm_config.with_environment("LIB", config.sycl_libs_dir, append_path=True)

elif platform.system() == "Darwin":
    # FIXME: surely there is a more elegant way to instantiate the Xcode directories.
    llvm_config.with_system_environment("CPATH")
    llvm_config.with_environment(
        "CPATH",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1",
        append_path=True,
    )
    llvm_config.with_environment(
        "CPATH",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/",
        append_path=True,
    )
    llvm_config.with_environment("DYLD_LIBRARY_PATH", config.sycl_libs_dir)

llvm_config.with_environment("PATH", config.sycl_tools_dir, append_path=True)

config.substitutions.append(("%threads_lib", config.sycl_threads_lib))
config.substitutions.append(("%sycl_libs_dir", config.sycl_libs_dir))
config.substitutions.append(("%sycl_include", config.sycl_include))
config.substitutions.append(("%sycl_source_dir", config.sycl_source_dir))
config.substitutions.append(("%llvm_main_include_dir", config.llvm_main_include_dir))
config.substitutions.append(("%opencl_libs_dir", config.opencl_libs_dir))
config.substitutions.append(("%level_zero_include_dir", config.level_zero_include_dir))
config.substitutions.append(("%opencl_include_dir", config.opencl_include_dir))
config.substitutions.append(("%cuda_toolkit_include", config.cuda_toolkit_include))
config.substitutions.append(("%sycl_tools_src_dir", config.sycl_tools_src_dir))
config.substitutions.append(("%llvm_build_lib_dir", config.llvm_build_lib_dir))
config.substitutions.append(("%llvm_build_bin_dir", config.llvm_build_bin_dir))

llvm_symbolizer = os.path.join(config.llvm_build_bin_dir, "llvm-symbolizer")
llvm_config.with_environment("LLVM_SYMBOLIZER_PATH", llvm_symbolizer)

sycl_host_only_options = "-std=c++17 -Xclang -fsycl-is-host"
for include_dir in [
    config.sycl_include,
    config.level_zero_include_dir,
    config.opencl_include_dir,
    config.sycl_include + "/sycl/",
]:
    if include_dir:
        sycl_host_only_options += " -isystem %s" % include_dir
config.substitutions.append(("%fsycl-host-only", sycl_host_only_options))

config.substitutions.append(
    ("%sycl_lib", " -lsycl7" if platform.system() == "Windows" else "-lsycl")
)

llvm_config.add_tool_substitutions(["llvm-spirv"], [config.sycl_tools_dir])

triple = lit_config.params.get("SYCL_TRIPLE", "spir64-unknown-unknown")
lit_config.note("Triple: {}".format(triple))
config.substitutions.append(("%sycl_triple", triple))

additional_flags = config.sycl_clang_extra_flags.split(" ")

if config.cuda_be == "ON":
    config.available_features.add("cuda_be")

if config.hip_be == "ON":
    config.available_features.add("hip_be")

if config.opencl_be == "ON":
    config.available_features.add("opencl_be")

if config.level_zero_be == "ON":
    config.available_features.add("level_zero_be")

if config.native_cpu_be == "ON":
    config.available_features.add("native_cpu_be")

if "nvptx64-nvidia-cuda" in triple:
    llvm_config.with_system_environment("CUDA_PATH")
    config.available_features.add("cuda")

if "amdgcn-amd-amdhsa" in triple:
    llvm_config.with_system_environment("ROCM_PATH")
    config.available_features.add("hip_amd")
    # For AMD the specific GPU has to be specified with --offload-arch
    if not any([f.startswith("--offload-arch") for f in additional_flags]):
        # If the offload arch wasn't specified in SYCL_CLANG_EXTRA_FLAGS,
        # hardcode it to gfx906, this is fine because only compiler tests
        additional_flags += [
            "-Xsycl-target-backend=amdgcn-amd-amdhsa",
            "--offload-arch=gfx906",
        ]

# Dump-only tests do not have clang available
if not dump_only_tests:
    llvm_config.use_clang(additional_flags=additional_flags)

# Set timeout for test = 10 mins
try:
    import psutil

    lit_config.maxIndividualTestTime = 600
except ImportError:
    pass
