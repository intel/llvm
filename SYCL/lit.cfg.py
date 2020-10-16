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
config.name = 'SYCL'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp']

config.excludes = ['Inputs']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.sycl_obj_root

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['PATH', 'OCL_ICD_FILENAMES',
    'CL_CONFIG_DEVICES', 'SYCL_DEVICE_ALLOWLIST', 'SYCL_CONFIG_FILE_NAME'])

llvm_config.with_environment('PATH', config.lit_tools_dir, append_path=True)

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
if platform.system() == "Linux":
    config.available_features.add('linux')
    llvm_config.with_system_environment('LD_LIBRARY_PATH')
    llvm_config.with_environment('LD_LIBRARY_PATH', config.sycl_libs_dir, append_path=True)

elif platform.system() == "Windows":
    config.available_features.add('windows')
    llvm_config.with_system_environment('LIB')
    llvm_config.with_environment('LIB', config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment('PATH', config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment('LIB', os.path.join(config.dpcpp_root_dir, 'lib'), append_path=True)

elif platform.system() == "Darwin":
    # FIXME: surely there is a more elegant way to instantiate the Xcode directories.
    llvm_config.with_system_environment('CPATH')
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1", append_path=True)
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/", append_path=True)
    llvm_config.with_environment('DYLD_LIBRARY_PATH', config.sycl_libs_dir)

llvm_config.with_environment('PATH', config.sycl_tools_dir, append_path=True)

for env_pair in config.extra_environment.split(','):
    if env_pair:
        [var,val]=env_pair.split("=")
        llvm_config.with_environment(var,val)

config.substitutions.append( ('%sycl_libs_dir',  config.sycl_libs_dir ) )
config.substitutions.append( ('%sycl_include',  config.sycl_include ) )
if config.opencl_libs_dir:
  config.substitutions.append( ('%opencl_libs_dir',  config.opencl_libs_dir) )
  config.available_features.add('opencl_icd')
config.substitutions.append( ('%opencl_include_dir',  config.opencl_include_dir) )

llvm_config.use_clang()

llvm_config.add_tool_substitutions(['llvm-spirv'], [config.sycl_tools_dir])

if not config.sycl_be:
    config.sycl_be="PI_OPENCL"

config.substitutions.append( ('%sycl_be', config.sycl_be) )
lit_config.note("Backend: {BACKEND}".format(BACKEND=config.sycl_be))

if config.dump_ir_supported:
   config.available_features.add('dump_ir')

cuda = False
if ( config.sycl_be == "PI_OPENCL" ):
    config.available_features.add('opencl')
elif ( config.sycl_be == "PI_CUDA" ):
    config.available_features.add('cuda')
    cuda = True
elif ( config.sycl_be == "PI_LEVEL_ZERO" ):
    config.available_features.add('level_zero')
else:
    lit_config.error("Unknown SYCL BE specified '" +
                     config.sycl_be +
                     "' supported values are PI_OPENCL, PI_CUDA, PI_LEVEL_ZERO")

# ESIMD-specific setup. Requires OpenCL for now.
if "opencl" in config.available_features:
    print(config.available_features)
    esimd_run_substitute = " env SYCL_BE=PI_OPENCL SYCL_DEVICE_TYPE=GPU SYCL_PROGRAM_COMPILE_OPTIONS=-cmc"
    config.substitutions.append( ('%ESIMD_RUN_PLACEHOLDER',  esimd_run_substitute) )
    config.substitutions.append( ('%clangxx-esimd',  config.dpcpp_compiler +
                                  ' ' + '-fsycl-explicit-simd' + ' ' +
                                  config.cxx_flags ) )

config.substitutions.append( ('%clangxx', ' '+ config.dpcpp_compiler + ' ' + config.cxx_flags ) )
config.substitutions.append( ('%clang', ' ' + config.dpcpp_compiler + ' ' + config.c_flags ) )
config.substitutions.append( ('%threads_lib', config.sycl_threads_lib) )


# Configure device-specific substitutions based on availability of corresponding
# devices/runtimes

found_at_least_one_device = False

host_run_substitute = "true"
host_run_on_linux_substitute = "true "
host_check_substitute = ""
host_check_on_linux_substitute = ""
supported_device_types=['cpu', 'gpu', 'acc', 'host']

for target_device in config.target_devices.split(','):
    if ( target_device not in supported_device_types ):
        lit_config.error("Unknown SYCL target device type specified '" +
                         target_device +
                         "' supported devices are " + ', '.join(supported_device_types))

if 'host' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Test HOST device")
    host_run_substitute = "env SYCL_DEVICE_TYPE=HOST SYCL_BE={SYCL_BE} ".format(SYCL_BE=config.sycl_be)
    host_check_substitute = "| FileCheck %s"
    config.available_features.add('host')
    if platform.system() == "Linux":
        host_run_on_linux_substitute = "env SYCL_DEVICE_TYPE=HOST SYCL_BE={SYCL_BE} ".format(SYCL_BE=config.sycl_be)
        host_check_on_linux_substitute = "| FileCheck %s"
else:
    lit_config.warning("HOST device not used")

config.substitutions.append( ('%HOST_RUN_PLACEHOLDER',  host_run_substitute) )
config.substitutions.append( ('%HOST_RUN_ON_LINUX_PLACEHOLDER',  host_run_on_linux_substitute) )
config.substitutions.append( ('%HOST_CHECK_PLACEHOLDER',  host_check_substitute) )
config.substitutions.append( ('%HOST_CHECK_ON_LINUX_PLACEHOLDER',  host_check_on_linux_substitute) )

cpu_run_substitute = "true"
cpu_run_on_linux_substitute = "true "
cpu_check_substitute = ""
cpu_check_on_linux_substitute = ""

if 'cpu' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Test CPU device")
    cpu_run_substitute = "env SYCL_DEVICE_TYPE=CPU SYCL_BE={SYCL_BE} ".format(SYCL_BE=config.sycl_be)
    cpu_check_substitute = "| FileCheck %s"
    config.available_features.add('cpu')
    if platform.system() == "Linux":
        cpu_run_on_linux_substitute = "env SYCL_DEVICE_TYPE=CPU SYCL_BE={SYCL_BE} ".format(SYCL_BE=config.sycl_be)
        cpu_check_on_linux_substitute = "| FileCheck %s"
else:
    lit_config.warning("CPU device not used")

config.substitutions.append( ('%CPU_RUN_PLACEHOLDER',  cpu_run_substitute) )
config.substitutions.append( ('%CPU_RUN_ON_LINUX_PLACEHOLDER',  cpu_run_on_linux_substitute) )
config.substitutions.append( ('%CPU_CHECK_PLACEHOLDER',  cpu_check_substitute) )
config.substitutions.append( ('%CPU_CHECK_ON_LINUX_PLACEHOLDER',  cpu_check_on_linux_substitute) )

esimd_run_substitute = "true"
gpu_run_substitute = "true"
gpu_run_on_linux_substitute = "true "
gpu_check_substitute = ""
gpu_check_on_linux_substitute = ""

if 'gpu' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Test GPU device")
    gpu_run_substitute = " env SYCL_DEVICE_TYPE=GPU SYCL_BE={SYCL_BE} ".format(SYCL_BE=config.sycl_be)
    gpu_check_substitute = "| FileCheck %s"
    config.available_features.add('gpu')

    if platform.system() == "Linux":
        gpu_run_on_linux_substitute = "env SYCL_DEVICE_TYPE=GPU SYCL_BE={SYCL_BE} ".format(SYCL_BE=config.sycl_be)
        gpu_check_on_linux_substitute = "| FileCheck %s"

else:
    lit_config.warning("GPU device not used")

config.substitutions.append( ('%GPU_RUN_PLACEHOLDER',  gpu_run_substitute) )
config.substitutions.append( ('%GPU_RUN_ON_LINUX_PLACEHOLDER',  gpu_run_on_linux_substitute) )
config.substitutions.append( ('%GPU_CHECK_PLACEHOLDER',  gpu_check_substitute) )
config.substitutions.append( ('%GPU_CHECK_ON_LINUX_PLACEHOLDER',  gpu_check_on_linux_substitute) )

acc_run_substitute = "true"
acc_check_substitute = ""
# Tests executed with FPGA emu on Windows are not stable
# Disabled until FPGA emulator is fixed
if platform.system() == "Windows":
    lit_config.warning("Accelerator device is disabled on Windows because of instability")
elif 'acc' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Tests accelerator device")
    acc_run_substitute = " env SYCL_DEVICE_TYPE=ACC "
    acc_check_substitute = "| FileCheck %s"
    config.available_features.add('accelerator')
else:
    lit_config.warning("Accelerator device not used")
config.substitutions.append( ('%ACC_RUN_PLACEHOLDER',  acc_run_substitute) )
config.substitutions.append( ('%ACC_CHECK_PLACEHOLDER',  acc_check_substitute) )

if cuda:
    config.substitutions.append( ('%sycl_triple',  "nvptx64-nvidia-cuda-sycldevice" ) )
else:
    config.substitutions.append( ('%sycl_triple',  "spir64-unknown-linux-sycldevice" ) )

if find_executable('sycl-ls'):
    config.available_features.add('sycl-ls')

# Device AOT compilation tools aren't part of the SYCL project,
# so they need to be pre-installed on the machine
aot_tools = ["ocloc", "aoc", "opencl-aot"]

for aot_tool in aot_tools:
    if find_executable(aot_tool) is not None:
        lit_config.note("Found pre-installed AOT device compiler " + aot_tool)
        config.available_features.add(aot_tool)
    else:
        lit_config.warning("Couldn't find pre-installed AOT device compiler " + aot_tool)

# Set timeout for test 1 min
try:
    import psutil
    lit_config.maxIndividualTestTime = 60
except ImportError:
    pass
