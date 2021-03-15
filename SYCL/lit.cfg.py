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

# Cleanup environment variables which may affect tests
possibly_dangerous_env_vars = ['COMPILER_PATH', 'RC_DEBUG_OPTIONS',
                               'CINDEXTEST_PREAMBLE_FILE', 'LIBRARY_PATH',
                               'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH',
                               'OBJC_INCLUDE_PATH', 'OBJCPLUS_INCLUDE_PATH',
                               'LIBCLANG_TIMING', 'LIBCLANG_OBJTRACKING',
                               'LIBCLANG_LOGGING', 'LIBCLANG_BGPRIO_INDEX',
                               'LIBCLANG_BGPRIO_EDIT', 'LIBCLANG_NOTHREADS',
                               'LIBCLANG_RESOURCE_USAGE',
                               'LIBCLANG_CODE_COMPLETION_LOGGING']
# Clang/Win32 may refer to %INCLUDE%. vsvarsall.bat sets it.
if platform.system() != 'Windows':
    possibly_dangerous_env_vars.append('INCLUDE')

for name in possibly_dangerous_env_vars:
    if name in llvm_config.config.environment:
        del llvm_config.config.environment[name]

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['PATH', 'OCL_ICD_FILENAMES',
    'CL_CONFIG_DEVICES', 'SYCL_DEVICE_ALLOWLIST', 'SYCL_CONFIG_FILE_NAME'])

llvm_config.with_environment('PATH', config.lit_tools_dir, append_path=True)

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
if platform.system() == "Linux":
    config.available_features.add('linux')
    llvm_config.with_system_environment(['LD_LIBRARY_PATH','LIBRARY_PATH','CPATH'])
    llvm_config.with_environment('LD_LIBRARY_PATH', config.sycl_libs_dir, append_path=True)

elif platform.system() == "Windows":
    config.available_features.add('windows')
    llvm_config.with_system_environment(['LIB','CPATH','INCLUDE'])
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

if config.extra_environment:
    lit_config.note("Extra environment variables")
    for env_pair in config.extra_environment.split(','):
        [var,val]=env_pair.split("=")
        if val:
           llvm_config.with_environment(var,val)
           lit_config.note("\t"+var+"="+val)
        else:
           lit_config.note("\tUnset "+var)
           llvm_config.with_environment(var,"")

config.substitutions.append( ('%sycl_libs_dir',  config.sycl_libs_dir ) )
config.substitutions.append( ('%sycl_include',  config.sycl_include ) )

if lit_config.params.get('gpu-intel-dg1', False):
    config.available_features.add('gpu-intel-dg1')

# check if compiler supports CL command line options
cl_options=False
sp = subprocess.getstatusoutput(config.dpcpp_compiler+' /help')
if sp[0] == 0:
    cl_options=True
    config.available_features.add('cl_options')

check_l0_file='l0_include.cpp'
with open(check_l0_file, 'w') as fp:
    fp.write('#include<level_zero/ze_api.h>\n')
    fp.write('int main() { uint32_t t; zeDriverGet(&t,nullptr); return t; }')

config.level_zero_libs_dir=lit_config.params.get("level_zero_libs_dir", config.level_zero_libs_dir)
config.level_zero_include=lit_config.params.get("level_zero_include", (config.level_zero_include if config.level_zero_include else os.path.join(config.sycl_include, '..')))

level_zero_options=level_zero_options = (' -L'+config.level_zero_libs_dir if config.level_zero_libs_dir else '')+' -lze_loader '+' -I'+config.level_zero_include
if cl_options:
    level_zero_options = ' '+( config.level_zero_libs_dir+'/ze_loader.lib ' if config.level_zero_libs_dir else 'ze_loader.lib')+' /I'+config.level_zero_include

config.substitutions.append( ('%level_zero_options', level_zero_options) )

sp = subprocess.getstatusoutput(config.dpcpp_compiler+' -fsycl  ' + check_l0_file + level_zero_options)
if sp[0] == 0:
    config.available_features.add('level_zero_dev_kit')
    config.substitutions.append( ('%level_zero_options', level_zero_options) )
else:
    config.substitutions.append( ('%level_zero_options', '') )

if config.opencl_libs_dir:
    if cl_options:
        config.substitutions.append( ('%opencl_lib',  ' '+config.opencl_libs_dir+'/OpenCL.lib') )
    else:
        config.substitutions.append( ('%opencl_lib',  '-L'+config.opencl_libs_dir+' -lOpenCL') )
    config.available_features.add('opencl_icd')
config.substitutions.append( ('%opencl_include_dir',  config.opencl_include_dir) )

if cl_options:
    config.substitutions.append( ('%sycl_options',  ' sycl.lib /I'+config.sycl_include ) )
    config.substitutions.append( ('%include_option',  '/FI' ) )
    config.substitutions.append( ('%debug_option',  '/DEBUG' ) )
    config.substitutions.append( ('%cxx_std_option',  '/std:' ) )
else:
    config.substitutions.append( ('%sycl_options', ' -lsycl -I'+config.sycl_include ) )
    config.substitutions.append( ('%include_option',  '-include' ) )
    config.substitutions.append( ('%debug_option',  '-g' ) )
    config.substitutions.append( ('%cxx_std_option',  '-std=' ) )

llvm_config.add_tool_substitutions(['llvm-spirv'], [config.sycl_tools_dir])

if not config.sycl_be:
     lit_config.error("SYCL backend is not specified")

# Mapping from SYCL_BE backend definition style to SYCL_DEVICE_FILTER used
# for backward compatibility
try:
  config.sycl_be = { 'PI_OPENCL': 'opencl',  'PI_CUDA': 'cuda', 'PI_LEVEL_ZERO': 'level_zero'}[config.sycl_be]
except:
  # do nothing a we expect that new format of plugin values are used
  pass

lit_config.note("Backend: {BACKEND}".format(BACKEND=config.sycl_be))

config.substitutions.append( ('%sycl_be', config.sycl_be) )
config.available_features.add(config.sycl_be)
config.substitutions.append( ('%BE_RUN_PLACEHOLDER', "env SYCL_DEVICE_FILTER={SYCL_PLUGIN} ".format(SYCL_PLUGIN=config.sycl_be)) )

if config.dump_ir_supported:
   config.available_features.add('dump_ir')

if config.sycl_be not in ['host', 'opencl','cuda', 'level_zero']:
    lit_config.error("Unknown SYCL BE specified '" +
                     config.sycl_be +
                     "' supported values are opencl, cuda, level_zero")

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
    host_run_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:host ".format(SYCL_PLUGIN=config.sycl_be)
    host_check_substitute = "| FileCheck %s"
    config.available_features.add('host')
    if platform.system() == "Linux":
        host_run_on_linux_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:host ".format(SYCL_PLUGIN=config.sycl_be)
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
    cpu_run_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:cpu ".format(SYCL_PLUGIN=config.sycl_be)
    cpu_check_substitute = "| FileCheck %s"
    config.available_features.add('cpu')
    if platform.system() == "Linux":
        cpu_run_on_linux_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:cpu ".format(SYCL_PLUGIN=config.sycl_be)
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
gpu_l0_check_substitute = ""
gpu_check_on_linux_substitute = ""

if 'gpu' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Test GPU device")
    gpu_run_substitute = " env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:gpu ".format(SYCL_PLUGIN=config.sycl_be)
    gpu_check_substitute = "| FileCheck %s"
    config.available_features.add('gpu')

    if config.sycl_be == "level_zero":
        gpu_l0_check_substitute = "| FileCheck %s"

    if platform.system() == "Linux":
        gpu_run_on_linux_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:gpu ".format(SYCL_PLUGIN=config.sycl_be)
        gpu_check_on_linux_substitute = "| FileCheck %s"

else:
    lit_config.warning("GPU device not used")

config.substitutions.append( ('%GPU_RUN_PLACEHOLDER',  gpu_run_substitute) )
config.substitutions.append( ('%GPU_RUN_ON_LINUX_PLACEHOLDER',  gpu_run_on_linux_substitute) )
config.substitutions.append( ('%GPU_CHECK_PLACEHOLDER',  gpu_check_substitute) )
config.substitutions.append( ('%GPU_L0_CHECK_PLACEHOLDER',  gpu_l0_check_substitute) )
config.substitutions.append( ('%GPU_CHECK_ON_LINUX_PLACEHOLDER',  gpu_check_on_linux_substitute) )

acc_run_substitute = "true"
acc_check_substitute = ""
if 'acc' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Tests accelerator device")
    acc_run_substitute = " env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:acc ".format(SYCL_PLUGIN=config.sycl_be)
    acc_check_substitute = "| FileCheck %s"
    config.available_features.add('accelerator')
else:
    lit_config.warning("Accelerator device not used")
config.substitutions.append( ('%ACC_RUN_PLACEHOLDER',  acc_run_substitute) )
config.substitutions.append( ('%ACC_CHECK_PLACEHOLDER',  acc_check_substitute) )

if config.sycl_be == 'cuda':
    config.substitutions.append( ('%sycl_triple',  "nvptx64-nvidia-cuda-sycldevice" ) )
else:
    config.substitutions.append( ('%sycl_triple',  "spir64-unknown-unknown-sycldevice" ) )

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
    lit_config.maxIndividualTestTime = 600
except ImportError:
    pass
