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
config.name = 'SYCL-on-device'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.dump'] #add .spv. Currently not clear what to do with those

# feature tests are considered not so lightweight, so, they are excluded by default
config.excludes = ['Inputs', 'feature-tests']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.sycl_obj_root, 'test')

llvm_config.use_clang(use_installed=True,
                      additional_flags=config.sycl_clang_extra_flags.split(' '))

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['PATH', 'OCL_ICD_FILENAMES', 'SYCL_DEVICE_ALLOWLIST', 'SYCL_CONFIG_FILE_NAME'])

# Propagate extra environment variables
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

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
if platform.system() == "Linux":
    config.available_features.add('linux')
    llvm_config.with_system_environment(['LD_LIBRARY_PATH','LIBRARY_PATH','CPATH'])
    llvm_config.with_environment('LD_LIBRARY_PATH', config.sycl_libs_dir, append_path=True)

elif platform.system() == "Windows":
    config.available_features.add('windows')
    llvm_config.with_system_environment(['LIB','CPATH','INCLUDE'])
    llvm_config.with_environment('LIB', config.sycl_libs_dir, append_path=True)

elif platform.system() == "Darwin":
    # FIXME: surely there is a more elegant way to instantiate the Xcode directories.
    llvm_config.with_system_environment('CPATH')
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1", append_path=True)
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/", append_path=True)
    llvm_config.with_environment('DYLD_LIBRARY_PATH', config.sycl_libs_dir)

llvm_config.with_environment('PATH', config.sycl_tools_dir, append_path=True)

config.substitutions.append( ('%threads_lib', config.sycl_threads_lib) )
config.substitutions.append( ('%sycl_libs_dir',  config.sycl_libs_dir ) )
config.substitutions.append( ('%sycl_include',  config.sycl_include ) )
config.substitutions.append( ('%sycl_source_dir', config.sycl_source_dir) )
config.substitutions.append( ('%opencl_libs_dir',  config.opencl_libs_dir) )
config.substitutions.append( ('%opencl_include_dir',  config.opencl_include_dir) )
config.substitutions.append( ('%level_zero_include_dir',  config.level_zero_include_dir) )
config.substitutions.append( ('%cuda_toolkit_include',  config.cuda_toolkit_include) )
config.substitutions.append( ('%sycl_tools_src_dir',  config.sycl_tools_src_dir ) )
config.substitutions.append( ('%llvm_build_lib_dir',  config.llvm_build_lib_dir ) )
config.substitutions.append( ('%llvm_build_bin_dir',  config.llvm_build_bin_dir ) )

if config.level_zero_include_dir:
    config.available_features.add("level_zero_headers")
else:
    lit_config.warning("Level_Zero headers path is not configured. Dependent tests are skipped.")


llvm_config.add_tool_substitutions(['llvm-spirv'], [config.sycl_tools_dir])
backend=lit_config.params.get('SYCL_PLUGIN', "opencl")
lit_config.note("Backend: {}".format(backend))
config.substitutions.append( ('%sycl_be', backend) )
config.substitutions.append( ('%BE_RUN_PLACEHOLDER', "env SYCL_DEVICE_FILTER={SYCL_PLUGIN} ".format(SYCL_PLUGIN=backend)) )
config.substitutions.append( ('%RUN_ON_HOST', "env SYCL_DEVICE_FILTER=host ") )

get_device_count_by_type_path = lit_config.params.get('GET_DEVICE_TOOL', os.path.join(config.llvm_tools_dir, "get_device_count_by_type"))
if 'GET_DEVICE_TOOL' in lit_config.params.keys():
    lit_config.warning("The tool from none-default path is used: "+get_device_count_by_type_path)

def getDeviceCount(device_type):
    is_cuda = False;
    is_level_zero = False;
    is_rocm_amd = False;
    is_rocm_nvidia = False;
    process = subprocess.Popen([get_device_count_by_type_path, device_type, backend],
        stdout=subprocess.PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()

    if exit_code != 0:
        lit_config.error("getDeviceCount {TYPE} {BACKEND}: Non-zero exit code {CODE}".format(
            TYPE=device_type, BACKEND=backend, CODE=exit_code))
        return [0,False,False]

    result = output.decode().replace('\n', '').split(':', 1)
    try:
        value = int(result[0])
    except ValueError:
        value = 0
        lit_config.error("getDeviceCount {TYPE} {BACKEND}: Cannot get value from output: {OUT}".format(
            TYPE=device_type, BACKEND=backend, OUT=result[0]))

    # if we have found gpu and there is additional information, let's check
    # whether this is CUDA device or Level Zero device or none of these.
    if device_type == "gpu" and value > 0 and len(result[1]):
        if re.match(r".*cuda", result[1]):
            is_cuda = True;
        if re.match(r".*level zero", result[1]):
            is_level_zero = True;
        if re.match(r".*rocm-amd", result[1]):
            is_rocm_amd = True;
        if re.match(r".*rocm-nvidia", result[1]):
            is_rocm_nvidia = True;

    if err:
        lit_config.warning("getDeviceCount {TYPE} {BACKEND} stderr:{ERR}".format(
            TYPE=device_type, BACKEND=backend, ERR=err))
    return [value,is_cuda,is_level_zero,is_rocm_amd,is_rocm_nvidia]

# check if compiler supports CL command line options
cl_options=False
sp = subprocess.getstatusoutput(config.clang + ' /help')
if sp[0] == 0:
    cl_options=True
    config.available_features.add('cl_options')

check_l0_file='l0_include.cpp'
with open(check_l0_file, 'w') as fp:
    fp.write('#include<level_zero/ze_api.h>\n')
    fp.write('int main() { uint32_t t; zeDriverGet(&t,nullptr); return t; }')

config.level_zero_libs_dir=lit_config.params.get("LEVEL_ZERO_LIBS_DIR", config.level_zero_libs_dir)
config.level_zero_include=lit_config.params.get("LEVEL_ZERO_INCLUDE_DIR", (config.level_zero_include if config.level_zero_include else os.path.join(config.sycl_include, '..')))

level_zero_options=level_zero_options = (' -L'+config.level_zero_libs_dir if config.level_zero_libs_dir else '')+' -lze_loader '+' -I'+config.level_zero_include
if cl_options:
    level_zero_options = ' '+( config.level_zero_libs_dir+'/ze_loader.lib ' if config.level_zero_libs_dir else 'ze_loader.lib')+' /I'+config.level_zero_include

config.substitutions.append( ('%level_zero_options', level_zero_options) )

sp = subprocess.getstatusoutput(config.clang + ' -fsycl  ' + check_l0_file + level_zero_options)
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

# Every SYCL implementation provides a host implementation.
config.available_features.add('host')

# Configure device-specific substitutions based on availability of corresponding
# devices/runtimes

found_at_least_one_device = False

cpu_run_substitute = "echo "
cpu_run_on_linux_substitute = "echo "
cpu_check_substitute = ""
cpu_check_on_linux_substitute = ""

if getDeviceCount("cpu")[0]:
    found_at_least_one_device = True
    lit_config.note("Found available CPU device")
    cpu_run_substitute = "env SYCL_DEVICE_FILTER=cpu,host "
    cpu_check_substitute = "| FileCheck %s"
    config.available_features.add('cpu')
    if platform.system() == "Linux":
        cpu_run_on_linux_substitute = "env SYCL_DEVICE_FILTER=cpu,host "
        cpu_check_on_linux_substitute = "| FileCheck %s"
else:
    lit_config.warning("CPU device not found")

config.substitutions.append( ('%CPU_RUN_PLACEHOLDER',  cpu_run_substitute) )
config.substitutions.append( ('%CPU_RUN_ON_LINUX_PLACEHOLDER',  cpu_run_on_linux_substitute) )
config.substitutions.append( ('%CPU_CHECK_PLACEHOLDER',  cpu_check_substitute) )
config.substitutions.append( ('%CPU_CHECK_ON_LINUX_PLACEHOLDER',  cpu_check_on_linux_substitute) )

gpu_run_substitute = "echo "
gpu_run_on_linux_substitute = "echo "
gpu_check_substitute = ""
gpu_check_on_linux_substitute = ""

cuda = False
level_zero = False
rocm_amd = False
rocm_nvidia = False
[gpu_count, cuda, level_zero, rocm_amd, rocm_nvidia] = getDeviceCount("gpu")

if gpu_count > 0:
    found_at_least_one_device = True
    lit_config.note("Found available GPU device")
    gpu_run_substitute = " env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:gpu,host ".format(SYCL_PLUGIN=backend)
    gpu_check_substitute = "| FileCheck %s"
    config.available_features.add('gpu')
    if cuda:
       config.available_features.add('cuda')
    elif level_zero:
       config.available_features.add('level_zero')
    elif rocm_amd:
       config.available_features.add('rocm_amd')
       # For AMD the specific GPU has to be specified with -mcpu
       if not re.match('.*-mcpu.*', config.sycl_clang_extra_flags):
           raise Exception("Error: missing -mcpu flag when trying to run lit "  \
                           "tests for AMD GPU, please add `-mcpu=<target>` to " \
                           "the CMake variable SYCL_CLANG_EXTRA_FLAGS")
    elif rocm_nvidia:
       config.available_features.add('rocm_nvidia')

    if platform.system() == "Linux":
        gpu_run_on_linux_substitute = "env SYCL_DEVICE_FILTER={SYCL_PLUGIN}:gpu,host ".format(SYCL_PLUGIN=backend)
        gpu_check_on_linux_substitute = "| FileCheck %s"
else:
    lit_config.warning("GPU device not found")

config.substitutions.append( ('%GPU_RUN_PLACEHOLDER',  gpu_run_substitute) )
config.substitutions.append( ('%GPU_RUN_ON_LINUX_PLACEHOLDER',  gpu_run_on_linux_substitute) )
config.substitutions.append( ('%GPU_CHECK_PLACEHOLDER',  gpu_check_substitute) )
config.substitutions.append( ('%GPU_CHECK_ON_LINUX_PLACEHOLDER',  gpu_check_on_linux_substitute) )

acc_run_substitute = "echo "
acc_check_substitute = ""
if getDeviceCount("accelerator")[0]:
    found_at_least_one_device = True
    lit_config.note("Found available accelerator device")
    acc_run_substitute = " env SYCL_DEVICE_FILTER=acc "
    acc_check_substitute = "| FileCheck %s"
    config.available_features.add('accelerator')
else:
    lit_config.warning("Accelerator device not found")
config.substitutions.append( ('%ACC_RUN_PLACEHOLDER',  acc_run_substitute) )
config.substitutions.append( ('%ACC_CHECK_PLACEHOLDER',  acc_check_substitute) )

# LIT testing either supports OpenCL or CUDA or Level Zero.
if not cuda and not level_zero and found_at_least_one_device:
    config.available_features.add('opencl')

if cuda or rocm_nvidia:
    config.substitutions.append( ('%sycl_triple',  "nvptx64-nvidia-cuda-sycldevice" ) )
elif rocm_amd:
    config.substitutions.append( ('%sycl_triple',  "amdgcn-amd-amdhsa-sycldevice" ) )
else:
    config.substitutions.append( ('%sycl_triple',  "spir64-unknown-unknown-sycldevice" ) )

if "opencl-aot" in config.llvm_enable_projects:
    lit_config.note("Using opencl-aot version which is built as part of the project")
    config.available_features.add("opencl-aot")
    llvm_config.add_tool_substitutions(['opencl-aot'], [config.sycl_tools_dir])

# Device AOT compilation tools aren't part of the SYCL project,
# so they need to be pre-installed on the machine
aot_tools = ["ocloc", "aoc"]
if "opencl-aot" not in config.llvm_enable_projects:
    aot_tools.append('opencl-aot')

for aot_tool in aot_tools:
    if find_executable(aot_tool) is not None:
        lit_config.note("Found pre-installed AOT device compiler " + aot_tool)
        config.available_features.add(aot_tool)
    else:
        lit_config.warning("Couldn't find pre-installed AOT device compiler " + aot_tool)

# Set timeout for test = 10 mins
try:
    import psutil
    lit_config.maxIndividualTestTime = 600
except ImportError:
    pass
