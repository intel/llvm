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
        [var,val]=env_pair.split("=", 1)
        if val:
           llvm_config.with_environment(var,val,append_path=True)
           lit_config.note("\t"+var+"="+val)
        else:
           lit_config.note("\tUnset "+var)
           llvm_config.with_environment(var,"")

config.substitutions.append( ('%sycl_libs_dir',  config.sycl_libs_dir ) )
if platform.system() == "Windows":
    config.substitutions.append( ('%sycl_static_libs_dir',  config.sycl_libs_dir + '/../lib' ) )
    config.substitutions.append( ('%obj_ext', '.obj') )
elif platform.system() == "Linux":
    config.substitutions.append( ('%sycl_static_libs_dir',  config.sycl_libs_dir ) )
    config.substitutions.append( ('%obj_ext', '.o') )
config.substitutions.append( ('%sycl_include',  config.sycl_include ) )

# Intel GPU FAMILY availability
if lit_config.params.get('gpu-intel-gen9', False):
    config.available_features.add('gpu-intel-gen9')
if lit_config.params.get('gpu-intel-gen11', False):
    config.available_features.add('gpu-intel-gen11')
if lit_config.params.get('gpu-intel-gen12', False):
    config.available_features.add('gpu-intel-gen12')

# Intel GPU DEVICE availability
if lit_config.params.get('gpu-intel-dg1', False):
    config.available_features.add('gpu-intel-dg1')
if lit_config.params.get('gpu-intel-dg2', False):
    config.available_features.add('gpu-intel-dg2')
if lit_config.params.get('gpu-intel-pvc', False):
    config.available_features.add('gpu-intel-pvc')

if lit_config.params.get('matrix', False):
    config.available_features.add('matrix')

if lit_config.params.get('matrix-xmx8', False):
    config.available_features.add('matrix-xmx8')

#support for LIT parameter ze_debug<num>
if lit_config.params.get('ze_debug'):
    config.ze_debug = lit_config.params.get('ze_debug')
    lit_config.note("ZE_DEBUG: "+config.ze_debug)

# check if compiler supports CL command line options
cl_options=False
sp = subprocess.getstatusoutput(config.dpcpp_compiler+' /help')
if sp[0] == 0:
    cl_options=True
    config.available_features.add('cl_options')

# Check for Level Zero SDK
check_l0_file='l0_include.cpp'
with open(check_l0_file, 'w') as fp:
    fp.write('#include<level_zero/ze_api.h>\n')
    fp.write('int main() { uint32_t t; zeDriverGet(&t,nullptr); return t; }')

config.level_zero_libs_dir=lit_config.params.get("level_zero_libs_dir", config.level_zero_libs_dir)
config.level_zero_include=lit_config.params.get("level_zero_include", (config.level_zero_include if config.level_zero_include else config.sycl_include))

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

# Check for CUDA SDK
check_cuda_file='cuda_include.cpp'
with open(check_cuda_file, 'w') as fp:
    fp.write('#include <cuda.h>\n')
    fp.write('int main() { CUresult r = cuInit(0); return r; }')

config.cuda_libs_dir=lit_config.params.get("cuda_libs_dir", config.cuda_libs_dir)
config.cuda_include=lit_config.params.get("cuda_include", (config.cuda_include if config.cuda_include else config.sycl_include))

cuda_options=cuda_options = (' -L'+config.cuda_libs_dir if config.cuda_libs_dir else '')+' -lcuda '+' -I'+config.cuda_include
if cl_options:
    cuda_options = ' '+( config.cuda_libs_dir+'/cuda.lib ' if config.cuda_libs_dir else 'cuda.lib')+' /I'+config.cuda_include

config.substitutions.append( ('%cuda_options', cuda_options) )

sp = subprocess.getstatusoutput(config.dpcpp_compiler+' -fsycl  ' + check_cuda_file + cuda_options)
if sp[0] == 0:
    config.available_features.add('cuda_dev_kit')
    config.substitutions.append( ('%cuda_options', cuda_options) )
else:
    config.substitutions.append( ('%cuda_options', '') )

# Check for OpenCL ICD
if config.opencl_libs_dir:
    if cl_options:
        config.substitutions.append( ('%opencl_lib',  ' '+config.opencl_libs_dir+'/OpenCL.lib') )
    else:
        config.substitutions.append( ('%opencl_lib',  '-L'+config.opencl_libs_dir+' -lOpenCL') )
    config.available_features.add('opencl_icd')
config.substitutions.append( ('%opencl_include_dir',  config.opencl_include_dir) )

if cl_options:
    config.substitutions.append( ('%sycl_options',  ' ' + config.sycl_libs_dir + '/../lib/sycl6.lib /I' +
                                config.sycl_include + ' /I' + os.path.join(config.sycl_include, 'sycl')) )
    config.substitutions.append( ('%include_option',  '/FI' ) )
    config.substitutions.append( ('%debug_option',  '/DEBUG' ) )
    config.substitutions.append( ('%cxx_std_option',  '/std:' ) )
    config.substitutions.append( ('%fPIC', '') )
    config.substitutions.append( ('%shared_lib', '/LD') )
else:
    config.substitutions.append( ('%sycl_options',
                                  (' -lsycl6' if platform.system() == "Windows" else " -lsycl") + ' -I' +
                                  config.sycl_include + ' -I' + os.path.join(config.sycl_include, 'sycl') +
                                  ' -L' + config.sycl_libs_dir) )
    config.substitutions.append( ('%include_option',  '-include' ) )
    config.substitutions.append( ('%debug_option',  '-g' ) )
    config.substitutions.append( ('%cxx_std_option',  '-std=' ) )
    # Position-independent code does not make sence on Windows. At the same
    # time providing this option for compilation targeting 
    # x86_64-pc-windows-msvc will cause compile time error on some
    # configurations
    config.substitutions.append( ('%fPIC', ('' if platform.system() == 'Windows' else '-fPIC')) )
    config.substitutions.append( ('%shared_lib', '-shared') )

if not config.gpu_aot_target_opts:
    config.gpu_aot_target_opts = '"-device *"'

config.substitutions.append( ('%gpu_aot_target_opts',  config.gpu_aot_target_opts ) )

if not config.sycl_be:
     lit_config.error("SYCL backend is not specified")

# Transforming from SYCL_BE backend definition style to SYCL_DEVICE_FILTER used
# for backward compatibility : e.g. 'PI_ABC_XYZ' -> 'abc_xyz'
if config.sycl_be.startswith("PI_"):
    config.sycl_be = config.sycl_be[3:]
config.sycl_be = config.sycl_be.lower()

# Replace deprecated backend names
deprecated_names_mapping = {'cuda'       : 'ext_oneapi_cuda',
                            'hip'        : 'ext_oneapi_hip',
                            'level_zero' : 'ext_oneapi_level_zero',
                            'esimd_cpu'  : 'ext_intel_esimd_emulator'}
if config.sycl_be in deprecated_names_mapping.keys():
    config.sycl_be = deprecated_names_mapping[config.sycl_be]

lit_config.note("Backend: {BACKEND}".format(BACKEND=config.sycl_be))

config.substitutions.append( ('%sycl_be', config.sycl_be) )
# Use short names for LIT rules
config.available_features.add(config.sycl_be.replace('ext_intel_', '').replace('ext_oneapi_', ''))
config.substitutions.append( ('%BE_RUN_PLACEHOLDER', "env ONEAPI_DEVICE_SELECTOR='{SYCL_PLUGIN}:* '".format(SYCL_PLUGIN=config.sycl_be)) )

if config.dump_ir_supported:
   config.available_features.add('dump_ir')

supported_sycl_be = ['opencl',
                     'ext_oneapi_cuda',
                     'ext_oneapi_hip',
                     'ext_oneapi_level_zero',
                     'ext_intel_esimd_emulator']

if config.sycl_be not in supported_sycl_be:
   lit_config.error("Unknown SYCL BE specified '" +
                    config.sycl_be +
                    "'. Supported values are {}".format(', '.join(supported_sycl_be)))

# Run only tests in ESIMD subforlder for the ext_intel_esimd_emulator
if config.sycl_be == 'ext_intel_esimd_emulator':
   config.test_source_root += "/ESIMD"
   config.test_exec_root += "/ESIMD"

# If HIP_PLATFORM flag is not set, default to AMD, and check if HIP platform is supported
supported_hip_platforms=["AMD", "NVIDIA"]
if config.hip_platform == "":
    config.hip_platform = "AMD"
if config.hip_platform not in supported_hip_platforms:
    lit_config.error("Unknown HIP platform '" + config.hip_platform + "' supported platforms are " + ', '.join(supported_hip_platforms))

if config.sycl_be == "ext_oneapi_hip" and config.hip_platform == "AMD":
    config.available_features.add('hip_amd')
    arch_flag = '-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=' + config.amd_arch
elif config.sycl_be == "ext_oneapi_hip" and config.hip_platform == "NVIDIA":
    config.available_features.add('hip_nvidia')
    arch_flag = ""
else:
    arch_flag = ""

if lit_config.params.get('compatibility_testing', False):
    config.substitutions.append( ('%clangxx', ' true ') )
    config.substitutions.append( ('%clang', ' true ') )
else:
    config.substitutions.append( ('%clangxx', ' '+ config.dpcpp_compiler + ' ' + config.cxx_flags + ' ' + arch_flag) )
    config.substitutions.append( ('%clang', ' ' + config.dpcpp_compiler + ' ' + config.c_flags) )

config.substitutions.append( ('%threads_lib', config.sycl_threads_lib) )

# Configure device-specific substitutions based on availability of corresponding
# devices/runtimes

found_at_least_one_device = False

supported_device_types=['cpu', 'gpu', 'acc']

for target_device in config.target_devices.split(','):
    if target_device == 'host':
        lit_config.warning("Host device type is no longer supported.")
    elif ( target_device not in supported_device_types ):
        lit_config.error("Unknown SYCL target device type specified '" +
                         target_device +
                         "' supported devices are " + ', '.join(supported_device_types))

cpu_run_substitute = "true"
cpu_run_on_linux_substitute = "true "
cpu_check_substitute = ""
cpu_check_on_linux_substitute = ""

if 'cpu' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Test CPU device")
    cpu_run_substitute = "env ONEAPI_DEVICE_SELECTOR={SYCL_PLUGIN}:cpu ".format(SYCL_PLUGIN=config.sycl_be)
    cpu_check_substitute = "| FileCheck %s"
    config.available_features.add('cpu')
    if platform.system() == "Linux":
        cpu_run_on_linux_substitute = cpu_run_substitute
        cpu_check_on_linux_substitute = "| FileCheck %s"

    if config.run_launcher:
        cpu_run_substitute += " {}".format(config.run_launcher)
else:
    lit_config.warning("CPU device not used")

config.substitutions.append( ('%CPU_RUN_PLACEHOLDER',  cpu_run_substitute) )
config.substitutions.append( ('%CPU_RUN_ON_LINUX_PLACEHOLDER',  cpu_run_on_linux_substitute) )
config.substitutions.append( ('%CPU_CHECK_PLACEHOLDER',  cpu_check_substitute) )
config.substitutions.append( ('%CPU_CHECK_ON_LINUX_PLACEHOLDER',  cpu_check_on_linux_substitute) )

gpu_run_substitute = "true"
gpu_run_on_linux_substitute = "true "
gpu_check_substitute = ""
gpu_l0_check_substitute = ""
gpu_check_on_linux_substitute = ""

if 'gpu' in config.target_devices.split(','):
    found_at_least_one_device = True
    lit_config.note("Test GPU device")
    gpu_run_substitute = " env ONEAPI_DEVICE_SELECTOR={SYCL_PLUGIN}:gpu ".format(SYCL_PLUGIN=config.sycl_be)
    gpu_check_substitute = "| FileCheck %s"
    config.available_features.add('gpu')

    if config.sycl_be == "ext_oneapi_level_zero":
        gpu_l0_check_substitute = "| FileCheck %s"
        if lit_config.params.get('ze_debug'):
            gpu_run_substitute = " env ZE_DEBUG={ZE_DEBUG} ONEAPI_DEVICE_SELECTOR=level_zero:gpu ".format(ZE_DEBUG=config.ze_debug)
            config.available_features.add('ze_debug'+config.ze_debug)
    elif config.sycl_be == "ext_intel_esimd_emulator":
        # ESIMD_EMULATOR backend uses CM_EMU library package for
        # multi-threaded execution on CPU, and the package emulates
        # multiple target platforms. In case user does not specify
        # what target platform to emulate, 'skl' is chosen by default.
        if not "CM_RT_PLATFORM" in os.environ:
            gpu_run_substitute += "CM_RT_PLATFORM=skl "

    if platform.system() == "Linux":
        gpu_run_on_linux_substitute = "env ONEAPI_DEVICE_SELECTOR={SYCL_PLUGIN}:gpu ".format(SYCL_PLUGIN=config.sycl_be)
        gpu_check_on_linux_substitute = "| FileCheck %s"

    if config.sycl_be == "ext_oneapi_cuda":
        gpu_run_substitute += "SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT=1 "

    if config.run_launcher:
        gpu_run_substitute += " {}".format(config.run_launcher)
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
    acc_run_substitute = " env ONEAPI_DEVICE_SELECTOR='*:acc' "
    acc_check_substitute = "| FileCheck %s"
    config.available_features.add('accelerator')

    if config.run_launcher:
        acc_run_substitute += " {}".format(config.run_launcher)
else:
    lit_config.warning("Accelerator device not used")
config.substitutions.append( ('%ACC_RUN_PLACEHOLDER',  acc_run_substitute) )
config.substitutions.append( ('%ACC_CHECK_PLACEHOLDER',  acc_check_substitute) )

if config.run_launcher:
    config.substitutions.append(('%e2e_tests_root', config.test_source_root))
    config.recursiveExpansionLimit = 10

if config.sycl_be == 'ext_oneapi_cuda' or (config.sycl_be == 'ext_oneapi_hip' and config.hip_platform == 'NVIDIA'):
    config.substitutions.append( ('%sycl_triple',  "nvptx64-nvidia-cuda" ) )
elif config.sycl_be == 'ext_oneapi_hip' and config.hip_platform == 'AMD':
    config.substitutions.append( ('%sycl_triple',  "amdgcn-amd-amdhsa" ) )
else:
    config.substitutions.append( ('%sycl_triple',  "spir64" ) )

if find_executable('sycl-ls'):
    config.available_features.add('sycl-ls')

# TODO properly set XPTIFW include and runtime dirs
xptifw_lib_dir = os.path.join(config.dpcpp_root_dir, 'lib')
xptifw_dispatcher = ""
if platform.system() == "Linux":
    xptifw_dispatcher = os.path.join(xptifw_lib_dir, 'libxptifw.so')
elif platform.system() == "Windows":
    xptifw_dispatcher = os.path.join(config.dpcpp_root_dir, 'bin', 'xptifw.dll')
xptifw_includes = os.path.join(config.dpcpp_root_dir, 'include')
if os.path.exists(xptifw_lib_dir) and os.path.exists(os.path.join(xptifw_includes, 'xpti', 'xpti_trace_framework.h')):
    config.available_features.add('xptifw')
    config.substitutions.append(('%xptifw_dispatcher', xptifw_dispatcher))
    if cl_options:
        config.substitutions.append(('%xptifw_lib', " {}/xptifw.lib /I{} ".format(xptifw_lib_dir, xptifw_includes)))
    else:
        config.substitutions.append(('%xptifw_lib', " -L{} -lxptifw -I{} ".format(xptifw_lib_dir, xptifw_includes)))


llvm_tools = ["llvm-spirv", "llvm-link"]
for llvm_tool in llvm_tools:
  llvm_tool_path = find_executable(llvm_tool)
  if llvm_tool_path:
    lit_config.note("Found " + llvm_tool)
    config.available_features.add(llvm_tool)
    config.substitutions.append( ('%' + llvm_tool.replace('-', '_'),
                                  os.path.realpath(llvm_tool_path)) )
  else:
    lit_config.warning("Can't find " + llvm_tool)

if find_executable('cmc'):
    config.available_features.add('cm-compiler')

# Device AOT compilation tools aren't part of the SYCL project,
# so they need to be pre-installed on the machine
aot_tools = ["ocloc", "opencl-aot"]

for aot_tool in aot_tools:
    if find_executable(aot_tool) is not None:
        lit_config.note("Found pre-installed AOT device compiler " + aot_tool)
        config.available_features.add(aot_tool)
    else:
        lit_config.warning("Couldn't find pre-installed AOT device compiler " + aot_tool)

# Check if kernel fusion is available by compiling a small program that will
# be ill-formed (compilation stops with non-zero exit code) if the feature
# test macro for kernel fusion is not defined.
check_fusion_file = 'check_fusion.cpp'
with open(check_fusion_file, 'w') as ff:
    ff.write('#include <sycl/sycl.hpp>\n')
    ff.write('#ifndef SYCL_EXT_CODEPLAY_KERNEL_FUSION\n')
    ff.write('#error \"Feature test for fusion failed\"\n')
    ff.write('#endif // SYCL_EXT_CODEPLAY_KERNEL_FUSION\n')
    ff.write('int main() { return 0; }\n')

status = subprocess.getstatusoutput(config.dpcpp_compiler + ' -fsycl  ' +
                                    check_fusion_file)
if status[0] == 0:
    lit_config.note('Kernel fusion extension enabled')
    config.available_features.add('fusion')

# Set timeout for a single test
try:
    import psutil
    lit_config.maxIndividualTestTime = 600
except ImportError:
    pass
