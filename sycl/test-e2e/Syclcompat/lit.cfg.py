# -*- Python -*-

import os
import platform
import copy
import subprocess

import lit.util
import lit.formats

from lit.llvm import llvm_config
from lit.llvm.subst import FindTool

# name: The name of this test suite.
config.name = 'SYCL-Unit-syclcompat'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.sycl_obj_root
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest('.', 'Tests')

# allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

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
    llvm_config.with_system_environment(['LD_LIBRARY_PATH','LIBRARY_PATH','CPATH'])
    llvm_config.with_environment('LD_LIBRARY_PATH', config.sycl_libs_dir, append_path=True)

elif platform.system() == "Windows":
    llvm_config.with_system_environment(['LIB','CPATH','INCLUDE'])
    llvm_config.with_environment('LIB', config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment('PATH', config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment('LIB', os.path.join(config.dpcpp_root_dir, 'lib'), append_path=True)

elif platform.system() == "Darwin":
    llvm_config.with_system_environment('CPATH')
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1", append_path=True)
    llvm_config.with_environment('CPATH', "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/", append_path=True)
    llvm_config.with_environment('DYLD_LIBRARY_PATH', config.sycl_libs_dir)

llvm_config.with_environment('PATH', config.sycl_tools_dir, append_path=True)

lit_config.note("Targeted devices: {}".format(', '.join(config.sycl_devices)))

sycl_ls = FindTool('sycl-ls').resolve(llvm_config, config.llvm_tools_dir)
if not sycl_ls:
    lit_config.fatal("can't find `sycl-ls`")

if len(config.sycl_devices) == 1 and config.sycl_devices[0] == 'all':
    devices = set()
    sp = subprocess.check_output(sycl_ls, text=True)
    for line in sp.splitlines():
        (backend, device, _) = line[1:].split(':', 2)
        devices.add('{}:{}'.format(backend, device))
    config.sycl_devices = list(devices)

lit_config.note("Expanded devices: {}".format(', '.join(config.sycl_devices)))

available_devices = {'opencl': ('cpu', 'gpu', 'acc'),
                     'ext_oneapi_cuda':('gpu'),
                     'ext_oneapi_level_zero':('gpu'),
                     'ext_oneapi_hip':('gpu'),
                     'ext_intel_esimd_emulator':('gpu'),
                     'native_cpu':('cpu')}

for d in config.sycl_devices:
     be, dev = d.split(':')
     if be not in available_devices or dev not in available_devices[be]:
          lit_config.error('Unsupported device {}'.format(d))

llvm_config.with_environment('ONEAPI_DEVICE_SELECTOR', ';'.join(config.sycl_devices))
lit_config.note("ONEAPI_DEVICE_SELECTOR={}".format(';'.join(config.sycl_devices)))

# If HIP_PLATFORM flag is not set, default to AMD, and check if HIP platform is supported
supported_hip_platforms=["AMD", "NVIDIA"]
if config.hip_platform == "":
    config.hip_platform = "AMD"
if config.hip_platform not in supported_hip_platforms:
    lit_config.error("Unknown HIP platform '" + config.hip_platform + "' supported platforms are " + ', '.join(supported_hip_platforms))

# We use this to simply detect errors in sycl devices
config.sycl_dev_features = {}
for sycl_device in config.sycl_devices:
    env = copy.copy(llvm_config.config.environment)
    env['ONEAPI_DEVICE_SELECTOR'] = sycl_device
    if sycl_device.startswith('ext_oneapi_cuda:'):
        env['SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT'] = '1'
    # When using the ONEAPI_DEVICE_SELECTOR environment variable, sycl-ls
    # prints warnings that might derail a user thinking something is wrong
    # with their test run. It's just us filtering here, so silence them unless
    # we get an exit status.
    try:
        sp = subprocess.run([sycl_ls, '--verbose'], env=env, text=True,
                            capture_output=True)
        sp.check_returncode()
    except subprocess.CalledProcessError as e:
        # capturing e allows us to see path resolution errors / system
        # permissions errors etc
        lit_config.fatal(f'Cannot list device aspects for {sycl_device}\n'
                         f'{e}\n'
                         f'stdout:{sp.stdout}\n'
                         f'stderr:{sp.stderr}\n')


# Set timeout for a single test
try:
    import psutil
    lit_config.maxIndividualTestTime = 600
except ImportError:
    pass
