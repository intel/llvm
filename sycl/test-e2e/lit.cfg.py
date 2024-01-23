# -*- Python -*-

import os
import platform
import copy
import re
import subprocess
import tempfile
import textwrap
from distutils.spawn import find_executable

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'SYCL'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp']

config.excludes = ['Inputs']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.sycl_obj_root

# allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

# To be filled by lit.local.cfg files.
config.required_features = []
config.unsupported_features = []

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
    'CL_CONFIG_DEVICES', 'SYCL_DEVICE_ALLOWLIST', 'SYCL_CONFIG_FILE_NAME', 'ASAN_OPTIONS'])

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
    config.available_features.add('matrix-fp16') # PVC implies the support of FP16 matrix
    config.available_features.add('matrix-tf32') # PVC implies the support of TF32 matrix

if lit_config.params.get('matrix', False):
    config.available_features.add('matrix')

if lit_config.params.get('matrix-tf32', False):
    config.available_features.add('matrix-tf32')

if lit_config.params.get('matrix-xmx8', False):
    config.available_features.add('matrix-xmx8')
    config.available_features.add('matrix-fp16') # XMX implies the support of FP16 matrix

if lit_config.params.get('matrix-fp16', False):
    config.available_features.add('matrix-fp16')

#support for LIT parameter ur_l0_debug<num>
if lit_config.params.get('ur_l0_debug'):
    config.ur_l0_debug = lit_config.params.get('ur_l0_debug')
    lit_config.note("UR_L0_DEBUG: "+config.ur_l0_debug)

#support for LIT parameter ur_l0_leaks_debug
if lit_config.params.get('ur_l0_leaks_debug'):
    config.ur_l0_leaks_debug = lit_config.params.get('ur_l0_leaks_debug')
    lit_config.note("UR_L0_LEAKS_DEBUG: "+config.ur_l0_leaks_debug)

# Make sure that any dynamic checks below are done in the build directory and
# not where the sources are located. This is important for the in-tree
# configuration (as opposite to the standalone one).
os.chdir(config.sycl_obj_root)

# check if compiler supports CL command line options
cl_options=False
sp = subprocess.getstatusoutput(config.dpcpp_compiler+' /help')
if sp[0] == 0:
    cl_options=True
    config.available_features.add('cl_options')

# Check for Level Zero SDK
check_l0_file='l0_include.cpp'
with open(check_l0_file, 'w') as fp:
    print(textwrap.dedent(
        '''
        #include <level_zero/ze_api.h>
        int main() { uint32_t t; zeDriverGet(&t, nullptr); return t; }
        '''
    ), file=fp)

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

# Check for sycl-preview library
check_preview_breaking_changes_file='preview_breaking_changes_link.cpp'
with open(check_preview_breaking_changes_file, 'w') as fp:
    print(textwrap.dedent(
        '''
        #include <sycl/sycl.hpp>
        namespace sycl { inline namespace _V1 { namespace detail {
        extern void PreviewMajorReleaseMarker();
        }}}
        int main() { sycl::detail::PreviewMajorReleaseMarker(); return 0; }
        '''
    ), file=fp)

sp = subprocess.getstatusoutput(config.dpcpp_compiler+' -fsycl -fpreview-breaking-changes ' + check_preview_breaking_changes_file)
if sp[0] == 0:
    config.available_features.add('preview-breaking-changes-supported')

# Check for CUDA SDK
check_cuda_file='cuda_include.cpp'
with open(check_cuda_file, 'w') as fp:
    print(textwrap.dedent(
        '''
        #include <cuda.h>
        int main() { CUresult r = cuInit(0); return r; }
        '''
    ), file=fp)

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
    config.substitutions.append( ('%sycl_options',  ' ' + os.path.normpath(os.path.join(config.sycl_libs_dir + '/../lib/sycl7.lib')) + ' /I' +
                                config.sycl_include + ' /I' + os.path.join(config.sycl_include, 'sycl')) )
    config.substitutions.append( ('%include_option',  '/FI' ) )
    config.substitutions.append( ('%debug_option',  '/DEBUG' ) )
    config.substitutions.append( ('%cxx_std_option',  '/std:' ) )
    config.substitutions.append( ('%fPIC', '') )
    config.substitutions.append( ('%shared_lib', '/LD') )
else:
    config.substitutions.append( ('%sycl_options',
                                  (' -lsycl7' if platform.system() == "Windows" else " -lsycl") + ' -I' +
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


config.substitutions.append( ('%vulkan_include_dir', config.vulkan_include_dir ) )
config.substitutions.append( ('%vulkan_lib', config.vulkan_lib ) )

if platform.system() == "Windows":
    config.substitutions.append(
        ('%link-vulkan',
         '-l %s -I %s' % (config.vulkan_lib, config.vulkan_include_dir)))
else:
    vulkan_lib_path = os.path.dirname(config.vulkan_lib)
    config.substitutions.append(('%link-vulkan', '-L %s -lvulkan -I %s' %
                                 (vulkan_lib_path, config.vulkan_include_dir)))

if config.vulkan_found == "TRUE":
    config.available_features.add('vulkan')

if not config.gpu_aot_target_opts:
    config.gpu_aot_target_opts = '"-device *"'

config.substitutions.append( ('%gpu_aot_target_opts',  config.gpu_aot_target_opts ) )

if config.dump_ir_supported:
    config.available_features.add('dump_ir')

lit_config.note("Targeted devices: {}".format(', '.join(config.sycl_devices)))

sycl_ls = FindTool('sycl-ls').resolve(llvm_config, config.llvm_tools_dir)
if not sycl_ls:
    lit_config.fatal("can't find `sycl-ls`")

if len(config.sycl_devices) == 1 and config.sycl_devices[0] == 'all':
    devices = set()
    cmd = '{} {}'.format(config.run_launcher, sycl_ls) if config.run_launcher else sycl_ls
    sp = subprocess.check_output(cmd, text=True, shell=True)
    for line in sp.splitlines():
        if "gfx90a" in line:
            config.available_features.add("gpu-amd-gfx90a")
        if not line.startswith('['):
            continue
        (backend, device, _) = line[1:].split(':', 2)
        devices.add('{}:{}'.format(backend, device))
    config.sycl_devices = list(devices)

if len(config.sycl_devices) > 1:
    lit_config.note('Running on multiple devices, XFAIL-marked tests will be skipped on corresponding devices')

config.sycl_devices = [x.replace('ext_oneapi_', '') for x in config.sycl_devices]

available_devices = {'opencl': ('cpu', 'gpu', 'acc'),
                     'cuda':('gpu'),
                     'level_zero':('gpu'),
                     'hip':('gpu'),
                     'native_cpu':('cpu')}
for d in config.sycl_devices:
     be, dev = d.split(':')
     if be not in available_devices or dev not in available_devices[be]:
          lit_config.error('Unsupported device {}'.format(d))

# If HIP_PLATFORM flag is not set, default to AMD, and check if HIP platform is supported
supported_hip_platforms=["AMD", "NVIDIA"]
if config.hip_platform == "":
    config.hip_platform = "AMD"
if config.hip_platform not in supported_hip_platforms:
    lit_config.error("Unknown HIP platform '" + config.hip_platform + "' supported platforms are " + ', '.join(supported_hip_platforms))

# FIXME: This needs to be made per-device as well, possibly with a helper.
if "hip:gpu" in config.sycl_devices and config.hip_platform == "AMD":
    llvm_config.with_system_environment('ROCM_PATH')
    config.available_features.add('hip_amd')
    arch_flag = '-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=' + config.amd_arch
elif "hip:gpu" in config.sycl_devices and config.hip_platform == "NVIDIA":
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

if lit_config.params.get('ze_debug'):
    config.available_features.add('ze_debug')

if config.run_launcher:
    config.substitutions.append(('%e2e_tests_root', config.test_source_root))

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

# Tools for which we add a corresponding feature when available.
feature_tools = [
  ToolSubst('llvm-spirv', unresolved='ignore'),
  ToolSubst('llvm-link', unresolved='ignore'),
]

tools = [
  ToolSubst('FileCheck', unresolved='ignore'),
  # not is only substituted in certain circumstances; this is lit's default
  # behaviour.
  ToolSubst(r'\| \bnot\b', command=FindTool('not'),
    verbatim=True, unresolved='ignore'),
  ToolSubst('sycl-ls', command=sycl_ls, unresolved='ignore'),
] + feature_tools

# Try and find each of these tools in the llvm tools directory or the PATH, in
# that order. If found, they will be added as substitutions with the full path
# to the tool. This allows us to support both in-tree builds and standalone
# builds, where the tools may be externally defined.
llvm_config.add_tool_substitutions(tools, [config.llvm_tools_dir,
                                           os.environ.get('PATH', '')])
for tool in feature_tools:
    if tool.was_resolved:
        config.available_features.add(tool.key)
    else:
        lit_config.warning("Can't find " + tool.key)

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

for sycl_device in config.sycl_devices:
    be, dev = sycl_device.split(':')
    config.available_features.add('any-device-is-' + dev)
    # Use short names for LIT rules.
    config.available_features.add('any-device-is-' + be)

# That has to be executed last so that all device-independent features have been
# discovered already.
config.sycl_dev_features = {}

# Version of the driver for a given device. Empty for non-Intel devices.
config.intel_driver_ver = {}
for sycl_device in config.sycl_devices:
    env = copy.copy(llvm_config.config.environment)
    env['ONEAPI_DEVICE_SELECTOR'] = sycl_device
    if sycl_device.startswith('cuda:'):
        env['SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT'] = '1'
    # When using the ONEAPI_DEVICE_SELECTOR environment variable, sycl-ls
    # prints warnings that might derail a user thinking something is wrong
    # with their test run. It's just us filtering here, so silence them unless
    # we get an exit status.
    try:
        cmd = '{} {} --verbose'.format(config.run_launcher or "", sycl_ls)
        sp = subprocess.run(cmd, env=env, text=True, shell=True,
                            capture_output=True)
        sp.check_returncode()
    except subprocess.CalledProcessError as e:
        # capturing e allows us to see path resolution errors / system
        # permissions errors etc
        lit_config.fatal(f'Cannot list device aspects for {sycl_device}\n'
                         f'{e}\n'
                         f'stdout:{sp.stdout}\n'
                         f'stderr:{sp.stderr}\n')

    dev_aspects = []
    dev_sg_sizes = []
    # See format.py's parse_min_intel_driver_req for explanation.
    is_intel_driver = False
    intel_driver_ver = {}
    for line in sp.stdout.splitlines():
        if re.match(r' *Vendor *: Intel\(R\) Corporation', line):
            is_intel_driver = True
        if re.match(r' *Driver *:', line):
            _, driver_str = line.split(':', 1)
            driver_str = driver_str.strip()
            lin = re.match(r'[0-9]{1,2}\.[0-9]{1,2}\.([0-9]{5})', driver_str)
            if lin:
                intel_driver_ver['lin'] = int(lin.group(1))
            win = re.match(r'[0-9]{1,2}\.[0-9]{1,2}\.([0-9]{3})\.([0-9]{4})', driver_str)
            if win:
                intel_driver_ver['win'] = (int(win.group(1)), int(win.group(2)))
        if re.match(r' *Aspects *:', line):
            _, aspects_str = line.split(':', 1)
            dev_aspects.append(aspects_str.strip().split(' '))
        if re.match(r' *info::device::sub_group_sizes:', line):
            # str.removeprefix isn't universally available...
            sg_sizes_str = line.strip().replace('info::device::sub_group_sizes: ', '')
            dev_sg_sizes.append(sg_sizes_str.strip().split(' '))

    if dev_aspects == []:
        lit_config.error('Cannot detect device aspect for {}\nstdout:\n{}\nstderr:\n{}'.format(
            sycl_device, sp.stdout, sp.stderr))
        dev_aspects.append(set())
    # We might have several devices matching the same filter in the system.
    # Compute intersection of aspects.
    aspects = set(dev_aspects[0]).intersection(*dev_aspects)
    lit_config.note('Aspects for {}: {}'.format(sycl_device, ', '.join(aspects)))

    if dev_sg_sizes == []:
        lit_config.error('Cannot detect device SG sizes for {}\nstdout:\n{}\nstderr:\n{}'.format(
            sycl_device, sp.stdout, sp.stderr))
        dev_sg_sizes.append(set())
    # We might have several devices matching the same filter in the system.
    # Compute intersection of aspects.
    sg_sizes = set(dev_sg_sizes[0]).intersection(*dev_sg_sizes)
    lit_config.note('SG sizes for {}: {}'.format(sycl_device, ', '.join(sg_sizes)))

    aspect_features = set('aspect-' + a for a in aspects)
    sg_size_features = set('sg-' + s for s in sg_sizes)
    features = set();
    features.update(aspect_features)
    features.update(sg_size_features)

    be, dev = sycl_device.split(':')
    features.add(dev.replace('acc', 'accelerator'))
    # Use short names for LIT rules.
    features.add(be)

    config.sycl_dev_features[sycl_device] = features.union(config.available_features)
    if is_intel_driver:
        config.intel_driver_ver[sycl_device] = intel_driver_ver
    else:
        config.intel_driver_ver[sycl_device] = {}

# Set timeout for a single test
try:
    import psutil
    lit_config.maxIndividualTestTime = 600
except ImportError:
    pass

config.substitutions.append( ('%device_sanitizer_flags', "-Xsycl-target-frontend=spir64 -fsanitize=address") )
