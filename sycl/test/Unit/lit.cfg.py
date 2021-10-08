# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import platform
import subprocess

import lit.formats
import lit.util

# name: The name of this test suite.
config.name = 'SYCL-Unit'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.sycl_obj_root, 'unittests')
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, 'Tests')

# Propagate the temp directory. Windows requires this because it uses \Windows\
# if none of these are present.
if 'TMP' in os.environ:
    config.environment['TMP'] = os.environ['TMP']
if 'TEMP' in os.environ:
    config.environment['TEMP'] = os.environ['TEMP']

if 'SYCL_CONFIG_FILE_NAME' in os.environ:
    config.environment['SYCL_CONFIG_FILE_NAME'] = os.environ['SYCL_CONFIG_FILE_NAME']
else:
    # Since SYCL RT can be now statically linked into the unit test binary,
    # dynamic library location resolution mechanisms can be incorrect for such
    # tests. Provide the runtime with non-existing configuration file name to
    # force it load the default configuration.
    config.environment['SYCL_CONFIG_FILE_NAME'] = "null.cfg"

if 'SYCL_DEVICELIB_NO_FALLBACK' in os.environ:
    config.environment['SYCL_DEVICELIB_NO_FALLBACK'] = os.environ['SYCL_DEVICELIB_NO_FALLBACK']
else:
    # Disable device library fallback for unit tests by default.
    config.environment['SYCL_DEVICELIB_NO_FALLBACK'] = "1"

# Propagate path to symbolizer for ASan/MSan.
for symbolizer in ['ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH']:
    if symbolizer in os.environ:
        config.environment[symbolizer] = os.environ[symbolizer]

def find_shlibpath_var():
    if platform.system() in ['Linux', 'FreeBSD', 'NetBSD', 'SunOS']:
        yield 'LD_LIBRARY_PATH'
    elif platform.system() == 'Darwin':
        yield 'DYLD_LIBRARY_PATH'
    elif platform.system() == 'Windows':
        yield 'PATH'
    elif platform.system() == 'AIX':
        yield 'LIBPATH'

for shlibpath_var in find_shlibpath_var():
    # in stand-alone builds, shlibdir is clang's build tree
    # while llvm_libs_dir is installed LLVM (and possibly older clang)
    shlibpath = os.path.pathsep.join(
        (config.shlibdir,
         config.llvm_libs_dir,
         config.environment.get(shlibpath_var, '')))
    config.environment[shlibpath_var] = shlibpath
    break
else:
    lit_config.warning("unable to inject shared library path on '{}'"
                       .format(platform.system()))

config.environment['SYCL_CACHE_DIR'] = config.llvm_obj_root + "/sycl_cache"
config.environment['SYCL_DEVICE_FILTER'] = lit_config.params.get('SYCL_PLUGIN', "opencl") + ",host"
lit_config.note("Backend: {}".format(config.environment['SYCL_DEVICE_FILTER']))
lit_config.note("SYCL cache directory: {}".format(config.environment['SYCL_CACHE_DIR']))
