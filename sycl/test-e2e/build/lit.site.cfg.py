

import sys
import platform
import subprocess

import site

site.addsitedir("/iusers/msabiani/llvm/sycl/test-e2e")

config.dpcpp_compiler = lit_config.params.get("dpcpp_compiler", "/iusers/msabiani/llvm/build/bin/clang++")
config.dpcpp_root_dir= os.path.dirname(os.path.dirname(config.dpcpp_compiler))
config.dpcpp_bin_dir = os.path.join(config.dpcpp_root_dir, 'bin')

def get_dpcpp_tool_path(name):
    try:
        return subprocess.check_output([config.dpcpp_compiler, "-print-prog-name=" + name], text=True)
    except subprocess.CalledProcessError:
        return os.path.join(config.dpcpp_bin_dir, name)

config.llvm_main_include_dir = ""
config.llvm_tools_dir = os.path.dirname(get_dpcpp_tool_path("llvm-config"))
config.lit_tools_dir = os.path.dirname("")
config.dump_ir_supported = lit_config.params.get("dump_ir", ("" if "" else False))
config.sycl_tools_dir = config.llvm_tools_dir
config.sycl_include = os.path.join(config.dpcpp_root_dir, 'include')
config.sycl_obj_root = "/iusers/msabiani/llvm/sycl/test-e2e/build"
config.sycl_libs_dir =  os.path.join(config.dpcpp_root_dir, ('bin' if platform.system() == "Windows" else 'lib'))

config.opencl_libs_dir = (os.path.dirname("OpenCL_LIBRARY-NOTFOUND") if "OpenCL_LIBRARY-NOTFOUND" else "")
config.level_zero_libs_dir = ""
config.level_zero_include = ""
config.cuda_libs_dir = ""
config.cuda_include = ""
config.hip_libs_dir = ""
config.hip_include = ""

config.opencl_include_dir = os.path.join(config.sycl_include, 'sycl')

config.igc_tag_file = os.path.join("/usr/local/lib/igc/", 'IGCTAG.txt')

config.sycl_devices = lit_config.params.get("sycl_devices", "opencl:cpu").split(';')

config.sycl_build_targets = set("target-" + t for t in lit_config.params.get(
    "sycl_build_targets", "all").split(';'))

config.amd_arch = lit_config.params.get("amd_arch", "")
config.sycl_threads_lib = ''
config.extra_environment = lit_config.params.get("extra_environment", "")
config.extra_system_environment = lit_config.params.get("extra_system_environment", "")
config.cxx_flags = lit_config.params.get("cxx_flags", " -Werror")
config.c_flags = ""
config.external_tests = ""
config.extra_include = "/iusers/msabiani/llvm/sycl/test-e2e/include"
config.gpu_aot_target_opts = lit_config.params.get("gpu_aot_target_opts", "")

config.vulkan_include_dir = "Vulkan_INCLUDE_DIR-NOTFOUND"
config.vulkan_lib = "Vulkan_LIBRARY-NOTFOUND"
config.vulkan_found = "FALSE"

config.run_launcher = lit_config.params.get('run_launcher', "")
config.allow_unknown_arch = "OFF"

import lit.llvm
lit.llvm.initialize(lit_config, config)

lit_config.load_config(config, "/iusers/msabiani/llvm/sycl/test-e2e/lit.cfg.py")

from format import SYCLEndToEndTest
config.test_format = SYCLEndToEndTest()
