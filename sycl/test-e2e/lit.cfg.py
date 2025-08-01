# -*- Python -*-

import os
import platform
import copy
import re
import subprocess
import textwrap
import shlex
import shutil

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool

# Configuration file for the 'lit' test runner.
config.backend_to_target = {
    "level_zero": "target-spir",
    "opencl": "target-spir",
    "cuda": "target-nvidia",
    "hip": "target-amd",
    "native_cpu": "target-native_cpu",
    "offload": config.offload_build_target,
}
config.target_to_triple = {
    "target-spir": "spir64",
    "target-nvidia": "nvptx64-nvidia-cuda",
    "target-amd": "amdgcn-amd-amdhsa",
    "target-native_cpu": "native_cpu",
}
config.triple_to_target = {v: k for k, v in config.target_to_triple.items()}
config.backend_to_triple = {
    k: config.target_to_triple.get(v) for k, v in config.backend_to_target.items()
}

# name: The name of this test suite.
config.name = "SYCL"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".cpp"]

config.excludes = ["Inputs"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.sycl_obj_root

# allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

# To be filled by lit.local.cfg files.
config.required_features = []
config.unsupported_features = []

# test-mode: Set if tests should run normally or only build/run
config.test_mode = lit_config.params.get("test-mode", "full")
config.fallback_build_run_only = False
if config.test_mode == "full":
    config.available_features.add("build-mode")
    config.available_features.add("run-mode")
elif config.test_mode == "run-only":
    lit_config.note("run-only test mode enabled, only executing tests")
    config.available_features.add("run-mode")
elif config.test_mode == "build-only":
    lit_config.note("build-only test mode enabled, only compiling tests")
    config.available_features.add("build-mode")
    config.sycl_devices = []
    if not config.amd_arch:
        config.amd_arch = "gfx1030"
else:
    lit_config.error("Invalid argument for test-mode")

# Dummy substitution to indicate line should be a run line
config.substitutions.append(("%{run-aux}", ""))

# Cleanup environment variables which may affect tests
possibly_dangerous_env_vars = [
    "COMPILER_PATH",
    "RC_DEBUG_OPTIONS",
    "CINDEXTEST_PREAMBLE_FILE",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH",
    "LIBCLANG_TIMING",
    "LIBCLANG_OBJTRACKING",
    "LIBCLANG_LOGGING",
    "LIBCLANG_BGPRIO_INDEX",
    "LIBCLANG_BGPRIO_EDIT",
    "LIBCLANG_NOTHREADS",
    "LIBCLANG_RESOURCE_USAGE",
    "LIBCLANG_CODE_COMPLETION_LOGGING",
]

# Names of the Release and Debug versions of the XPTIFW library
XPTIFW_RELEASE = "xptifw"
XPTIFW_DEBUG = "xptifwd"

# Clang/Win32 may refer to %INCLUDE%. vsvarsall.bat sets it.
if platform.system() != "Windows":
    possibly_dangerous_env_vars.append("INCLUDE")

for name in possibly_dangerous_env_vars:
    if name in llvm_config.config.environment:
        del llvm_config.config.environment[name]

# Propagate some variables from the host environment.
llvm_config.with_system_environment(
    [
        "PATH",
        "OCL_ICD_FILENAMES",
        "OCL_ICD_VENDORS",
        "CL_CONFIG_DEVICES",
        "SYCL_DEVICE_ALLOWLIST",
        "SYCL_CONFIG_FILE_NAME",
    ]
)

# Take into account extra system environment variables if provided via parameter.
if config.extra_system_environment:
    lit_config.note(
        "Extra system variables to propagate value from: "
        + config.extra_system_environment
    )
    extra_env_vars = config.extra_system_environment.split(",")
    for var in extra_env_vars:
        if var in os.environ:
            llvm_config.with_system_environment(var)

llvm_config.with_environment("PATH", config.lit_tools_dir, append_path=True)

# Configure LD_LIBRARY_PATH or corresponding os-specific alternatives
if platform.system() == "Linux":
    config.available_features.add("linux")
    llvm_config.with_system_environment(
        ["LD_LIBRARY_PATH", "LIBRARY_PATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"]
    )
    llvm_config.with_environment(
        "LD_LIBRARY_PATH", config.sycl_libs_dir, append_path=True
    )

elif platform.system() == "Windows":
    config.available_features.add("windows")
    llvm_config.with_system_environment(
        ["LIB", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "INCLUDE"]
    )
    llvm_config.with_environment("LIB", config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment("PATH", config.sycl_libs_dir, append_path=True)
    llvm_config.with_environment(
        "LIB", os.path.join(config.dpcpp_root_dir, "lib"), append_path=True
    )

elif platform.system() == "Darwin":
    # FIXME: surely there is a more elegant way to instantiate the Xcode directories.
    llvm_config.with_system_environment(["C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"])
    llvm_config.with_environment(
        "CPLUS_INCLUDE_PATH",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1",
        append_path=True,
    )
    llvm_config.with_environment(
        "C_INCLUDE_PATH",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/",
        append_path=True,
    )
    llvm_config.with_environment(
        "CPLUS_INCLUDE_PATH",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/",
        append_path=True,
    )
    llvm_config.with_environment("DYLD_LIBRARY_PATH", config.sycl_libs_dir)

llvm_config.with_environment("PATH", config.sycl_tools_dir, append_path=True)

if config.extra_environment:
    lit_config.note("Extra environment variables")
    for env_pair in config.extra_environment.split(","):
        [var, val] = env_pair.split("=", 1)
        if val:
            llvm_config.with_environment(var, val)
            lit_config.note("\t" + var + "=" + val)
        else:
            lit_config.note("\tUnset " + var)
            llvm_config.with_environment(var, "")

# Disable the UR logger callback sink during test runs as output to SYCL RT can interfere with some tests relying on standard input/output
llvm_config.with_environment("UR_LOG_CALLBACK", "disabled")


# Temporarily modify environment to be the same that we use when running tests
class test_env:
    def __enter__(self):
        self.old_environ = dict(os.environ)
        os.environ.clear()
        os.environ.update(config.environment)
        self.old_dir = os.getcwd()
        os.chdir(config.sycl_obj_root)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.environ.clear()
        os.environ.update(self.old_environ)
        os.chdir(self.old_dir)


config.substitutions.append(("%sycl_libs_dir", config.sycl_libs_dir))
if platform.system() == "Windows":
    config.substitutions.append(
        ("%sycl_static_libs_dir", config.sycl_libs_dir + "/../lib")
    )
    config.substitutions.append(("%obj_ext", ".obj"))
    config.substitutions.append(
        ("%sycl_include", "-Xclang -isystem -Xclang " + config.sycl_include)
    )
elif platform.system() == "Linux":
    config.substitutions.append(("%sycl_static_libs_dir", config.sycl_libs_dir))
    config.substitutions.append(("%obj_ext", ".o"))
    config.substitutions.append(("%sycl_include", "-isystem " + config.sycl_include))

# Intel GPU FAMILY availability
if lit_config.params.get("gpu-intel-gen11", False):
    config.available_features.add("gpu-intel-gen11")
if lit_config.params.get("gpu-intel-gen12", False):
    config.available_features.add("gpu-intel-gen12")

# Intel GPU DEVICE availability
if lit_config.params.get("gpu-intel-dg2", False):
    config.available_features.add("gpu-intel-dg2")
if lit_config.params.get("gpu-intel-pvc-vg", False):
    config.available_features.add("gpu-intel-pvc-vg")

if lit_config.params.get("igc-dev", False):
    config.available_features.add("igc-dev")

# Map between device family and architecture types.
device_family_arch_map = {
    # <Family name> : Set of architectures types (and aliases)
    # DG2
    "gpu-intel-dg2": {
        "intel_gpu_acm_g12",
        "intel_gpu_dg2_g12",
        "intel_gpu_acm_g11",
        "intel_gpu_dg2_g11",
        "intel_gpu_acm_g10",
        "intel_gpu_dg2_g10",
    },
    # Gen12
    "gpu-intel-gen12": {"intel_gpu_tgllp", "intel_gpu_tgl"},
    # Gen11
    "gpu-intel-gen11": {"intel_gpu_icllp", "intel_gpu_icl"},
}


def get_device_family_from_arch(arch):
    for device_family, arch_set in device_family_arch_map.items():
        if arch in arch_set:
            return device_family
    return None


def check_igc_tag_and_add_feature():
    if os.path.isfile(config.igc_tag_file):
        with open(config.igc_tag_file, "r") as tag_file:
            contents = tag_file.read()
            if "igc-dev" in contents:
                config.available_features.add("igc-dev")


def quote_path(path):
    if not path:
        return ""
    if platform.system() == "Windows":
        return f'"{path}"'
    return shlex.quote(path)


# Call the function to perform the check and add the feature
check_igc_tag_and_add_feature()

# support for LIT parameter ur_l0_debug<num>
if lit_config.params.get("ur_l0_debug"):
    config.ur_l0_debug = lit_config.params.get("ur_l0_debug")
    lit_config.note("UR_L0_DEBUG: " + config.ur_l0_debug)

# support for LIT parameter ur_l0_leaks_debug
if lit_config.params.get("ur_l0_leaks_debug"):
    config.ur_l0_leaks_debug = lit_config.params.get("ur_l0_leaks_debug")
    lit_config.note("UR_L0_LEAKS_DEBUG: " + config.ur_l0_leaks_debug)

if lit_config.params.get("enable-perf-tests", False):
    config.available_features.add("enable-perf-tests")

if lit_config.params.get("spirv-backend", False):
    config.available_features.add("spirv-backend")


# Use this to make sure that any dynamic checks below are done in the build
# directory and not where the sources are located. This is important for the
# in-tree configuration (as opposite to the standalone one).
def open_check_file(file_name):
    return open(os.path.join(config.sycl_obj_root, file_name), "w")


# check if compiler supports CL command line options
cl_options = False
with test_env():
    sp = subprocess.getstatusoutput(config.dpcpp_compiler + " /help")
    if sp[0] == 0:
        cl_options = True
        config.available_features.add("cl_options")

# check if the compiler was built in NDEBUG configuration
has_ndebug = False
ps = subprocess.Popen(
    [config.dpcpp_compiler, "-mllvm", "-debug", "-x", "c", "-", "-S", "-o", os.devnull],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
_ = ps.communicate(input=b"int main(){}\n")
if ps.wait() == 0:
    config.available_features.add("has_ndebug")

# Check for Level Zero SDK
check_l0_file = "l0_include.cpp"
with open_check_file(check_l0_file) as fp:
    print(
        textwrap.dedent(
            """
        #include <level_zero/ze_api.h>
        int main() { uint32_t t; zeDriverGet(&t, nullptr); return t; }
        """
        ),
        file=fp,
    )

config.level_zero_libs_dir = quote_path(
    lit_config.params.get("level_zero_libs_dir", config.level_zero_libs_dir)
)
config.level_zero_include = quote_path(
    lit_config.params.get(
        "level_zero_include",
        (
            config.level_zero_include
            if config.level_zero_include
            else config.sycl_include
        ),
    )
)

level_zero_options = level_zero_options = (
    (" -L" + config.level_zero_libs_dir if config.level_zero_libs_dir else "")
    + " -lze_loader "
    + " -I"
    + config.level_zero_include
)
if cl_options:
    level_zero_options = (
        " "
        + (
            config.level_zero_libs_dir + "/ze_loader.lib "
            if config.level_zero_libs_dir
            else "ze_loader.lib"
        )
        + " /I"
        + config.level_zero_include
    )

config.substitutions.append(("%level_zero_options", level_zero_options))

with test_env():
    sp = subprocess.getstatusoutput(
        config.dpcpp_compiler + " -fsycl  " + check_l0_file + level_zero_options
    )
    if sp[0] == 0:
        config.available_features.add("level_zero_dev_kit")
        config.substitutions.append(("%level_zero_options", level_zero_options))
    else:
        config.substitutions.append(("%level_zero_options", ""))

if lit_config.params.get("test-preview-mode", "False") != "False":
    config.available_features.add("preview-mode")
else:
    # Check for sycl-preview library
    check_preview_breaking_changes_file = "preview_breaking_changes_link.cpp"
    with open_check_file(check_preview_breaking_changes_file) as fp:
        print(
            textwrap.dedent(
                """
            #include <sycl/sycl.hpp>
            namespace sycl { inline namespace _V1 { namespace detail {
            extern void PreviewMajorReleaseMarker();
            }}}
            int main() { sycl::detail::PreviewMajorReleaseMarker(); return 0; }
            """
            ),
            file=fp,
        )

    with test_env():
        sp = subprocess.getstatusoutput(
            config.dpcpp_compiler
            + " -fsycl -fpreview-breaking-changes "
            + check_preview_breaking_changes_file
        )
        if sp[0] == 0:
            config.available_features.add("preview-breaking-changes-supported")

# Check if clang is built with ZSTD and compression support.
fPIC_opt = "-fPIC" if platform.system() != "Windows" else ""
# -shared is invalid for icx on Windows, use /LD instead.
dll_opt = "/LD" if cl_options else "-shared"

ps = subprocess.Popen(
    [
        config.dpcpp_compiler,
        "-fsycl",
        "--offload-compress",
        dll_opt,
        fPIC_opt,
        "-x",
        "c++",
        "-",
        "-o",
        os.devnull,
    ],
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
op = ps.communicate(input=b"")
if ps.wait() == 0:
    config.available_features.add("zstd")

# Check for CUDA SDK
check_cuda_file = "cuda_include.cpp"
with open_check_file(check_cuda_file) as fp:
    print(
        textwrap.dedent(
            """
        #include <cuda.h>
        int main() { CUresult r = cuInit(0); return r; }
        """
        ),
        file=fp,
    )

config.cuda_libs_dir = quote_path(
    lit_config.params.get("cuda_libs_dir", config.cuda_libs_dir)
)
config.cuda_include = quote_path(
    lit_config.params.get(
        "cuda_include",
        (config.cuda_include if config.cuda_include else config.sycl_include),
    )
)

cuda_options = (
    (" -L" + config.cuda_libs_dir if config.cuda_libs_dir else "")
    + " -lcuda "
    + " -I"
    + config.cuda_include
)
if cl_options:
    cuda_options = (
        " "
        + (config.cuda_libs_dir + "/cuda.lib " if config.cuda_libs_dir else "cuda.lib")
        + " /I"
        + config.cuda_include
    )
if platform.system() == "Windows":
    cuda_options += (
        " --cuda-path=" + os.path.dirname(os.path.dirname(config.cuda_libs_dir)) + f'"'
    )

config.substitutions.append(("%cuda_options", cuda_options))

with test_env():
    sp = subprocess.getstatusoutput(
        config.dpcpp_compiler + " -fsycl  " + check_cuda_file + cuda_options
    )
    if sp[0] == 0:
        config.available_features.add("cuda_dev_kit")
        config.substitutions.append(("%cuda_options", cuda_options))
    else:
        config.substitutions.append(("%cuda_options", ""))

# Check for HIP SDK
check_hip_file = "hip_include.cpp"
with open_check_file(check_hip_file) as fp:
    print(
        textwrap.dedent(
            """
        #define __HIP_PLATFORM_AMD__ 1
        #include <hip/hip_runtime.h>
        int main() {  hipError_t r = hipInit(0); return r; }
        """
        ),
        file=fp,
    )
config.hip_libs_dir = quote_path(
    lit_config.params.get("hip_libs_dir", config.hip_libs_dir)
)
config.hip_include = quote_path(
    lit_config.params.get(
        "hip_include",
        (config.hip_include if config.hip_include else config.sycl_include),
    )
)

hip_options = (
    (" -L" + config.hip_libs_dir if config.hip_libs_dir else "")
    + " -lamdhip64 "
    + " -I"
    + config.hip_include
)
if cl_options:
    hip_options = (
        " "
        + (
            config.hip_libs_dir + "/amdhip64.lib "
            if config.hip_libs_dir
            else "amdhip64.lib"
        )
        + " /I"
        + config.hip_include
    )
if platform.system() == "Windows":
    hip_options += " --rocm-path=" + os.path.dirname(config.hip_libs_dir) + f'"'
with test_env():
    sp = subprocess.getstatusoutput(
        config.dpcpp_compiler + " -fsycl  " + check_hip_file + hip_options
    )
    if sp[0] == 0:
        config.available_features.add("hip_dev_kit")
        config.substitutions.append(("%hip_options", hip_options))
    else:
        config.substitutions.append(("%hip_options", ""))

# Add ROCM_PATH from system environment, this is used by clang to find ROCm
# libraries in non-standard installation locations.
llvm_config.with_system_environment("ROCM_PATH")

# Check for OpenCL ICD
if config.opencl_libs_dir:
    config.opencl_libs_dir = quote_path(config.opencl_libs_dir)
    config.opencl_include_dir = quote_path(config.opencl_include_dir)
    if cl_options:
        config.substitutions.append(
            ("%opencl_lib", " " + config.opencl_libs_dir + "/OpenCL.lib")
        )
    else:
        config.substitutions.append(
            ("%opencl_lib", "-L" + config.opencl_libs_dir + " -lOpenCL")
        )
    config.available_features.add("opencl_icd")
config.substitutions.append(("%opencl_include_dir", config.opencl_include_dir))

if cl_options:
    config.substitutions.append(
        (
            "%sycl_options",
            " "
            + os.path.normpath(os.path.join(config.sycl_libs_dir + "/../lib/sycl8.lib"))
            + " -Xclang -isystem -Xclang "
            + config.sycl_include
            + " -Xclang -isystem -Xclang "
            + os.path.join(config.sycl_include, "sycl"),
        )
    )
    config.substitutions.append(("%include_option", "/FI"))
    config.substitutions.append(("%debug_option", "/Zi /DEBUG"))
    config.substitutions.append(("%cxx_std_option", "/clang:-std="))
    config.substitutions.append(("%fPIC", ""))
    config.substitutions.append(("%shared_lib", "/LD"))
    config.substitutions.append(("%O0", "/Od"))
    config.substitutions.append(("%fp-model-", "/fp:"))
else:
    config.substitutions.append(
        (
            "%sycl_options",
            (" -lsycl8" if platform.system() == "Windows" else " -lsycl")
            + " -isystem "
            + config.sycl_include
            + " -isystem "
            + os.path.join(config.sycl_include, "sycl")
            + " -L"
            + config.sycl_libs_dir,
        )
    )
    config.substitutions.append(("%include_option", "-include"))
    config.substitutions.append(("%debug_option", "-g"))
    config.substitutions.append(("%cxx_std_option", "-std="))
    # Position-independent code does not make sence on Windows. At the same
    # time providing this option for compilation targeting
    # x86_64-pc-windows-msvc will cause compile time error on some
    # configurations
    config.substitutions.append(
        ("%fPIC", ("" if platform.system() == "Windows" else "-fPIC"))
    )
    config.substitutions.append(("%shared_lib", "-shared"))
    config.substitutions.append(("%O0", "-O0"))
    config.substitutions.append(("%fp-model-", "-ffp-model="))

# Check if user passed verbose-print parameter, if yes, add VERBOSE_PRINT macro
if "verbose-print" in lit_config.params:
    config.substitutions.append(("%verbose_print", "-DVERBOSE_PRINT"))
else:
    config.substitutions.append(("%verbose_print", ""))

# Enable `vulkan` feature if Vulkan was found.
if config.vulkan_found == "TRUE":
    config.available_features.add("vulkan")

# Add Vulkan include and library paths to the configuration for substitution.
link_vulkan = "-I %s " % (config.vulkan_include_dir)
if platform.system() == "Windows":
    if cl_options:
        link_vulkan += "/clang:-l%s" % (config.vulkan_lib)
    else:
        link_vulkan += "-l %s" % (config.vulkan_lib)
else:
    vulkan_lib_path = os.path.dirname(config.vulkan_lib)
    link_vulkan += "-L %s -lvulkan" % (vulkan_lib_path)
config.substitutions.append(("%link-vulkan", link_vulkan))

# Add DirectX 12 libraries to the configuration for substitution.
if platform.system() == "Windows":
    directx_libs = ["-ld3d11", "-ld3d12", "-ldxgi", "-ldxguid"]
    if cl_options:
        directx_libs = ["/clang:" + l for l in directx_libs]
    config.substitutions.append(("%link-directx", " ".join(directx_libs)))

if not config.gpu_aot_target_opts:
    config.gpu_aot_target_opts = '"-device *"'

config.substitutions.append(("%gpu_aot_target_opts", config.gpu_aot_target_opts))

if config.dump_ir_supported:
    config.available_features.add("dump_ir")

lit_config.note("Targeted devices: {}".format(", ".join(config.sycl_devices)))

sycl_ls = FindTool("sycl-ls").resolve(
    llvm_config, os.pathsep.join([config.dpcpp_bin_dir, config.llvm_tools_dir])
)
if not sycl_ls:
    lit_config.fatal("can't find `sycl-ls`")

syclbin_dump = FindTool("syclbin-dump").resolve(
    llvm_config, os.pathsep.join([config.dpcpp_bin_dir, config.llvm_tools_dir])
)
if not syclbin_dump:
    lit_config.fatal("can't find `syclbin-dump`")

if (
    len(config.sycl_build_targets) == 1
    and next(iter(config.sycl_build_targets)) == "target-all"
):
    config.sycl_build_targets = {"target-spir"}
    sp = subprocess.getstatusoutput(config.dpcpp_compiler + " --print-targets")
    if "nvptx64" in sp[1]:
        config.sycl_build_targets.add("target-nvidia")
    if "amdgcn" in sp[1]:
        config.sycl_build_targets.add("target-amd")

with test_env():
    cmd = (
        "{} {}".format(config.run_launcher, sycl_ls) if config.run_launcher else sycl_ls
    )
    sycl_ls_output = subprocess.check_output(cmd, text=True, shell=True)

    # In contrast to `cpu` feature this is a compile-time feature, which is needed
    # to check if we can build cpu AOT tests.
    if "opencl:cpu" in sycl_ls_output:
        config.available_features.add("opencl-cpu-rt")

    if len(config.sycl_devices) == 1 and config.sycl_devices[0] == "all":
        devices = set()
        for line in sycl_ls_output.splitlines():
            if not line.startswith("["):
                continue
            (backend, device) = line[1:].split("]")[0].split(":")
            devices.add("{}:{}".format(backend, device))
        config.sycl_devices = list(devices)

if len(config.sycl_devices) > 1:
    lit_config.note(
        "Running on multiple devices, XFAIL-marked tests will be skipped on corresponding devices"
    )


def remove_level_zero_suffix(devices):
    return [device.replace("_v2", "").replace("_v1", "") for device in devices]


available_devices = {
    "opencl": ("cpu", "gpu", "fpga"),
    "cuda": "gpu",
    "level_zero": "gpu",
    "hip": "gpu",
    "native_cpu": "cpu",
    "offload": "gpu",
}
for d in remove_level_zero_suffix(config.sycl_devices):
    be, dev = d.split(":")
    # Verify platform
    if be not in available_devices:
        lit_config.error("Unsupported device {}".format(d))
    # Verify device from available_devices or accept if contains "arch-"
    if dev not in available_devices[be] and not "arch-" in dev:
        lit_config.error("Unsupported device {}".format(d))

if "cuda:gpu" in config.sycl_devices:
    if "CUDA_PATH" not in os.environ:
        if platform.system() == "Windows":
            cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            cuda_versions = []
            if os.path.exists(cuda_root):
                for entry in os.listdir(cuda_root):
                    if os.path.isdir(
                        os.path.join(cuda_root, entry)
                    ) and entry.startswith("v"):
                        version = entry[1:]  # Remove the leading 'v'
                        if re.match(
                            r"^\d+\.\d+$", version
                        ):  # Match version pattern like 12.3
                            cuda_versions.append(version)
                latest_cuda_version = max(
                    cuda_versions, key=lambda v: [int(i) for i in v.split(".")]
                )
                os.environ["CUDA_PATH"] = os.path.join(
                    cuda_root, f"v{latest_cuda_version}"
                )
        else:
            cuda_root = "/usr/local"
            cuda_versions = []
            if os.path.exists(cuda_root):
                for entry in os.listdir(cuda_root):
                    if os.path.isdir(
                        os.path.join(cuda_root, entry)
                    ) and entry.startswith("cuda-"):
                        version = entry.split("-")[1]
                        if re.match(
                            r"^\d+\.\d+$", version
                        ):  # Match version pattern like 12.3
                            cuda_versions.append(version)
                if cuda_versions:
                    latest_cuda_version = max(
                        cuda_versions, key=lambda v: [int(i) for i in v.split(".")]
                    )
                    os.environ["CUDA_PATH"] = os.path.join(
                        cuda_root, f"cuda-{latest_cuda_version}"
                    )
                elif os.path.exists(os.path.join(cuda_root, "cuda")):
                    os.environ["CUDA_PATH"] = os.path.join(cuda_root, "cuda")

    if "CUDA_PATH" not in os.environ:
        lit_config.error("Cannot run tests for CUDA without valid CUDA_PATH.")

    llvm_config.with_system_environment("CUDA_PATH")
    if platform.system() == "Windows":
        config.cuda_libs_dir = (
            '"' + os.path.join(os.environ["CUDA_PATH"], r"lib\x64") + '"'
        )
        config.cuda_include = (
            '"' + os.path.join(os.environ["CUDA_PATH"], "include") + '"'
        )
    else:
        config.cuda_libs_dir = os.path.join(os.environ["CUDA_PATH"], r"lib64")
        config.cuda_include = os.path.join(os.environ["CUDA_PATH"], "include")

config.substitutions.append(("%threads_lib", config.sycl_threads_lib))

if lit_config.params.get("ze_debug"):
    config.available_features.add("ze_debug")

if config.run_launcher:
    config.substitutions.append(("%e2e_tests_root", config.test_source_root))

# TODO properly set XPTIFW include and runtime dirs
xptifw_lib_dir = os.path.join(config.dpcpp_root_dir, "lib")
xptifw_dispatcher = ""
if platform.system() == "Linux":
    xptifw_dispatcher = os.path.join(xptifw_lib_dir, "libxptifw.so")
elif platform.system() == "Windows":
    # Use debug version of xptifw library if tests are built with \MDd.
    xptifw_dispatcher_name = (
        XPTIFW_DEBUG if "/MDd" in config.cxx_flags else XPTIFW_RELEASE
    )
    xptifw_dispatcher = os.path.join(
        config.dpcpp_root_dir, "bin", xptifw_dispatcher_name + ".dll"
    )
xptifw_includes = os.path.join(config.dpcpp_root_dir, "include")
if os.path.exists(xptifw_lib_dir) and os.path.exists(
    os.path.join(xptifw_includes, "xpti", "xpti_trace_framework.h")
):
    config.available_features.add("xptifw")
    config.substitutions.append(("%xptifw_dispatcher", xptifw_dispatcher))
    if cl_options:
        # Use debug version of xptifw library if tests are built with \MDd.
        xptifw_lib_name = XPTIFW_DEBUG if "/MDd" in config.cxx_flags else XPTIFW_RELEASE
        xptifw_lib = os.path.normpath(
            os.path.join(xptifw_lib_dir, xptifw_lib_name + ".lib")
        )
        config.substitutions.append(
            (
                "%xptifw_lib",
                f" {xptifw_lib} /I{xptifw_includes} ",
            )
        )
    else:
        config.substitutions.append(
            (
                "%xptifw_lib",
                " -L{} -lxptifw -I{} ".format(xptifw_lib_dir, xptifw_includes),
            )
        )

# Tools for which we add a corresponding feature when available.
feature_tools = [
    ToolSubst("llvm-spirv", unresolved="ignore"),
    ToolSubst("llvm-link", unresolved="ignore"),
]

tools = [
    ToolSubst("FileCheck", unresolved="ignore"),
    # not is only substituted in certain circumstances; this is lit's default
    # behaviour.
    ToolSubst(
        r"\| \bnot\b", command=FindTool("not"), verbatim=True, unresolved="ignore"
    ),
    ToolSubst("sycl-ls", command=sycl_ls, unresolved="ignore"),
    ToolSubst("syclbin-dump", command=syclbin_dump, unresolved="ignore"),
] + feature_tools

# Try and find each of these tools in the DPC++ bin directory, in the llvm tools directory
# or the PATH, in that order. If found, they will be added as substitutions with the full path
# to the tool. This allows us to support both in-tree builds and standalone
# builds, where the tools may be externally defined.
# The DPC++ bin directory is different from the LLVM bin directory when using
# the Intel Compiler (icx), which puts tools into $dpcpp_root_dir/bin/compiler
llvm_config.add_tool_substitutions(
    tools, [config.dpcpp_bin_dir, config.llvm_tools_dir, os.environ.get("PATH", "")]
)
for tool in feature_tools:
    if tool.was_resolved:
        config.available_features.add(tool.key)
    else:
        lit_config.warning("Can't find " + tool.key)

if shutil.which("cmc") is not None:
    config.available_features.add("cm-compiler")

# Device AOT compilation tools aren't part of the SYCL project,
# so they need to be pre-installed on the machine
aot_tools = ["ocloc", "opencl-aot"]

for aot_tool in aot_tools:
    if shutil.which(aot_tool) is not None:
        lit_config.note("Found pre-installed AOT device compiler " + aot_tool)
        config.available_features.add(aot_tool)
    else:
        lit_config.warning(
            "Couldn't find pre-installed AOT device compiler " + aot_tool
        )

# Clear build targets when not in build-only, to populate according to devices
if config.test_mode != "build-only":
    config.sycl_build_targets = set()


def get_sycl_ls_verbose(sycl_device, env):
    with test_env():
        # When using the ONEAPI_DEVICE_SELECTOR environment variable, sycl-ls
        # prints warnings that might derail a user thinking something is wrong
        # with their test run. It's just us filtering here, so silence them unless
        # we get an exit status.
        try:
            cmd = "{} {} --verbose".format(config.run_launcher or "", sycl_ls)
            sp = subprocess.run(
                cmd, env=env, text=True, shell=True, capture_output=True
            )
            sp.check_returncode()
        except subprocess.CalledProcessError as e:
            # capturing e allows us to see path resolution errors / system
            # permissions errors etc
            lit_config.fatal(
                f"Cannot find devices under {sycl_device}\n"
                f"{e}\n"
                f"stdout:{sp.stdout}\n"
                f"stderr:{sp.stderr}\n"
            )
        return sp


# A device filter such as level_zero:gpu can have multiple devices under it and
# the order is not guaranteed. The aspects enabled are also restricted to what
# is supported on all devices under the label. It is possible for level_zero:gpu
# and level_zero:0 to select different devices on different machines with the
# same hardware. It is not currently possible to pass the device architecture to
# ONEAPI_DEVICE_SELECTOR. Instead, if "BACKEND:arch-DEVICE_ARCH" is provided to
# "sycl_devices", giving the desired device architecture, select a device that
# matches that architecture using the backend:device-num device selection
# scheme.
filtered_sycl_devices = []
for sycl_device in config.sycl_devices:
    backend, device_arch = sycl_device.split(":", 1)

    if not "arch-" in device_arch:
        filtered_sycl_devices.append(sycl_device)
        continue

    env = copy.copy(llvm_config.config.environment)

    # Find all available devices under the backend
    env["ONEAPI_DEVICE_SELECTOR"] = backend + ":*"

    detected_architectures = []

    platform_devices = remove_level_zero_suffix(backend + ":*")

    for line in get_sycl_ls_verbose(platform_devices, env).stdout.splitlines():
        if re.match(r" *Architecture:", line):
            _, architecture = line.strip().split(":", 1)
            detected_architectures.append(architecture.strip())

    device = device_arch.replace("arch-", "")

    if device in detected_architectures:
        device_num = detected_architectures.index(device)
        filtered_sycl_devices.append(backend + ":" + str(device_num))
    else:
        lit_config.warning(
            "Couldn't find device with architecture {}"
            " under {} device selector! Skipping device "
            "{}".format(device, backend + ":*", sycl_device)
        )

if not filtered_sycl_devices and not config.test_mode == "build-only":
    lit_config.error(
        "No sycl devices selected! Check your device architecture filters."
    )

config.sycl_devices = filtered_sycl_devices

for sycl_device in remove_level_zero_suffix(config.sycl_devices):
    be, dev = sycl_device.split(":")
    config.available_features.add("any-device-is-" + dev)
    # Use short names for LIT rules.
    config.available_features.add("any-device-is-" + be)

    target = config.backend_to_target[be]
    config.sycl_build_targets.add(target)

for target in config.sycl_build_targets:
    config.available_features.add("any-target-is-" + target.replace("target-", ""))

if config.llvm_main_include_dir:
    lit_config.note("Using device config file built from LLVM")
    config.available_features.add("device-config-file")
    config.substitutions.append(
        ("%device_config_file_include_flag", f"-I {config.llvm_main_include_dir}")
    )
elif os.path.exists(f"{config.sycl_include}/llvm/SYCLLowerIR/DeviceConfigFile.hpp"):
    lit_config.note("Using installed device config file")
    config.available_features.add("device-config-file")
    config.substitutions.append(("%device_config_file_include_flag", ""))

# That has to be executed last so that all device-independent features have been
# discovered already.
config.sycl_dev_features = {}

# Version of the driver for a given device. Empty for non-Intel devices.
config.intel_driver_ver = {}
for full_name, sycl_device in zip(
    config.sycl_devices, remove_level_zero_suffix(config.sycl_devices)
):
    env = copy.copy(llvm_config.config.environment)

    env["ONEAPI_DEVICE_SELECTOR"] = sycl_device
    if sycl_device.startswith("cuda:"):
        env["SYCL_UR_CUDA_ENABLE_IMAGE_SUPPORT"] = "1"

    features = set()
    dev_aspects = []
    dev_sg_sizes = []
    architectures = set()
    # See format.py's parse_min_intel_driver_req for explanation.
    is_intel_driver = False
    intel_driver_ver = {}
    sycl_ls_sp = get_sycl_ls_verbose(sycl_device, env)
    for line in sycl_ls_sp.stdout.splitlines():
        if re.match(r" *Vendor *: Intel\(R\) Corporation", line):
            is_intel_driver = True
        if re.match(r" *Driver *:", line):
            _, driver_str = line.split(":", 1)
            driver_str = driver_str.strip()
            lin = re.match(r"[0-9]{1,2}\.[0-9]{1,2}\.([0-9]{5})", driver_str)
            if lin:
                intel_driver_ver["lin"] = int(lin.group(1))
            win = re.match(
                r"[0-9]{1,2}\.[0-9]{1,2}\.([0-9]{3})\.([0-9]{4})", driver_str
            )
            if win:
                intel_driver_ver["win"] = (int(win.group(1)), int(win.group(2)))
        if re.match(r" *Aspects *:", line):
            _, aspects_str = line.split(":", 1)
            dev_aspects.append(aspects_str.strip().split(" "))
        if re.match(r" *info::device::sub_group_sizes:", line):
            # str.removeprefix isn't universally available...
            sg_sizes_str = line.strip().replace("info::device::sub_group_sizes: ", "")
            dev_sg_sizes.append(sg_sizes_str.strip().split(" "))
        if re.match(r" *DeviceID*", line):
            gpu_intel_pvc_1T_device_id = "3034"
            gpu_intel_pvc_2T_device_id = "3029"
            _, device_id = line.strip().split(":", 1)
            device_id = device_id.strip()
            if device_id == gpu_intel_pvc_1T_device_id:
                config.available_features.add("gpu-intel-pvc-1T")
            if device_id == gpu_intel_pvc_2T_device_id:
                config.available_features.add("gpu-intel-pvc-2T")
        if re.match(r" *Architecture:", line):
            _, architecture = line.strip().split(":", 1)
            architectures.add(architecture.strip())
        if re.match(r" *Name *:", line) and "Level-Zero V2" in line:
            features.add("level_zero_v2_adapter")

    if dev_aspects == []:
        lit_config.error(
            "Cannot detect device aspect for {}\nstdout:\n{}\nstderr:\n{}".format(
                sycl_device, sycl_ls_sp.stdout, sycl_ls_sp.stderr
            )
        )
        dev_aspects.append(set())
    # We might have several devices matching the same filter in the system.
    # Compute intersection of aspects.
    aspects = set(dev_aspects[0]).intersection(*dev_aspects)
    lit_config.note("Aspects for {}: {}".format(sycl_device, ", ".join(aspects)))

    if dev_sg_sizes == []:
        lit_config.error(
            "Cannot detect device SG sizes for {}\nstdout:\n{}\nstderr:\n{}".format(
                sycl_device, sycl_ls_sp.stdout, sycl_ls_sp.stderr
            )
        )
        dev_sg_sizes.append(set())
    # We might have several devices matching the same filter in the system.
    # Compute intersection of aspects.
    sg_sizes = set(dev_sg_sizes[0]).intersection(*dev_sg_sizes)
    lit_config.note("SG sizes for {}: {}".format(sycl_device, ", ".join(sg_sizes)))

    # Currently, for fpga, the architecture reported by sycl-ls will always
    # be unknown, as there are currently no architectures specified for fpga
    # in sycl_ext_oneapi_device_architecture. Skip adding architecture features
    # in this case.
    if sycl_device == "opencl:fpga":
        architectures = set()
    else:
        lit_config.note(
            "Architectures for {}: {}".format(sycl_device, ", ".join(architectures))
        )
        if len(architectures) != 1 or "unknown" in architectures:
            if not config.allow_unknown_arch:
                lit_config.error(
                    "Cannot detect architecture for {}\nstdout:\n{}\nstderr:\n{}".format(
                        sycl_device, sycl_ls_sp.stdout, sycl_ls_sp.stderr
                    )
                )
            architectures = set()

    aspect_features = set("aspect-" + a for a in aspects)
    sg_size_features = set("sg-" + s for s in sg_sizes)
    architecture_feature = set("arch-" + s for s in architectures)
    # Add device family features like intel-gpu-gen12, intel-gpu-dg2 based on
    # the architecture reported by sycl-ls.
    device_family = set(
        get_device_family_from_arch(arch)
        for arch in architectures
        if get_device_family_from_arch(arch) is not None
    )

    # Print the detected GPU family name.
    if len(device_family) > 0:
        lit_config.note(
            "Detected GPU family for {}: {}".format(
                sycl_device, ", ".join(device_family)
            )
        )

    features.update(aspect_features)
    features.update(sg_size_features)
    features.update(architecture_feature)
    features.update(device_family)

    be, dev = sycl_device.split(":")
    features.add(dev.replace("fpga", "accelerator"))
    if "level_zero_v2" in full_name:
        features.add("level_zero_v2_adapter")
    elif "level_zero_v1" in full_name:
        features.discard("level_zero_v2_adapter")

    if "level_zero_v2_adapter" in features:
        lit_config.note("Using Level Zero V2 adapter for {}".format(sycl_device))

    # Use short names for LIT rules.
    features.add(be)
    # Add corresponding target feature
    target = config.backend_to_target[be]
    features.add(target)

    if be == "hip":
        if not config.amd_arch:
            # Guaranteed to be a single element in the set
            arch = [x for x in architecture_feature][0]
            amd_arch_prefix = "arch-amd_gpu_"
            if amd_arch_prefix not in arch or len(architecture_feature) != 1:
                lit_config.error(
                    "Cannot detect architecture for AMD HIP device, specify it explicitly"
                )
            config.amd_arch = arch.replace(amd_arch_prefix, "")

    config.sycl_dev_features[full_name] = features.union(config.available_features)
    if is_intel_driver:
        config.intel_driver_ver[full_name] = intel_driver_ver
    else:
        config.intel_driver_ver[full_name] = {}

if lit_config.params.get("compatibility_testing", "False") != "False":
    config.substitutions.append(("%clangxx", " true "))
    config.substitutions.append(("%clang", " true "))
else:
    clangxx = " " + config.dpcpp_compiler + " "
    if "preview-mode" in config.available_features:
        # Technically, `-fpreview-breaking-changes` is reported as unused option
        # if used without `-fsycl`. However, we have far less tests compiling
        # pure C++ (without `-fsycl`) than we have tests doing `%clangxx -fsycl`
        # and not relying on simple `%{build}`. As such, it's easier and less
        # error prone to silence the warning in those instances than to risk not
        # running some tests properly in the `test-preview-mode`.
        clangxx += "-fpreview-breaking-changes "
    clangxx += config.cxx_flags
    config.substitutions.append(("%clangxx", clangxx))

# Check that no runtime features are available when in build-only
from E2EExpr import E2EExpr

if config.test_mode == "build-only":
    E2EExpr.check_build_features(config.available_features)

if lit_config.params.get("print_features", False):
    lit_config.note(
        "Global features: {}".format(" ".join(sorted(config.available_features)))
    )
    lit_config.note("Per-device features:")
    for dev, features in config.sycl_dev_features.items():
        lit_config.note("\t{}: {}".format(dev, " ".join(sorted(features))))

# Set timeout for a single test
try:
    import psutil

    if config.test_mode == "run-only":
        lit_config.maxIndividualTestTime = 300
    else:
        lit_config.maxIndividualTestTime = 600

except ImportError:
    pass
