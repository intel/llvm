import argparse
import os
import platform
import shlex
import subprocess
import sys

def do_configure(args):
    # Get absolute path to source directory
    abs_src_dir = os.path.abspath(args.src_dir if args.src_dir else os.path.join(__file__, "../.."))
    # Get absolute path to build directory
    abs_obj_dir = os.path.abspath(args.obj_dir) if args.obj_dir else os.path.join(abs_src_dir, "build")
    # Create build directory if it doesn't exist
    if not os.path.isdir(abs_obj_dir):
      os.makedirs(abs_obj_dir)

    llvm_external_projects = 'sycl;llvm-spirv;opencl;xpti;xptifw'

    # libdevice build requires a working SYCL toolchain, which is not the case
    # with macOS target right now.
    if sys.platform != "darwin":
        llvm_external_projects += ';libdevice'

    libclc_amd_target_names = ';amdgcn--amdhsa'
    libclc_nvidia_target_names = ';nvptx64--nvidiacl'

    sycl_enable_fusion = "OFF"
    if not args.disable_fusion:
        llvm_external_projects += ";sycl-fusion"
        sycl_enable_fusion = "ON"

    if args.llvm_external_projects:
        llvm_external_projects += ";" + args.llvm_external_projects.replace(",", ";")

    llvm_dir = os.path.join(abs_src_dir, "llvm")
    sycl_dir = os.path.join(abs_src_dir, "sycl")
    spirv_dir = os.path.join(abs_src_dir, "llvm-spirv")
    xpti_dir = os.path.join(abs_src_dir, "xpti")
    xptifw_dir = os.path.join(abs_src_dir, "xptifw")
    libdevice_dir = os.path.join(abs_src_dir, "libdevice")
    fusion_dir = os.path.join(abs_src_dir, "sycl-fusion")
    llvm_targets_to_build = args.host_target
    llvm_enable_projects = 'clang;' + llvm_external_projects
    libclc_targets_to_build = ''
    libclc_gen_remangled_variants = 'OFF'
    sycl_build_pi_hip_platform = 'AMD'
    sycl_clang_extra_flags = ''
    sycl_werror = 'OFF'
    llvm_enable_assertions = 'ON'
    llvm_enable_doxygen = 'OFF'
    llvm_enable_sphinx = 'OFF'
    llvm_build_shared_libs = 'OFF'
    llvm_enable_lld = 'OFF'
    sycl_enabled_plugins = ["opencl"]
    sycl_preview_lib = 'ON'

    sycl_enable_xpti_tracing = 'ON'
    xpti_enable_werror = 'OFF'

    if sys.platform != "darwin":
        sycl_enabled_plugins.append("level_zero")

    # lld is needed on Windows or for the HIP plugin on AMD
    if platform.system() == 'Windows' or (args.hip and args.hip_platform == 'AMD'):
        llvm_enable_projects += ';lld'

    if args.cuda or args.hip or args.native_cpu:
        llvm_enable_projects += ';libclc'

    if args.cuda:
        llvm_targets_to_build += ';NVPTX'
        libclc_targets_to_build = libclc_nvidia_target_names
        libclc_gen_remangled_variants = 'ON'
        sycl_enabled_plugins.append("cuda")

    if args.hip:
        if args.hip_platform == 'AMD':
            llvm_targets_to_build += ';AMDGPU'
            libclc_targets_to_build += libclc_amd_target_names

        elif args.hip_platform == 'NVIDIA' and not args.cuda:
            llvm_targets_to_build += ';NVPTX'
            libclc_targets_to_build += libclc_nvidia_target_names
        libclc_gen_remangled_variants = 'ON'

        sycl_build_pi_hip_platform = args.hip_platform
        sycl_enabled_plugins.append("hip")

    if args.native_cpu:
        #Todo: we should set whatever targets we support for native cpu
        libclc_targets_to_build += ';x86_64-unknown-linux-gnu'
        sycl_enabled_plugins.append("native_cpu")


    # all llvm compiler targets don't require 3rd party dependencies, so can be
    # built/tested even if specific runtimes are not available
    if args.enable_all_llvm_targets:
        llvm_targets_to_build += ';NVPTX;AMDGPU'

    if args.werror or args.ci_defaults:
        sycl_werror = 'ON'
        xpti_enable_werror = 'ON'

    if args.no_assertions:
        llvm_enable_assertions = 'OFF'

    if args.docs:
        llvm_enable_doxygen = 'ON'
        llvm_enable_sphinx = 'ON'

    if args.shared_libs:
        llvm_build_shared_libs = 'ON'

    if args.use_lld:
        llvm_enable_lld = 'ON'

    # CI Default conditionally appends to options, keep it at the bottom of
    # args handling
    if args.ci_defaults:
        print("#############################################")
        print("# Default CI configuration will be applied. #")
        print("#############################################")

        # For clang-format, clang-tidy and code coverage
        llvm_enable_projects += ";clang-tools-extra;compiler-rt"
        if sys.platform != "darwin":
            # libclc is required for CI validation
            if 'libclc' not in llvm_enable_projects:
                llvm_enable_projects += ';libclc'
            # libclc passes `--nvvm-reflect-enable=false`, build NVPTX to enable it
            if 'NVPTX' not in llvm_targets_to_build:
                llvm_targets_to_build += ';NVPTX'
            # since we are building AMD libclc target we must have AMDGPU target
            if 'AMDGPU' not in llvm_targets_to_build:
                llvm_targets_to_build += ';AMDGPU'
            # Add both NVIDIA and AMD libclc targets
            if libclc_amd_target_names not in libclc_targets_to_build:
                libclc_targets_to_build += libclc_amd_target_names
            if libclc_nvidia_target_names not in libclc_targets_to_build:
                libclc_targets_to_build += libclc_nvidia_target_names
            libclc_gen_remangled_variants = 'ON'

    if args.enable_plugin:
        sycl_enabled_plugins += args.enable_plugin

    if args.disable_preview_lib:
        sycl_preview_lib = 'OFF'

    install_dir = os.path.join(abs_obj_dir, "install")

    cmake_cmd = [
        "cmake",
        "-G", args.cmake_gen,
        "-DCMAKE_BUILD_TYPE={}".format(args.build_type),
        "-DLLVM_ENABLE_ASSERTIONS={}".format(llvm_enable_assertions),
        "-DLLVM_TARGETS_TO_BUILD={}".format(llvm_targets_to_build),
        "-DLLVM_EXTERNAL_PROJECTS={}".format(llvm_external_projects),
        "-DLLVM_EXTERNAL_SYCL_SOURCE_DIR={}".format(sycl_dir),
        "-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR={}".format(spirv_dir),
        "-DLLVM_EXTERNAL_XPTI_SOURCE_DIR={}".format(xpti_dir),
        "-DXPTI_SOURCE_DIR={}".format(xpti_dir),
        "-DLLVM_EXTERNAL_XPTIFW_SOURCE_DIR={}".format(xptifw_dir),
        "-DLLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR={}".format(libdevice_dir),
        "-DLLVM_EXTERNAL_SYCL_FUSION_SOURCE_DIR={}".format(fusion_dir),
        "-DLLVM_ENABLE_PROJECTS={}".format(llvm_enable_projects),
        "-DLIBCLC_TARGETS_TO_BUILD={}".format(libclc_targets_to_build),
        "-DLIBCLC_GENERATE_REMANGLED_VARIANTS={}".format(libclc_gen_remangled_variants),
        "-DSYCL_BUILD_PI_HIP_PLATFORM={}".format(sycl_build_pi_hip_platform),
        "-DLLVM_BUILD_TOOLS=ON",
        "-DSYCL_ENABLE_WERROR={}".format(sycl_werror),
        "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
        "-DSYCL_INCLUDE_TESTS=ON", # Explicitly include all kinds of SYCL tests.
        "-DLLVM_ENABLE_DOXYGEN={}".format(llvm_enable_doxygen),
        "-DLLVM_ENABLE_SPHINX={}".format(llvm_enable_sphinx),
        "-DBUILD_SHARED_LIBS={}".format(llvm_build_shared_libs),
        "-DSYCL_ENABLE_XPTI_TRACING={}".format(sycl_enable_xpti_tracing),
        "-DLLVM_ENABLE_LLD={}".format(llvm_enable_lld),
        "-DXPTI_ENABLE_WERROR={}".format(xpti_enable_werror),
        "-DSYCL_CLANG_EXTRA_FLAGS={}".format(sycl_clang_extra_flags),
        "-DSYCL_ENABLE_PLUGINS={}".format(';'.join(set(sycl_enabled_plugins))),
        "-DSYCL_ENABLE_KERNEL_FUSION={}".format(sycl_enable_fusion),
        "-DSYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB={}".format(sycl_preview_lib),
        "-DBUG_REPORT_URL=https://github.com/intel/llvm/issues",
    ]

    if args.l0_headers and args.l0_loader:
      cmake_cmd.extend([
            "-DLEVEL_ZERO_INCLUDE_DIR={}".format(args.l0_headers),
            "-DLEVEL_ZERO_LIBRARY={}".format(args.l0_loader)])
    elif args.l0_headers or args.l0_loader:
      sys.exit("Please specify both Level Zero headers and loader or don't specify "
               "none of them to let download from github.com")

    # Add additional CMake options if provided
    if args.cmake_opt:
      cmake_cmd += args.cmake_opt
    
    if args.add_security_flags:
      cmake_cmd.extend(["-DEXTRA_SECURITY_FLAGS={}".format(args.add_security_flags)])

    # Add path to root CMakeLists.txt
    cmake_cmd.append(llvm_dir)

    if args.use_libcxx:
      if not (args.libcxx_include and args.libcxx_library):
        sys.exit("Please specify include and library path of libc++ when building sycl "
                 "runtime with it")
      cmake_cmd.extend([
            "-DSYCL_USE_LIBCXX=ON",
            "-DSYCL_LIBCXX_INCLUDE_PATH={}".format(args.libcxx_include),
            "-DSYCL_LIBCXX_LIBRARY_PATH={}".format(args.libcxx_library)])

    print("[Cmake Command]: {}".format(" ".join(map(shlex.quote, cmake_cmd))))

    try:
        subprocess.check_call(cmake_cmd, cwd=abs_obj_dir)
    except subprocess.CalledProcessError:
        cmake_cache = os.path.join(abs_obj_dir, "CMakeCache.txt")
        if os.path.isfile(cmake_cache):
           print("There is CMakeCache.txt at " + cmake_cache +
             " ... you can try to remove it and rerun.")
           print("Configure failed!")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(prog="configure.py",
                                     description="Generate build files from CMake configuration files",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # CI system options
    parser.add_argument("-n", "--build-number", metavar="BUILD_NUM", help="build number")
    parser.add_argument("-b", "--branch", metavar="BRANCH", help="pull request branch")
    parser.add_argument("-d", "--base-branch", metavar="BASE_BRANCH", help="pull request base branch")
    parser.add_argument("-r", "--pr-number", metavar="PR_NUM", help="pull request number")
    parser.add_argument("-w", "--builder-dir", metavar="BUILDER_DIR",
                        help="builder directory, which is the directory containing source and build directories")
    # User options
    parser.add_argument("-s", "--src-dir", metavar="SRC_DIR", help="source directory (autodetected by default)")
    parser.add_argument("-o", "--obj-dir", metavar="OBJ_DIR", help="build directory. (<src>/build by default)")
    parser.add_argument("--l0-headers", metavar="L0_HEADER_DIR", help="directory with Level Zero headers")
    parser.add_argument("--l0-loader", metavar="L0_LOADER", help="path to the Level Zero loader")
    parser.add_argument("-t", "--build-type",
                        metavar="BUILD_TYPE", default="Release", help="build type: Debug, Release")
    parser.add_argument("--cuda", action='store_true', help="switch from OpenCL to CUDA")
    parser.add_argument("--native_cpu", action='store_true', help="Enable SYCL Native CPU")
    parser.add_argument("--hip", action='store_true', help="switch from OpenCL to HIP")
    parser.add_argument("--hip-platform", type=str, choices=['AMD', 'NVIDIA'], default='AMD', help="choose hardware platform for HIP backend")
    parser.add_argument("--host-target", default='X86',
                        help="host LLVM target architecture, defaults to X86, multiple targets may be provided as a semi-colon separated string")
    parser.add_argument("--enable-all-llvm-targets", action='store_true', help="build compiler with all supported targets, it doesn't change runtime build")
    parser.add_argument("--no-assertions", action='store_true', help="build without assertions")
    parser.add_argument("--docs", action='store_true', help="build Doxygen documentation")
    parser.add_argument("--werror", action='store_true', help="Treat warnings as errors")
    parser.add_argument("--shared-libs", action='store_true', help="Build shared libraries")
    parser.add_argument("--cmake-opt", action='append', help="Additional CMake option not configured via script parameters")
    parser.add_argument("--cmake-gen", default="Ninja", help="CMake generator")
    parser.add_argument("--use-libcxx", action="store_true", help="build sycl runtime with libcxx")
    parser.add_argument("--libcxx-include", metavar="LIBCXX_INCLUDE_PATH", help="libcxx include path")
    parser.add_argument("--libcxx-library", metavar="LIBCXX_LIBRARY_PATH", help="libcxx library path")
    parser.add_argument("--use-lld", action="store_true", help="Use LLD linker for build")
    parser.add_argument("--llvm-external-projects", help="Add external projects to build. Add as comma seperated list.")
    parser.add_argument("--ci-defaults", action="store_true", help="Enable default CI parameters")
    parser.add_argument("--enable-plugin", action='append', help="Enable SYCL plugin")
    parser.add_argument("--disable-preview-lib", action='store_true', help="Disable building of the SYCL runtime major release preview library")
    parser.add_argument("--disable-fusion", action="store_true", help="Disable the kernel fusion JIT compiler")
    parser.add_argument("--add_security_flags", type=str, choices=['none', 'default', 'sanitize'], default=None, help="Enables security flags for compile & link. Two values are supported: 'default' and 'sanitize'. 'Sanitize' option is an extension of 'default' set.")
    args = parser.parse_args()

    print("args:{}".format(args))

    return do_configure(args)

if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)
