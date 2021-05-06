import argparse
import os
import platform
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

    llvm_external_projects = 'sycl;llvm-spirv;opencl;opencl-aot;libdevice;xpti;xptifw'

    llvm_dir = os.path.join(abs_src_dir, "llvm")
    sycl_dir = os.path.join(abs_src_dir, "sycl")
    spirv_dir = os.path.join(abs_src_dir, "llvm-spirv")
    xpti_dir = os.path.join(abs_src_dir, "xpti")
    xptifw_dir = os.path.join(abs_src_dir, "xptifw")
    libdevice_dir = os.path.join(abs_src_dir, "libdevice")
    llvm_targets_to_build = 'X86'
    llvm_enable_projects = 'clang;' + llvm_external_projects
    libclc_targets_to_build = ''
    sycl_build_pi_cuda = 'OFF'
    sycl_werror = 'ON'
    llvm_enable_assertions = 'ON'
    llvm_enable_doxygen = 'OFF'
    llvm_enable_sphinx = 'OFF'
    llvm_build_shared_libs = 'OFF'

    sycl_enable_xpti_tracing = 'ON'

    # replace not append, so ARM ^ X86
    if args.arm:
        llvm_targets_to_build = 'ARM;AArch64'

    if args.cuda:
        llvm_targets_to_build += ';NVPTX'
        llvm_enable_projects += ';libclc'
        libclc_targets_to_build = 'nvptx64--;nvptx64--nvidiacl'
        sycl_build_pi_cuda = 'ON'

    if args.no_werror:
        sycl_werror = 'OFF'

    if args.no_assertions:
        llvm_enable_assertions = 'OFF'

    if args.docs:
        llvm_enable_doxygen = 'ON'
        llvm_enable_sphinx = 'ON'

    if args.shared_libs:
        llvm_build_shared_libs = 'ON'

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
        "-DLLVM_ENABLE_PROJECTS={}".format(llvm_enable_projects),
        "-DLIBCLC_TARGETS_TO_BUILD={}".format(libclc_targets_to_build),
        "-DSYCL_BUILD_PI_CUDA={}".format(sycl_build_pi_cuda),
        "-DLLVM_BUILD_TOOLS=ON",
        "-DSYCL_ENABLE_WERROR={}".format(sycl_werror),
        "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
        "-DSYCL_INCLUDE_TESTS=ON", # Explicitly include all kinds of SYCL tests.
        "-DLLVM_ENABLE_DOXYGEN={}".format(llvm_enable_doxygen),
        "-DLLVM_ENABLE_SPHINX={}".format(llvm_enable_sphinx),
        "-DBUILD_SHARED_LIBS={}".format(llvm_build_shared_libs),
        "-DSYCL_ENABLE_XPTI_TRACING={}".format(sycl_enable_xpti_tracing)
    ]

    if args.l0_headers and args.l0_loader:
      cmake_cmd.extend([
            "-DL0_INCLUDE_DIR={}".format(args.l0_headers),
            "-DL0_LIBRARY={}".format(args.l0_loader)])
    elif args.l0_headers or args.l0_loader:
      sys.exit("Please specify both Level Zero headers and loader or don't specify "
               "none of them to let download from github.com")

    # Add additional CMake options if provided
    if args.cmake_opt:
      cmake_cmd += args.cmake_opt

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

    print("[Cmake Command]: {}".format(" ".join(cmake_cmd)))

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
    parser.add_argument("--arm", action='store_true', help="build ARM support rather than x86")
    parser.add_argument("--no-assertions", action='store_true', help="build without assertions")
    parser.add_argument("--docs", action='store_true', help="build Doxygen documentation")
    parser.add_argument("--no-werror", action='store_true', help="Don't treat warnings as errors")
    parser.add_argument("--shared-libs", action='store_true', help="Build shared libraries")
    parser.add_argument("--cmake-opt", action='append', help="Additional CMake option not configured via script parameters")
    parser.add_argument("--cmake-gen", default="Ninja", help="CMake generator")
    parser.add_argument("--use-libcxx", action="store_true", help="build sycl runtime with libcxx")
    parser.add_argument("--libcxx-include", metavar="LIBCXX_INCLUDE_PATH", help="libcxx include path")
    parser.add_argument("--libcxx-library", metavar="LIBCXX_LIBRARY_PATH", help="libcxx library path")
    args = parser.parse_args()

    print("args:{}".format(args))

    return do_configure(args)

if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)
