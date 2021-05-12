import argparse
import multiprocessing
import subprocess
import sys
import os

DEFAULT_CPU_COUNT = 4


def do_compile(args):
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = DEFAULT_CPU_COUNT

    if args.build_parallelism:
        cpu_count = int(args.build_parallelism)

    # Get absolute path to source directory
    if args.src_dir:
      abs_src_dir = os.path.abspath(args.src_dir)
    else:
      abs_src_dir = os.path.abspath(os.path.join(__file__, "../.."))
    # Get absolute path to build directory
    if args.obj_dir:
      abs_obj_dir = os.path.abspath(args.obj_dir)
    else:
      abs_obj_dir = os.path.join(abs_src_dir, "build")

    cmake_cmd = [
        "cmake",
        "--build", abs_obj_dir,
        "--",
        "deploy-sycl-toolchain",
        "-j", str(cpu_count)]

    if args.verbose:
      cmake_cmd.append("--verbose")

    print("[Cmake Command]: {}".format(" ".join(cmake_cmd)))

    subprocess.check_call(cmake_cmd, cwd=abs_obj_dir)

    return True


def main():
    parser = argparse.ArgumentParser(prog="compile.py",
                                     description="script to do compile",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--build-number", metavar="BUILD_NUM", help="build number")
    parser.add_argument("-b", "--branch", metavar="BRANCH", help="pull request branch")
    parser.add_argument("-d", "--base-branch", metavar="BASE_BRANCH", help="pull request base branch")
    parser.add_argument("-r", "--pr-number", metavar="PR_NUM", help="pull request number")
    parser.add_argument("-w", "--builder-dir", metavar="BUILDER_DIR",
                        help="builder directory, which is the directory containing source and build directories")
    parser.add_argument("-s", "--src-dir", metavar="SRC_DIR", help="source directory")
    parser.add_argument("-o", "--obj-dir", metavar="OBJ_DIR", help="build directory")
    parser.add_argument("-j", "--build-parallelism", metavar="BUILD_PARALLELISM", help="build parallelism")
    parser.add_argument("-v", "--verbose", action='store_true', help="verbose build output")

    args = parser.parse_args()

    print("args:{}".format(args))

    return do_compile(args)


if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)
