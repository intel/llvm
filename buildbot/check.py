import argparse
import os
import multiprocessing
import subprocess
import sys

DEFAULT_CPU_COUNT = 4

def do_check(args):
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = DEFAULT_CPU_COUNT

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
        args.test_suite,
        "-j", str(cpu_count)]

    print("[Cmake Command]: {}".format(" ".join(cmake_cmd)))

    env_tmp=os.environ
    env_tmp["LIT_ARGS"]="\"{}\"".format("-v")

    subprocess.check_call(cmake_cmd, cwd=abs_obj_dir, env=env_tmp)

    ret = True
    return ret

def main():
    parser = argparse.ArgumentParser(prog="check.py",
                                     description="script to do LIT testing",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--build-number", metavar="BUILD_NUM", help="build number")
    parser.add_argument("-b", "--branch", metavar="BRANCH", help="pull request branch")
    parser.add_argument("-d", "--base-branch", metavar="BASE_BRANCH", help="pull request base branch")
    parser.add_argument("-r", "--pr-number", metavar="PR_NUM", help="pull request number")
    parser.add_argument("-w", "--builder-dir", metavar="BUILDER_DIR",
                        help="builder directory, which is the directory containing source and build directories")
    parser.add_argument("-s", "--src-dir", metavar="SRC_DIR", help="source directory")
    parser.add_argument("-o", "--obj-dir", metavar="OBJ_DIR", help="build directory")
    parser.add_argument("-t", "--test-suite", metavar="TEST_SUITE", default="check-all", help="check-xxx target")

    args = parser.parse_args()

    print("args:{}".format(args))

    return do_check(args)

if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)
