# REQUIRES: windows
# RUN: %python %s %llvm_build_bin_dir
#
# Check for Windows URCT dependencies in library files in a specified path. If a
# library uses a the debug URCT it must be postfixed with "d".
#
import argparse
import os
import subprocess
import sys


# Some libraries are excluded from the check:
#  * OpenCL.dll: The OpenCL ICD loader is ignored as it is generally not part of
#                the compiler release packages.
exclude_list = {"OpenCL.dll"}


def check_file(filepath):
    filename, file_ext = os.path.splitext(entry.name)

    # Only consider .dll or .lib files.
    if not (file_ext == ".dll" or file_ext == ".lib"):
        return 0

    has_debug_postfix = filename.endswith("d")
    dep_output = subprocess.run(
        ["dumpbin", "/dependents", filepath], shell=False, capture_output=True
    )

    if str(dep_output.stdout).find("ucrtbased.dll") != -1:
        if not has_debug_postfix:
            print("Unexpected use of ucrtbased.dll:", filepath)
            return 1
    elif str(dep_output.stdout).find("ucrtbase.dll") != -1:
        if has_debug_postfix:
            print("Unexpected use of ucrtbase.dll:", filepath)
            return 1
    elif str(dep_output.stdout).find("api-ms-win-crt-") != -1:
        if has_debug_postfix:
            print("Unexpected use of api-ms-win-crt-*.dll: ", filepath)
            return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windows UCRT checker utility.")
    parser.add_argument("target_path", type=str)
    args = parser.parse_args()

    # Scan the path for library files.
    failures = 0
    with os.scandir(args.target_path) as it:
        for entry in it:
            if entry.is_file():
                failures += check_file(entry.name)
    if failures > 0:
        exit(failures)
