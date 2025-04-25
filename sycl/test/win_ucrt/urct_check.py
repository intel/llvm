# REQUIRES: windows
# RUN: %python %s %llvm_build_bin_dir
#
# Check for Windows URCT dependencies in library files in a specified path. If a
# library uses a the debug URCT it must be postfixed with either "d" or
# "d-preview".
#
import argparse
import os
import subprocess
import sys


def check_file(filepath):
    filename, file_ext = os.path.splitext(entry.name)

    # Only consider .dll or .lib files.
    if not (file_ext == ".dll" or file_ext == ".lib"):
        return

    has_debug_postfix = filename.endswith("d") or filename.endswith("d-preview")
    dep_output = subprocess.run(
        ["dumpbin", "/dependents", filepath], shell=False, capture_output=True
    )
    
    if str(dep_output.stdout).find("ucrtbased.dll"):
        if not has_debug_postfix:
            print("Unexpected use of ucrtbased.dll:", filepath)
            sys.exit(1)
    elif str(dep_output.stdout).find("ucrtbase.dll"):
        if has_debug_postfix:
            print("Unexpected use of ucrtbase.dll:", filepath)
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windows UCRT checker utility.")
    parser.add_argument("target_path", type=str)
    args = parser.parse_args()

    # Scan the path for library files.
    with os.scandir(args.target_path) as it:
        for entry in it:
            if entry.is_file():
                check_file(entry.name)
