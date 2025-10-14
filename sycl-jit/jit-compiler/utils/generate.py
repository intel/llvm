import os
import argparse
import sys
import fnmatch
import glob

def main():
    parser = argparse.ArgumentParser(
        description="Generate SYCL Headers Resource C++ file."
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output C++ file")
    parser.add_argument("-i", "--toolchain-dir", type=str, required=True, help="Path to toolchain root directory.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for virtual file locations")
    parser.add_argument("-m", "--manifest-input", type=str, help="Build from this whitelist manifest (read-only).")
    parser.add_argument("--manifest-output", type=str, help="Glob for files and write them to this capture manifest.")
    parser.add_argument("--blacklist", type=str, help="Path to a file containing glob patterns of resources to exclude.")
    
    args = parser.parse_args()

    if args.manifest_input and args.manifest_output:
        print("Error: --manifest-input and --manifest-output are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    blacklist_patterns = set()
    if args.blacklist:
        print(f"Loading blacklist from: {args.blacklist}")
        with open(args.blacklist, "r") as f:
            for line in f:
                pattern = line.strip()
                if pattern and not pattern.startswith('#'):
                    blacklist_patterns.add(pattern)

    toolchain_dir = os.path.abspath(args.toolchain_dir)
    
    manifest_to_write = open(args.manifest_output, "w") if args.manifest_output else open(os.devnull, "w")

    with manifest_to_write as manifest_out, open(args.output, "w") as out:
        if args.manifest_output:
            preamble = f"""# This manifest was auto-geneerated by the sycl-jit build process
            # It contains the list of all candidate resource files found when globbing.
            #
            # If any of these files should NOT be included in the final library
            # (e.g. for IP reasons), add their relative path to the blacklist file at:
            # {args.blacklist}
            """
            manifest_out.write(preamble + '\n')

        out.write(
            """
#include <Resource.h>
namespace jit_compiler::resource {
const resource_file ToolchainFiles[] = {"""
        )

        def generate_cpp_for_file(absolute_path):
            relative_path = os.path.relpath(absolute_path, toolchain_dir)
            portable_relative_path = relative_path.replace(os.sep, '/')

            for pattern in blacklist_patterns:
                # Compare the pattern against the portable relative path
                if fnmatch.fnmatch(portable_relative_path, pattern):
                    print(f"  -> Skipping blacklisted file: {portable_relative_path}")
                    return None

            out.write(
                f"""
        {{
          {{"{args.prefix}{portable_relative_path}"}} ,
          []() {{
            static const char data[] = {{
            #embed "{absolute_path}" if_empty(0)
                , 0}};
            return resource_string_view{{data}};
          }}()
        }},"""
            )
            return portable_relative_path

        if args.manifest_input:
            print(f"Reading resource list from whitelist manifest: {args.manifest_input}")
            with open(args.manifest_input, "r") as manifest_file:
                for line in manifest_file:
                    relative_path = line.strip()
                    if relative_path:
                        absolute_path = os.path.join(toolchain_dir, relative_path)
                        generate_cpp_for_file(absolute_path)
        else:
            if args.manifest_output:
                print(f"Globbing for resources and writing capture manifest to: {args.manifest_output}")
            else:
                print("Globbing for resources (no capture manifest output)...")

            def process_and_log_file(absolute_path):
                relative_path = generate_cpp_for_file(absolute_path)
                if relative_path:
                    manifest_out.write(relative_path + '\n')

            def process_dir(dir):
                for root, _, files in os.walk(dir):
                    for file in files:
                        process_and_log_file(os.path.join(root, file))

            process_dir(os.path.join(args.toolchain_dir, "include/"))
            process_dir(os.path.join(args.toolchain_dir, "lib/clang/"))
            process_dir(os.path.join(args.toolchain_dir, "lib/clc/"))
            
            print("Recursively searching for .bc files in lib/...")
            lib_dir = os.path.join(args.toolchain_dir, "lib")
            search_pattern = os.path.join(lib_dir, "**", "*.bc")
            
            for file_path in glob.glob(search_pattern, recursive=True):
                process_and_log_file(file_path)

        out.write(
            f"""
}};

unsigned long long NumToolchainFiles = size(ToolchainFiles);
resource_string_view ToolchainPrefix{{"{args.prefix}"}};
}} // namespace jit_compiler::resource
"""
        )

if __name__ == "__main__":
    main()
    