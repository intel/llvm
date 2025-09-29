import os
import argparse
import glob


def main():
    parser = argparse.ArgumentParser(
        description="Generate SYCL Headers Resource C++ file"
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    parser.add_argument(
        "-i",
        "--toolchain-dir",
        type=str,
        required=True,
        help="Path to toolchain root directory",
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Prefix for file locations"
    )
    args = parser.parse_args()

    # abspath also strips trailing "/"
    toolchain_dir = os.path.abspath(args.toolchain_dir)

    with open(args.output, "w") as out:
        out.write(
            """
#include <Resource.h>

namespace jit_compiler::resource {
const resource_file ToolchainFiles[] = {"""
        )

        def process_file(file_path):
            out.write(
                f"""
{{
    {{"{args.prefix}{os.path.relpath(file_path, toolchain_dir).replace(os.sep, "/")}"}} ,
    []() {{
    static const char data[] = {{
    #embed "{file_path}" if_empty(0)
        , 0}};
    return resource_string_view{{data}};
    }}()
}},"""
            )

        def process_dir(dir):
            for root, _, files in os.walk(dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    process_file(file_path)

        process_dir(os.path.join(args.toolchain_dir, "include/"))
        process_dir(os.path.join(args.toolchain_dir, "lib/clang/"))
        process_dir(os.path.join(args.toolchain_dir, "lib/clc/"))

        for file in glob.iglob(
            "*.bc", root_dir=os.path.join(args.toolchain_dir, "lib")
        ):
            file_path = os.path.join(args.toolchain_dir, "lib", file)
            process_file(file_path)

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
