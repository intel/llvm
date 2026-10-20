import os
import argparse
import glob
import re


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
    parser.add_argument(
        "--shard-dir",
        type=str,
        required=True,
        help="Directory to write out-of-line shard files for embedded files",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=16,
        help="Number of out-of-line shard translation units to spread "
        "embedded files across. This is a fixed count (rather than one "
        "shard per file) so CMake can declare the shard compile commands "
        "at configure time, before generate.py has run.",
    )
    args = parser.parse_args()

    # abspath also strips trailing "/"
    toolchain_dir = os.path.abspath(args.toolchain_dir)

    os.makedirs(args.shard_dir, exist_ok=True)

    def collect_files(dir):
        for root, _, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                # We only need .bc files from libdevice:
                if re.search(r"[/\\]libsycl-.*\.(o|obj|spv)$", file_path):
                    continue
                yield file_path

    all_files = [
        (os.path.getsize(file_path), file_path)
        for file_path in collect_files(toolchain_dir)
    ]

    # Greedily bin-pack every embedded file into a fixed number of shards
    # (largest file first, always placed into the currently-lightest shard)
    # so the per-shard compile time is roughly balanced. This keeps all
    # #embed's out of resource.cpp itself, which otherwise would serialize
    # at the tail of the build with nothing left to overlap it with.
    all_files.sort(reverse=True)
    shard_sizes = [0] * args.num_shards
    shard_members = [[] for _ in range(args.num_shards)]
    for size, file_path in all_files:
        shard_index = shard_sizes.index(min(shard_sizes))
        shard_members[shard_index].append(file_path)
        shard_sizes[shard_index] += size

    externs = []
    entries = []
    symbol_index = 0
    for shard_index in range(args.num_shards):
        shard_path = os.path.join(args.shard_dir, f"resource-shard-{shard_index}.cpp")
        with open(shard_path, "w") as shard_out:
            for file_path in shard_members[shard_index]:
                symbol = f"ToolchainShardData{symbol_index}"
                symbol_index += 1
                size = os.path.getsize(file_path)
                array_size = size + 1  # trailing 0 appended below
                rel_path = os.path.relpath(file_path, toolchain_dir).replace(
                    os.sep, "/"
                )
                shard_out.write(f"""extern const char {symbol}[] = {{
#embed "{file_path}" if_empty(0)
    , 0}};
""")
                externs.append(f"extern const char {symbol}[{array_size}];")
                entries.append(f"""
{{
    {{"{args.prefix}{rel_path}"}} ,
    []() {{ return resource_string_view{{{symbol}}}; }}()
}},""")

    with open(args.output, "w") as out:
        out.write("\n#include <Resource.h>\n\n")
        for extern_decl in externs:
            out.write(extern_decl + "\n")
        out.write("""
namespace jit_compiler::resource {
const resource_file ToolchainFiles[] = {""")
        for entry in entries:
            out.write(entry)
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
