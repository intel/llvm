"""
 Copyright (C) 2022 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import argparse
import os
import re
import subprocess
import sys

from mako.template import Template

HEADER_TEMPLATE = Template("""/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${file_name}.h
 *
 */

#include <map>
#include <string>
#include <vector>

namespace uur {
namespace device_binaries {
    std::map<std::string, std::vector<std::string>> program_kernel_map = {
% for program, entry_points in kernel_name_dict.items():
        {"${program}", {
  % for entry_point in entry_points:
            "${entry_point}",
  % endfor
        }},
% endfor
    };
}
}
""")


def generate_header(output_file, kernel_name_dict):
    """Render the template and write it to the output file."""
    file_name = os.path.basename(output_file)
    rendered = HEADER_TEMPLATE.render(file_name=file_name,
                                      kernel_name_dict=kernel_name_dict)
    rendered = re.sub(r"\r\n", r"\n", rendered)

    with open(output_file, "w") as fout:
        fout.write(rendered)


def get_mangled_names(dpcxx_path, source_file, output_header):
    """Return a list of all the entry point names from a given sycl source file.

    Filters out wrapper and offset handler entry points.
    """
    output_dir = os.path.dirname(output_header)
    il_file = os.path.join(output_dir, os.path.basename(source_file) + ".ll")
    generate_il_command = f"""\
        {dpcxx_path} -S -fsycl -fsycl-device-code-split=off \
        -fsycl-device-only -o {il_file} {source_file}"""
    subprocess.run(generate_il_command, shell=True)
    kernel_line_regex = re.compile("define.*spir_kernel")
    definition_lines = []
    with open(il_file) as f:
        lines = f.readlines()
        for line in lines:
            if kernel_line_regex.search(line) is not None:
                definition_lines.append(line)

    entry_point_names = []
    kernel_name_regex = re.compile(r"@(.*?)\(")
    for line in definition_lines:
        if kernel_name_regex.search(line) is None:
            continue
        kernel_name = kernel_name_regex.search(line).group(1)
        if "kernel_wrapper" not in kernel_name and "with_offset" not in kernel_name:
            entry_point_names.append(kernel_name)

    os.remove(il_file)
    return entry_point_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpcxx_path",
                        help="Full path to dpc++ compiler executable.")
    parser.add_argument(
        "-o",
        "--output",
        help="Full path to header file that will be generated.")
    parser.add_argument("source_files", nargs="+")
    args = parser.parse_args()

    mangled_names = {}

    for source_file in args.source_files:
        program_name = os.path.splitext(os.path.basename(source_file))[0]
        mangled_names[program_name] = get_mangled_names(
            args.dpcxx_path, source_file, args.output)
    generate_header(args.output, mangled_names)


if __name__ == "__main__":
    sys.exit(main())
