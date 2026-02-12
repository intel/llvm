# Copyright (C) 2022 Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import re
import subprocess
import sys

HEADER_TEMPLATE = """/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file %s.h
 *
 */

#include <map>
#include <string>
#include <vector>

namespace uur {
namespace device_binaries {
    std::map<std::string, std::vector<std::string>> program_kernel_map = {
%s
    };
}
}
"""

PROGRAM_TEMPLATE = """\
        {"%s", {
%s
        }},
"""

ENTRY_POINT_TEMPLATE = """\
            "%s",
"""


def generate_header(output_file, kernel_name_dict):
    """Render the template and write it to the output file."""
    file_name = os.path.basename(output_file)
    device_binaries = ""
    for program, entry_points in kernel_name_dict.items():
        content = ""
        for entry_point in entry_points:
            content += ENTRY_POINT_TEMPLATE % entry_point
        device_binaries += PROGRAM_TEMPLATE % (program, content)
    rendered = HEADER_TEMPLATE % (file_name, device_binaries)
    rendered = re.sub(r"\r\n", r"\n", rendered)
    with open(output_file, "w") as fout:
        fout.write(rendered)


def get_mangled_names(source_file, output_header):
    """Return a list of all the entry point names from a given sycl source file.

    Filters out wrapper and offset handler entry points.
    """
    output_dir = os.path.dirname(output_header)
    name = os.path.splitext(os.path.basename(source_file))[0]
    ih_file = os.path.join(output_dir, name, name + ".ih")
    definitions = []
    writing = False
    with open(ih_file) as f:
        lines = f.readlines()
        for line in lines:
            if "}" in line and writing:
                break
            # __pf_kernel_wrapper seems to be an internal function used by dpcpp
            if writing and "19__pf_kernel_wrapper" not in line:
                definitions.append(line.replace(",", "").strip()[1:-1])
            if "const char* const kernel_names[] = {" in line:
                writing = True

    return definitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", help="Full path to header file that will be generated."
    )
    parser.add_argument("source_files", nargs="+")
    args = parser.parse_args()

    mangled_names = {}

    for source_file in args.source_files:
        program_name = os.path.splitext(os.path.basename(source_file))[0]
        mangled_names[program_name] = get_mangled_names(source_file, args.output)
    generate_header(args.output, mangled_names)


if __name__ == "__main__":
    sys.exit(main())
