# !__INTEL_PREVIEW_BREAKING_CHANGES
# This script is intended to generate "device_aspect_macros.h" from "aspects.def" and "aspects_deprecated.def".
# else
# This script is intended to generate "device_aspect_macros.h" from "aspects.def".
# end !__INTEL_PREVIEW_BREAKING_CHANGES

import os
import sys


# !__INTEL_PREVIEW_BREAKING_CHANGES
def process_aspects(file_path, is_deprecated=False):
# else
# def process_aspects(file_path):
# end !__INTEL_PREVIEW_BREAKING_CHANGES
    with open(file_path, "r") as file:
        content = file.read()
    lines = content.strip().splitlines()

    output = ""
    for line in lines:
        # __INTEL_PREVIEW_BREAKING_CHANGES
        stripped_line = line.strip()
        # !__INTEL_PREVIEW_BREAKING_CHANGES
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#") or stripped_line.startswith("//"):
            continue
        if is_deprecated: 
            aspect_macro = (
                stripped_line.replace("__SYCL_ASPECT_DEPRECATED(", "").replace(")", "")
            )
            aspect_name, aspect_number, _ = aspect_macro.split(
                ", ", 2
            )  # ignore the third parameter (message)
            output += f"// __SYCL_ASPECT_DEPRECATED({aspect_name}, {aspect_number})\n"
        else:
            aspect_macro = stripped_line.replace("__SYCL_ASPECT(", "").replace(")", "")
            aspect_name, aspect_number = aspect_macro.split(", ")
            output += f"// __SYCL_ASPECT({aspect_name}, {aspect_number})\n"
		# else !__INTEL_PREVIEW_BREAKING_CHANGES
		# aspect_macro = line.strip().replace("__SYCL_ASPECT(", "").replace(")", "")
		# aspect_name, aspect_number = aspect_macro.split(", ")
		# output += f"// __SYCL_ASPECT({aspect_name}, {aspect_number})\n"
		# end !__INTEL_PREVIEW_BREAKING_CHANGES

        output += f"#ifndef __SYCL_ALL_DEVICES_HAVE_{aspect_name}__\n"
        output += f"#define __SYCL_ALL_DEVICES_HAVE_{aspect_name}__ 0\n"
        output += "#endif\n"
        output += f"#ifndef __SYCL_ANY_DEVICE_HAS_{aspect_name}__\n"
        output += f"#define __SYCL_ANY_DEVICE_HAS_{aspect_name}__ 0\n"
        output += "#endif\n\n"

    return output


header_output = """//==------------------- device_aspect_macros.hpp - SYCL device -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// IMPORTANT: device_aspect_macros.hpp is a generated file - DO NOT EDIT
//            original definitions are in aspects.def & aspects_deprecated.def
//

#pragma once\n
"""

include_sycl_dir = sys.argv[1]
header_output += process_aspects(
    os.path.join(include_sycl_dir, "info/aspects_deprecated.def"), is_deprecated=True
)
header_output += process_aspects(os.path.join(include_sycl_dir, "info/aspects.def"))

build_include_sycl_dir = sys.argv[2]
with open(
    os.path.join(build_include_sycl_dir, "device_aspect_macros.hpp"), "w"
) as header_file:
    header_file.write(header_output)
