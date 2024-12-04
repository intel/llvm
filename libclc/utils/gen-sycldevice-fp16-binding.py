#!/usr/bin/env python3
#===----------------------------------------------------------------------===
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===

#
# Generate the SPIR-V builtin declaration from the JSON format
# output by clang-tblgen --gen-clang-progmodel-builtins-as-json
#

import argparse
import subprocess
import json
import sys
import io
import re


def ignore_function(fun):
    ignorelist = [
        "GenericCastToPtrExplicit",
        "GenericPtrMemSemantics",
        "GroupAll",
        "GroupAny",
        "GroupBroadcast",
        "GroupFAdd",
        "GroupFMax",
        "GroupFMin",
        "GroupIAdd",
        "GroupSMax",
        "GroupSMin",
        "GroupUMax",
        "GroupUMin",
        "printf"
    ]
    return any([fun.find(b) != -1 for b in ignorelist])

def ignore(overload):
    ignorelist = ["__private", "__generic"]

    def is_ignored(s):
        return any([s.find(b) != -1 for b in ignorelist])

    return any([is_ignored(s) for s in overload]) or all([s.find('fp16') == -1  for s in overload[1:]])


function_attributes = {
    "const": "_CLC_CONSTFN",
    "convergent": "_CLC_CONVERGENT",
    "pure": "_CLC_PURE",
    "variadic": ""
}

# Assign weight to types to allow stable sorting
type_weight_map = {}


def build_type_weight():
    weight = 0

    def add_type(ty, weight):
        type_weight_map[ty] = weight
        return weight + 1

    weight = add_type("void", weight)
    weight = add_type("__clc_bool_t", weight)
    weight = add_type("__clc_event_t", weight)
    # Those are to handle __clc_event_t __private* and __clc_event_t __generic* overloads
    weight = add_type("__clc_event_t __private", weight)
    weight = add_type("__clc_event_t __generic", weight)
    weight = add_type("__clc_size_t", weight)
    for ty in [
            "char", "int8", "int16", "int32", "int64", "uint8", "uint16",
            "uint32", "uint64", "fp16", "fp32", "fp64", "float16"
    ]:
        for vlen in [1, 2, 3, 4, 8, 16]:
            for asp in [
                    "", " __private", " __local", " __global", " __constant", " __generic"
            ]:
                vec = "_vec{}".format(str(vlen)) if vlen != 1 else ""
                weight = add_type(
                    "__clc{VEC}_{TY}_t{ASP}".format(TY=ty, VEC=vec, ASP=asp),
                    weight)


build_type_weight()


def overload_requires(overload, ext):
    return any([ty.find(ext) != -1 for ty in overload])


def overload_requires_fp64(overload):
    return overload_requires(overload, "fp64")

def overload_requires_fp16(overload):
    return overload_requires(overload, "fp16")

def overload_requires_float16(overload):
    return overload_requires(overload, "float16")

def expand_overload(overload_list, func):
    """
    Allow some extra overload to ease integrations with OpenCL.
    """
    return overload_list
    # new_overload_list = list()
    # for overload, attr in overload_list:
    #     new_overload_list.append([list([ty.replace("_fp16", "_float16") for ty in overload]),
    #                               attr])
    #     # new_overload_list.append([list([ty.replace("_fp16", "_float16") for ty in overload]),
    #     #                           attr])
    # return new_overload_list


def remove_ignored_overload(overload_list):
    """
    Allow some extra overload to ease integrations with OpenCL.
    """
    new_overload_list = list()
    for overload in overload_list:
        if ignore(overload[0]):
            continue
        new_overload_list.append(overload)
    return new_overload_list


def sort_overload(overload_list):
    """
    Sort overloads and group them by used extension.
    """

    # Sort overloads
    def strip_ptr(ty):
        return ty.replace('*', '').replace(' const', '').strip()

    nb_types = len(overload_list[0][0])
    for ty_idx in reversed(range(nb_types)):
        overload_list = sorted(
            overload_list,
            key=lambda x: type_weight_map[strip_ptr(x[0][ty_idx])])
    # 0 -> no extension
    # 1 -> use fp64
    # 2 -> use fp16
    # 3 -> use fp64 and fp16
    # 4 -> use float16
    new_overload_list = list([list() for _ in range(8)])
    for overload, attr in overload_list:
        idx = overload_requires_fp64(
            overload) + 2 * overload_requires_fp16(overload) + 4 * overload_requires_float16(overload)
        new_overload_list[idx].append([overload, attr])
    return new_overload_list


def emit_guards(fd, overload):
    """
    Emits guards according the function type.
    Returns the number of emitted guards.
    """
    requires_double = overload_requires_fp64(overload)
    nb_guards = 0
    if requires_double:
        nb_guards += 1
        fd.write("#ifdef cl_khr_fp64\n")
    return nb_guards


def close_guards(fd, nb_guards):
    """
    Emits 'nb_guards' guards closing statement.
    """
    for _ in range(nb_guards):
        fd.write("#endif\n")

to_as_fp16 = {
    "__clc_float16_t" : "as_half",
    "__clc_vec2_float16_t" : "as_half2",
    "__clc_vec3_float16_t" : "as_half3",
    "__clc_vec4_float16_t" : "as_half4",
    "__clc_vec8_float16_t" : "as_half8",
    "__clc_vec16_float16_t" : "as_half16",
    "__clc_fp16_t" : "as_half",
    "__clc_vec2_fp16_t" : "as_half2",
    "__clc_vec3_fp16_t" : "as_half3",
    "__clc_vec4_fp16_t" : "as_half4",
    "__clc_vec8_fp16_t" : "as_half8",
    "__clc_vec16_fp16_t" : "as_half16"
    }

def get_as_fp16(ty):
    if ty.find("fp16") != -1:
        if ty.find("*") != -1:
            return "({})".format(ty)
        return to_as_fp16[ty]
    return ""

def format_argument(i, ty):
    arg = "args_{}".format(str(i))
    if ty.find("fp16") != -1:
        as_prefix = get_as_fp16(ty)
        if as_prefix != "":
            arg = "{}({})".format(as_prefix, arg)
    return arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
Generate SPIR-V interface header.
Typical usage:
 clang-tblgen --gen-clang-progmodel-builtins-as-json path/to/SPIRVBuiltins.td \
   --json-type-prefix=__clc -o builtin.json
 generate-spirv-header.py builtin.json -format clang-format -o spirv_builtin.h
""")
    parser.add_argument("input",
                        metavar="PATH",
                        type=argparse.FileType('r'),
                        help="Path to builtin json")
    parser.add_argument("-format",
                        metavar="clang-format",
                        nargs='?',
                        help="clang-format the output file")
    parser.add_argument("-guard",
                        metavar="NAME",
                        default="CLC_SPIRV_BINDING",
                        help="Preprocessor guard")
    parser.add_argument("-o", metavar="PATH", help="Outfile")
    args = parser.parse_args()
    print(args.o)

    with args.input as f:
        mapping = json.load(f)
    keys = list(mapping.keys())
    keys.sort()
    with open(args.o, 'w') as out_fd:
        out_fd.write(
            """//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#ifdef cl_khr_fp16
#ifdef __CLC_HAS_FLOAT16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

""")
        for k in keys:
            if ignore_function(k):
                continue
            fn_desc = mapping[k]
            fn_desc = expand_overload(fn_desc, k)
            fn_desc = remove_ignored_overload(fn_desc)
            if len(fn_desc) == 0:
                continue
            fn_desc_by_ext = sort_overload(fn_desc)
            for overloads in fn_desc_by_ext:
                if len(overloads) == 0:
                    continue
                nb_guards = emit_guards(out_fd, overloads[0][0])
                for proto, fn_attrs in overloads:
                    ret = proto[0]
                    proto = proto[1:]
                    attr = " ".join(
                            ["_CLC_OVERLOAD", " _CLC_DEF"] +
                            [function_attributes[attr] for attr in fn_attrs])
                    param = {
                        "ATTR": attr,
                        "RET" : ret,
                        "FN"  : k,
                        "PARAM" : ", ".join(["{} args_{}".format(ty.replace('fp16', 'float16'), str(i)) for i, ty in enumerate(proto)]),
                        "RETURN" : "return" if ret != "void" else "",
                        "ARG" : ", ".join([format_argument(i, ty) for i, ty in enumerate(proto)])
                    }
                    out_fd.write("""{ATTR} {RET} {FN}({PARAM}) {{
  {RETURN} {FN}({ARG});
}}

""".format(**param))

                close_guards(out_fd, nb_guards)
                out_fd.write("\n")
        out_fd.write("#endif\n#endif\n")
    if args.format:
        # The ouput of clang-format is not stable, so we have to run the format twice
        subprocess.check_output([args.format, "-i", args.o])
        subprocess.check_output([args.format, "-i", args.o])
