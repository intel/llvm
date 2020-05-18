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
    whitelist = [
        "_abs",
        "_abs_diff",
        "add_sat",
        "clamp",
        "ConvertFToS",
        "ConvertFToU",
        "ConvertSToF",
        "ConvertUToF",
        "fclamp",
        "FConvert",
        "hadd",
        "mad24",
        "mad_hi",
        "mad_sat",
        "mul24",
        "mul_hi",
        "rhadd",
        "s_max",
        "s_min",
        "SatConvertSToU",
        "SatConvertUToS",
        "SConvert",
        "sub_sat",
        "u_max",
        "u_min",
        "UConvert",
        "upsample"
    ]
    return not any([fun.find(b) != -1 for b in whitelist])


def ignore(overload):
    blacklist = ["__generic", "__private"]

    def is_black_listed(s):
        return any([s.find(b) != -1 for b in blacklist])

    return any([is_black_listed(s) for s in overload])


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

    floats = ["fp16", "fp32", "fp64"]
    weight = add_type("void", weight)
    weight = add_type("__clc_bool_t", weight)
    weight = add_type("__clc_event_t", weight)
    weight = add_type("__clc_size_t", weight)
    for ty in [
            "char", "int8", "int16", "int32", "int64", "uint8", "uint16",
            "uint32", "uint64", "fp16", "fp32", "fp64"
    ]:
        for vlen in [1, 2, 3, 4, 8, 16]:
            for asp in [
                    "", " __private", " __local", " __global", " __constant"
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


def expand_overload(overload_list):
    """
    Allow some extra overload to ease integrations with OpenCL.
    """
    new_overload_list = list()
    for overload, attr in overload_list:
        new_overload_list.append([overload, attr])
        # Do this only it doesn't not create ambiguities,
        # so ignore the return type for the test.
        if any([ty.find("_int8") != -1 for ty in overload[1:]]):
            new_overload_list.append([
                list([ty.replace("_int8", "_char") for ty in overload]), attr
            ])
    return new_overload_list


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
    new_overload_list = list([list() for _ in range(4)])
    for overload, attr in overload_list:
        idx = overload_requires_fp64(
            overload) + 2 * overload_requires_fp16(overload)
        new_overload_list[idx].append([overload, attr])
    return new_overload_list


def emit_guards(fd, overload):
    """
    Emits guards according the function type.
    Returns the number of emitted guards.
    """
    requires_half = overload_requires_fp16(overload)
    requires_double = overload_requires_fp64(overload)
    nb_guards = 0
    if requires_half:
        nb_guards += 1
        fd.write("#ifdef cl_khr_fp16\n")
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

//
// Automatically generated file, do not edit!
//

#include <func.h>
#include <spirv/spirv_types.h>

#ifndef {GUARD}
#define {GUARD}

""".format(GUARD=args.guard))
        for k in keys:
            if ignore_function(k):
                continue
            fn_desc = mapping[k]
            # expand int8 to char so we have
            # mapping for char, signed char, unsigned char
            fn_desc = expand_overload(fn_desc)
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
                    param = {
                        "ATTR":
                        " ".join(
                            ["_CLC_OVERLOAD", "_CLC_DECL"] +
                            [function_attributes[attr] for attr in fn_attrs]),
                        "RET":
                        ret,
                        "FN":
                        k,
                        "PARAM":
                        ", ".join(proto),
                        "VARIADIC":
                        ", ..." if "variadic" in fn_attrs else ""
                    }
                    out_fd.write(
                        "{ATTR} {RET} {FN}({PARAM}{VARIADIC});\n".format(
                            **param))
                close_guards(out_fd, nb_guards)
                out_fd.write("\n")
        out_fd.write("#endif\n")
    if args.format:
        # The ouput of clang-format is not stable, so we have to run the format twice
        subprocess.check_output([args.format, "-i", args.o])
        subprocess.check_output([args.format, "-i", args.o])
