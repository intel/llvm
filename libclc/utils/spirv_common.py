#!/usr/bin/env python3
#===----------------------------------------------------------------------===
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===

import subprocess

def ignore_overload(overload):
    blacklist = ["__generic", "__private"]

    def is_black_listed(s):
        return any([s.find(b) != -1 for b in blacklist])

    return any([is_black_listed(s) for s in overload])

def overload_requires(overload, ext):
    return any([ty.find(ext) != -1 for ty in overload])


def overload_requires_fp64(overload):
    return overload_requires(overload, "fp64")


def overload_requires_fp16(overload):
    return overload_requires(overload, "fp16")

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

def clang_format(format_tool, out_file):
    # The ouput of clang-format is not stable, so we have to run the format twice
    subprocess.check_output([format_tool, "-i", out_file])
    subprocess.check_output([format_tool, "-i", out_file])
