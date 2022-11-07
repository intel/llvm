"""
 Copyright (C) 2022 Intel Corporation

 SPDX-License-Identifier: MIT

"""
import os
import re
import util

"""
    generates c/c++ files from the specification documents
"""
def _mako_api_h(path, namespace, tags, version, revision, specs, meta):
    template = "api.h.mako"
    fin = os.path.join("templates", template)

    filename = "%s_api.h"%(namespace)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        ver=version,
        rev=revision,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

"""
    generates c/c++ files from the specification documents
"""
def _mako_api_cpp(path, namespace, tags, version, revision, specs, meta):
    template = "api.cpp.mako"
    fin = os.path.join("templates", template)

    filename = "%s_api.cpp"%(namespace)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        ver=version,
        rev=revision,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

"""
    generates c/c++ files from the specification documents
"""
def _mako_ddi_h(path, namespace, tags, version, revision, specs, meta):
    template = "ddi.h.mako"
    fin = os.path.join("templates", template)

    filename = "%s_ddi.h"%(namespace)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        ver=version,
        rev=revision,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

"""
    generates python files from the specification documents
"""
def _mako_api_py(path, namespace, tags, version, revision, specs, meta):
    template = "api.py.mako"
    fin = os.path.join("templates", template)

    filename = "%s.py"%(namespace)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        ver=version,
        rev=revision,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

"""
    generates c/c++ files from the specification documents
"""
def _generate_api_cpp(incpath, srcpath, namespace, tags, version, revision, specs, meta):
    loc = _mako_api_h(incpath, namespace, tags, version, revision, specs, meta)
    loc += _mako_api_cpp(srcpath, namespace, tags, version, revision, specs, meta)
    loc += _mako_ddi_h(incpath, namespace, tags, version, revision, specs, meta)

    return loc

"""
    generates python files from the specification documents
"""
def _generate_api_py(incpath, namespace, tags, version, revision, specs, meta):
    loc = _mako_api_py(incpath, namespace, tags, version, revision, specs, meta)
    return loc

"""
Entry-point:
    generates api code
"""
def generate_api(incpath, srcpath, namespace, tags, version, revision, specs, meta):
    util.makePath(incpath)
    util.makePath(srcpath)

    loc = 0
    loc += _generate_api_cpp(incpath, srcpath, namespace, tags, version, revision, specs, meta)
    loc += _generate_api_py(incpath, namespace, tags, version, revision, specs, meta)
    print("Generated %s lines of code.\n"%loc)
