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

loader_templates_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates")

"""
    generates c/c++ files from the specification documents
"""
def _mako_lib_cpp(path, namespace, tags, version, specs, meta):
    loc = 0
    template = "libapi.cpp.mako"
    fin = os.path.join(loader_templates_dir, template)

    name = "%s_libapi"%(namespace)
    filename = "%s.cpp"%(name)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    loc += util.makoWrite(
        fin, fout,
        name = name,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta = meta)

    template = "libddi.cpp.mako"
    fin = os.path.join(loader_templates_dir, template)

    name = "%s_libddi"%(namespace)
    filename = "%s.cpp"%(name)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    loc += util.makoWrite(
        fin, fout,
        name=name,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)
    return loc

"""
    generates c/c++ files from the specification documents
"""
def _mako_loader_cpp(path, namespace, tags, version, specs, meta):
    print("make_loader_cpp path %s namespace %s version %s\n" %(path, namespace, version))
    loc = 0
    template = "ldrddi.hpp.mako"
    fin = os.path.join(loader_templates_dir, template)

    name = "%s_ldrddi"%(namespace)
    filename = "%s.hpp"%(name)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    loc += util.makoWrite(
        fin, fout,
        name=name,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

    template = "ldrddi.cpp.mako"
    fin = os.path.join(loader_templates_dir, template)

    name = "%s_ldrddi"%(namespace)
    filename = "%s.cpp"%(name)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    loc += util.makoWrite(
        fin, fout,
        name=name,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)
    return loc

"""
    generates c/c++ files from the specification documents
"""
def _mako_null_driver_cpp(path, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "null")
    os.makedirs(dstpath, exist_ok=True)

    template = "nullddi.cpp.mako"
    fin = os.path.join(loader_templates_dir, template)

    name = "%s_nullddi"%(namespace)
    filename = "%s.cpp"%(name)
    fout = os.path.join(dstpath, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        name=name,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

"""
Entry-point:
    generates lib code
"""
def generate_lib(path, section, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "loader") # lib code lives alongside the loader
    os.makedirs(dstpath, exist_ok=True)

    loc = 0
    loc += _mako_lib_cpp(dstpath, namespace, tags, version, specs, meta)
    print("Generated %s lines of code.\n"%loc)

"""
Entry-point:
    generates loader for unified_runtime driver
"""
def generate_loader(path, section, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "loader")
    os.makedirs(dstpath, exist_ok=True)

    loc = 0
    loc += _mako_loader_cpp(dstpath, namespace, tags, version, specs, meta)
    print("Generated %s lines of code.\n"%loc)

"""
Entry-point:
    generates drivers for unified_runtime driver
"""
def generate_drivers(path, section, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "drivers")
    os.makedirs(dstpath, exist_ok=True)

    loc = 0
    loc += _mako_null_driver_cpp(dstpath, namespace, tags, version, specs, meta)
    print("Generated %s lines of code.\n"%loc)

