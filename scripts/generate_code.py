"""
 Copyright (C) 2022 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    generates c/c++ files from the mako template
"""
def _mako_print_h(path, namespace, tags, version, specs, meta):
    template = "print.h.mako"
    fin = os.path.join("templates", template)

    filename = "%s_print.h"%(namespace)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

"""
    generates c/c++ files from the mako template
"""
def _mako_print_cpp(path, namespace, tags, version, specs, meta):
    template = "print.cpp.mako"
    fin = os.path.join("templates", template)

    filename = "%s_print.cpp"%(namespace)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)


def _mako_api_funcs(path, namespace, tags, version, revision, specs, meta):
    template = "api_funcs.def.mako"
    fin = os.path.join(templates_dir, template)

    name = "%s_api_funcs"%(namespace)
    filename = "%s.def"%(name)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        name=name,
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
    loc += _mako_print_hpp(incpath, namespace, tags, version, revision, specs, meta)
    loc += _mako_api_funcs(incpath, namespace, tags, version, revision, specs, meta)

    return loc

"""
Entry-point:
    generates api code
"""
def generate_api(incpath, srcpath, namespace, tags, version, revision, specs, meta):
    util.makePath(incpath)
    util.makePath(srcpath)

    loc = 0
    loc += _mako_print_h(incpath, namespace, tags, version, specs, meta)
    loc += _generate_api_cpp(incpath, srcpath, namespace, tags, version, revision, specs, meta)
    print("Generated %s lines of code.\n"%loc)

templates_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates")

"""
    generates c/c++ files from the specification documents
"""
def _mako_lib_cpp(path, namespace, tags, version, specs, meta):
    loc = 0
    template = "libapi.cpp.mako"
    fin = os.path.join(templates_dir, template)

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
    fin = os.path.join(templates_dir, template)

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
    fin = os.path.join(templates_dir, template)

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
    fin = os.path.join(templates_dir, template)

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
def _mako_mock_adapter_cpp(path, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "mock")
    os.makedirs(dstpath, exist_ok=True)

    template = "mockddi.cpp.mako"
    fin = os.path.join(templates_dir, template)

    name = "%s_mockddi"%(namespace)
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
    generates c/c++ files from the specification documents
"""
def _mako_validation_layer_cpp(path, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "validation")
    os.makedirs(dstpath, exist_ok=True)

    template = "valddi.cpp.mako"
    fin = os.path.join(templates_dir, template)

    name = "%s_valddi"%(namespace)
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
    generates c/c++ files from the specification documents
"""
def _mako_tracing_layer_cpp(path, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "tracing")
    os.makedirs(dstpath, exist_ok=True)

    template = "trcddi.cpp.mako"
    fin = os.path.join(templates_dir, template)

    name = "%s_trcddi"%(namespace)
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
    generates c/c++ files from the specification documents
"""
def _mako_print_hpp(path, namespace, tags, version, revision, specs, meta):
    template = "print.hpp.mako"
    fin = os.path.join(templates_dir, template)

    name = "%s_print"%(namespace)
    filename = "%s.hpp"%(name)
    fout = os.path.join(path, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        name=name,
        ver=version,
        rev=revision,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta)

"""
Entry-point:
    generates tools code
"""
def _mako_info_hpp(path, namespace, tags, version, specs, meta):
    fin = os.path.join(templates_dir, "tools-info.hpp.mako")
    name = f"{namespace}info"
    filename = f"{name}.hpp"
    fout = os.path.join(path, filename)
    print("Generating %s..." % fout)
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
    generates linker version scripts
"""
def _mako_linker_scripts(path, name, ext, namespace, tags, version, specs, meta):
    filename = f"{name}.{ext}.in"
    fin = os.path.join(templates_dir, f"{filename}.mako")
    fout = os.path.join(path, filename)
    print("Generating %s..." % fout)
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
    generates loader for unified_runtime adapter
"""
def generate_loader(path, section, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "loader")
    os.makedirs(dstpath, exist_ok=True)

    loc = 0
    loc += _mako_loader_cpp(dstpath, namespace, tags, version, specs, meta)
    loc += _mako_print_cpp(dstpath, namespace, tags, version, specs, meta)
    loc += _mako_linker_scripts(
        dstpath, "loader", "map", namespace, tags, version, specs, meta
    )
    loc += _mako_linker_scripts(
        dstpath, "loader", "def", namespace, tags, version, specs, meta
    )
    print("Generated %s lines of code.\n"%loc)

"""
    generates c/c++ files from the specification documents
"""
def _mako_interface_loader_api(path, adapter, ext, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, adapter)
    os.makedirs(dstpath, exist_ok=True)

    template = f"ur_interface_loader.{ext}.mako"
    fin = os.path.join(templates_dir, template)

    name = f"ur_interface_loader"

    filename = f"{name}.{ext}"
    fout = os.path.join(dstpath, filename)

    print("Generating %s..."%fout)
    return util.makoWrite(
        fin, fout,
        name=name,
        adapter=adapter,
        ver=version,
        namespace=namespace,
        tags=tags,
        specs=specs,
        meta=meta,)

"""
Entry-point:
    generates adapter for unified_runtime
"""
def generate_adapters(path, section, namespace, tags, version, specs, meta):
    dstpath = os.path.join(path, "adapters")
    os.makedirs(dstpath, exist_ok=True)

    loc = 0
    loc += _mako_mock_adapter_cpp(dstpath, namespace, tags, version, specs, meta)
    loc += _mako_linker_scripts(
        dstpath, "adapter", "map", namespace, tags, version, specs, meta
    )
    loc += _mako_linker_scripts(
        dstpath, "adapter", "def", namespace, tags, version, specs, meta
    )

    loc += _mako_interface_loader_api(dstpath, "level_zero", "cpp", namespace, tags, version, specs, meta)
    loc += _mako_interface_loader_api(dstpath, "level_zero", "hpp", namespace, tags, version, specs, meta)

    print("Generated %s lines of code.\n"%loc)

"""
Entry-point:
    generates layers for unified_runtime adapter
"""
def generate_layers(path, section, namespace, tags, version, specs, meta):
    print("GL section %s\n"%section)
    print("GL namespace %s\n"%namespace)
    layer_dstpath = os.path.join(path, "loader", "layers")
    include_dstpath = os.path.join(path, "../include")
    os.makedirs(layer_dstpath, exist_ok=True)
    os.makedirs(include_dstpath, exist_ok=True)

    loc = 0
    loc += _mako_validation_layer_cpp(layer_dstpath, namespace, tags, version, specs, meta)
    print("VALIDATION Generated %s lines of code.\n"%loc)

    loc = 0
    loc += _mako_tracing_layer_cpp(layer_dstpath, namespace, tags, version, specs, meta)
    print("TRACING Generated %s lines of code.\n"%loc)

"""
Entry-point:
    generates common utilities for unified_runtime
"""
def generate_common(path, section, namespace, tags, version, specs, meta):
    template = "stype_map_helpers.hpp.mako"
    fin = os.path.join("templates", template)

    filename = "stype_map_helpers.def"
    layer_dstpath = os.path.join(path, "common")
    os.makedirs(layer_dstpath, exist_ok=True)
    fout = os.path.join(layer_dstpath, filename)

    print("Generating %s..." % fout)

    loc = util.makoWrite(
            fin, fout,
            ver=version,
            namespace=namespace,
            tags=tags,
            specs=specs,
            meta=meta)
    print("COMMON Generated %s lines of code.\n" % loc)


"""
Entry-point:
    generates tools for unified_runtime
"""
def generate_tools(path, section, namespace, tags, version, specs, meta):
    loc = 0

    infodir = os.path.join(path, f"{namespace}info")
    os.makedirs(infodir, exist_ok=True)
    loc += _mako_info_hpp(infodir, namespace, tags, version, specs, meta)

    print("TOOLS Generated %s lines of code.\n" % loc)

"""
Entry-point:
    generates API functions that accept queue for level_zero
"""
def generate_level_zero_queue_api(path, section, namespace, tags, version, specs, meta):
    template = "queue_api.cpp.mako"
    fin = os.path.join("templates", template)

    name = "queue_api"
    filename = "queue_api.cpp"
    layer_dstpath = os.path.join(path, "adapters", "level_zero", "v2")
    os.makedirs(layer_dstpath, exist_ok=True)
    fout = os.path.join(layer_dstpath, filename)

    print("Generating %s..." % fout)

    loc = util.makoWrite(
            fin, fout,
            ver=version,
            name = name,
            namespace=namespace,
            tags=tags,
            specs=specs,
            meta=meta)

    template = "queue_api.hpp.mako"
    fin = os.path.join("templates", template)

    filename = "queue_api.hpp"
    fout = os.path.join(layer_dstpath, filename)

    print("Generating %s..." % fout)

    loc += util.makoWrite(
            fin, fout,
            ver=version,
            name = name,
            namespace=namespace,
            tags=tags,
            specs=specs,
            meta=meta)
    print("QUEUE Generated %s lines of code.\n" % loc)

"""
Entry-point:
    generates headers used by the CTS, for example containing meta-information
    about info query enums
"""
def generate_cts_headers(path, section, namespace, tags, version, specs, meta):
    template = "optional_queries.h.mako"
    fin = os.path.join("templates", template)
    name = "optional_queries"
    filename = "optional_queries.h"
    dstpath = os.path.join(path, "conformance", "testing", "include", "uur")
    fout  = os.path.join(dstpath, filename)

    print("Generating %s..." % fout)

    loc = util.makoWrite(fin, fout,
            filename = name, namespace = namespace,
            tags = tags, specs = specs, meta = meta)

    print("CTS Generated %s lines of code.\n" % loc)
