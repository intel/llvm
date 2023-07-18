#! /usr/bin/env python3
"""
 Copyright (C) 2019-2021 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import argparse
import util
import generate_code
import os, sys
import time
import json

"""
    helper for adding mutually-exclusive boolean arguments "--name" and "--skip-name"
"""
def add_argument(parser, name, help, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, help="Enable "+help, action="store_true")
    group.add_argument("--skip-" + name, dest=name, help="Skip "+help, action="store_false")
    parser.set_defaults(**{name:default})

"""
    helpers to strip loader only api constructs from the json
"""
def strip_specs_class(specs, strip_class):
    for spec in specs:
        remove_obj = []
        for obj in spec["objects"]:
            if "class" in obj and strip_class in obj["class"]:
                remove_obj.append(obj)
        for obj in remove_obj:
            spec["objects"].remove(obj)

def strip_meta_entry(meta, entry_name, pattern):
    loader_entries = []
    for entry in meta[entry_name]:
        if pattern in entry:
            loader_entries.append(entry)

    for entry in loader_entries:
        del meta[entry_name][entry]

def strip_loader_meta(meta):
    strip_meta_entry(meta, "class", "Loader")
    strip_meta_entry(meta, "function", "Loader")
    strip_meta_entry(meta, "enum", "loader")
    strip_meta_entry(meta, "handle", "loader")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser, "lib", "generation of lib files.", True)
    add_argument(parser, "loader", "generation of loader files.", True)
    add_argument(parser, "layers", "generation of layer files.", True)
    add_argument(parser, "adapters", "generation of null adapter files.", True)
    add_argument(parser, "common", "generation of common files.", True)
    parser.add_argument("--debug", action='store_true', help="dump intermediate data to disk.")
    parser.add_argument("--sections", type=list, default=None, help="Optional list of sections for which to generate source, default is all")
    parser.add_argument("--ver", type=str, default="1.0", help="specification version to generate.")
    parser.add_argument('--api-json', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="JSON file containing the API specification, by default read from stdin")
    parser.add_argument("out_dir", type=str, help="Root of the loader repository.")
    args = parser.parse_args()

    input = json.loads(args.api_json.read())

    start = time.time()

    srcpath = os.path.join(args.out_dir, "source")

    for idx, specs in enumerate(input['specs']):
        config = input['configs'][idx]
        if args.sections == None or config['name'] in args.sections:
            if args.lib:
                generate_code.generate_lib(srcpath, config['name'], config['namespace'], config['tags'], args.ver, specs, input['meta'])
            # From here only generate code for functions adapters can implement.
            strip_specs_class(specs, "Loader")
            strip_loader_meta(input['meta'])
            if args.loader:
                generate_code.generate_loader(srcpath, config['name'], config['namespace'], config['tags'], args.ver, specs, input['meta'])
            if args.layers:
                generate_code.generate_layers(srcpath, config['name'], config['namespace'], config['tags'], args.ver, specs, input['meta'])
            if args.adapters:
                generate_code.generate_adapters(srcpath, config['name'], config['namespace'], config['tags'], args.ver, specs, input['meta'])
            if args.common:
                generate_code.generate_common(srcpath, config['name'], config['namespace'], config['tags'], args.ver, specs, input['meta'])

    if args.debug:
        util.makoFileListWrite("generated.json")

    print("\nCompleted in %.1f seconds!"%(time.time() - start))
