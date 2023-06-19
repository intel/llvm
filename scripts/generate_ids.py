"""
 Copyright (C) 2023 Intel Corporation
 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 Generates a unique id for each spec function that doesn't have it.
"""

from fileinput import FileInput
import util
import yaml

ENUM_NAME = '$x_function_t'

class quoted(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

def generate_registry(path, specs):
    try:
        existing_registry = list(util.yamlRead(path))[1]['etors']
        existing_etors = {etor["name"]: etor["value"] for etor in existing_registry}
        max_etor = int(max(existing_registry, key = lambda x : int(x["value"]))["value"])
        functions = [obj['class'][len('$x'):] + obj['name'] for s in specs for obj in s['objects'] if obj['type'] == 'function']
        registry = list()
        for fname in functions:
            etor_name = util.to_snake_case(fname).upper()
            id = existing_etors.get(etor_name)
            if id is None:
                max_etor += 1
                id = max_etor
            registry.append({'name': util.to_snake_case(fname).upper(), 'desc': 'Enumerator for $x'+fname, 'value': str(id)})
        registry = sorted(registry, key=lambda x: int(x['value']))
        wrapper = { 'name': ENUM_NAME, 'type': 'enum', 'desc': 'Defines unique stable identifiers for all functions' , 'etors': registry}
        header = {'type': 'header', 'desc': quoted('Intel $OneApi Unified Runtime function registry'), 'ordinal': quoted(9)}
        with open(path, 'w') as fout:
            yaml.add_representer(quoted, quoted_presenter)
            yaml.dump_all([header, wrapper], fout,
                default_flow_style=False,
                sort_keys=False,
                explicit_start=True)
    except BaseException as e:
        print("Failed to generate registry.yml... %s", e)
        raise e
