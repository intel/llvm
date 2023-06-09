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
    functions = list()
    etors = dict()

    for s in specs:
        for obj in s['objects']:
            if obj['type'] == 'function':
                functions.append(obj['class'][len('$x'):] + obj['name'])
    
    # find the function with the largest ID
    max_reg_entry = 0
    try:
        existing_registry = list(util.yamlRead(path))[1]['etors']
        max_reg_entry = int(max(existing_registry, key=lambda x: int(x['value']))['value'])
        for etor in existing_registry:
            etors[etor['name']] = etor['value']
    except (TypeError, IndexError, KeyError) as e:
        raise Exception('invalid existing registry... ' + str(e))

    new_registry = list()
    for fname in functions:
        etor_name = util.to_snake_case(fname)
        id = etors.get(etor_name)
        if id is None:
            max_reg_entry += 1
            id = max_reg_entry
        new_registry.append({'name': util.to_snake_case(fname), 'desc': 'Enumerator for $x'+fname, 'value': str(id)})


    print("Generating registry %s"%path)

    ids = new_registry
    ids = sorted(ids, key=lambda x: int(x['value']))
    wrapper = { 'name': ENUM_NAME, 'type': 'enum', 'desc': 'Defines unique stable identifiers for all functions' , 'etors': ids}
    header = {'type': 'header', 'desc': quoted('Intel $OneApi Unified Runtime function registry'), 'ordinal': quoted(9)}

    try:
        with open(path, 'w') as fout:
            yaml.add_representer(quoted, quoted_presenter)
            yaml.dump_all([header, wrapper], fout,
                default_flow_style=False,
                sort_keys=False,
                explicit_start=True)
    except:
        print("error: unable to write %s"%path)

