"""
 Copyright (C) 2023 Intel Corporation
 SPDX-License-Identifier: MIT
 Generates a unique id for each spec function that doesn't have it.
"""

from fileinput import FileInput
import util
import yaml

MAX_FUNCS = 1024 # We could go up to UINT32_MAX...
ENUM_NAME = '$x_function_t'
valid_ids = set(range(1, MAX_FUNCS))

class quoted(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

def generate_registry(path, specs):
    functions = list()
    etors = dict()

    for s in specs:
        for i, obj in enumerate(s['objects']):
            if obj['type'] == 'function':
                functions.append(obj['class'][len('$x'):] + obj['name'])

    try:
        existing_registry = list(util.yamlRead(path))[1]['etors']
        for etor in existing_registry:
            valid_ids.discard(etor['value'])
            etors[etor['name']] = etor['value']
    except (TypeError, IndexError, KeyError) as e:
        print('invalid existing registry, ignoring... ' + str(e))

    updated = False

    new_registry = list()
    for fname in functions:
        etor_name = util.to_snake_case(fname)
        id = etors.get(etor_name)
        if id is None:
            updated = True
            id = valid_ids.pop()
        new_registry.append({'name': util.to_snake_case(fname), 'desc': 'Enumerator for $x'+fname, 'value': str(id)})

    if updated is False:
        return

    print("Updating registry %s"%path)

    ids = new_registry
    wrapper = { 'name': ENUM_NAME, 'type': 'enum', 'desc': 'Defines unique stable identifiers for all functions' , 'etors': ids}
    header = {'type': 'header', 'desc': quoted('Intel$OneApi Unified Rutime function registry'), 'ordinal': quoted(9)}

    try:
        with open(path, 'w') as fout:
            yaml.add_representer(quoted, quoted_presenter)
            yaml.dump_all([header, wrapper], fout,
                default_flow_style=False,
                sort_keys=False,
                explicit_start=True)
    except:
        print("error: unable to write %s"%path)

