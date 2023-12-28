"""
 Copyright (C) 2022-2023 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import os
import generate_ids
import util
import re
import hashlib
import json
import yaml
import copy
from templates.helper import param_traits, type_traits, value_traits
import ctypes
import itertools

default_version = "0.9"
all_versions = ["0.6", "0.7", "0.8", "0.9"]

"""
    preprocess object
"""
def _preprocess(d):
    if 'enum' == d['type']:
        use_hex = False
        next = 0
        for etor in d['etors']:
            if type_traits.is_flags(d['name']):
                etor['name'] = "%s_%s"%(d['name'][:-3].upper(), etor['name'])
                etor['value'] = etor.get('value',"$X_BIT(%s)"%next)
                next = int(value_traits.get_bit_count(etor['value']))+1
            else:
                etor['name'] = d['name'][:-1].upper() + etor['name']
                use_hex = use_hex or value_traits.is_hex(etor.get('value'))
                if use_hex:
                    etor['value'] = etor.get('value',"%s"%hex(next))
                    next = int(etor['value'], 16)+1
                elif not value_traits.is_ver(etor.get('value')):
                    etor['value'] = etor.get('value',"%s"%next)
                    next = int(etor['value'])+1
    return d

"""
    substitute tags
"""
def _subt(name, tags):
    name = re.sub(r"(\w+)\(.*\)", r"\1", name) # removes '()' part of macros
    for key, value in tags.items():
        name = re.sub(re.escape(key), value, name)
        name = re.sub(re.escape(key.upper()), value.upper(), name)
    return name

"""
    get the line number of each document
"""
def _get_line_nums(f):
    nums = []
    for line_num, line in enumerate(util.textRead(f)):
        if re.match(r"^\-\-\-.*", line):
            nums.append(line_num+2)
    return nums

"""
    convert etor to int
"""
def _get_etor_value(value, prev):
    if value:
        if value_traits.is_ver(value):
            return (value_traits.get_major_ver(value) << 16) + value_traits.get_minor_ver(value)
        elif value_traits.is_bit(value):
            return 1 << value_traits.get_bit_count(value)
        elif value_traits.is_hex(value):
            return int(value, 16)
        else:
            return int(value)
    else:
        return prev+1

"""
    validate documents meet some basic (easily detectable) requirements of code generation
"""
def _validate_doc(f, d, tags, line_num, meta):
    is_iso = lambda x : re.match(r"[_a-zA-Z][_a-zA-Z0-9]{0,30}", x)

    def __validate_ordinal(d):
        if 'ordinal' in d:
            if not isinstance(d['ordinal'], str):
                raise Exception("'ordinal' must be a string: '%s'"%type(d['ordinal']))

            try:
                ordinal = str(int(d['ordinal']))
            except:
                ordinal = None

            if ordinal != d['ordinal']:
                raise Exception("'ordinal' invalid value: '%s'"%d['ordinal'])

    def __validate_version(d, prefix="", base_version=default_version):
        if 'version' in d:
            if not isinstance(d['version'], str):
                raise Exception(prefix+"'version' must be a string: '%s'"%type(d['version']))

            try:
                version = str(float(d['version']))
            except:
                version = None

            if version != d['version']:
                raise Exception(prefix+"'version' invalid value: '%s'"%d['version'])

        return float(d.get('version', base_version))

    def __validate_tag(d, key, tags, case):
        for x in tags:
            if d[key].startswith(x.upper() if case == 'upper' else x):
                return x
        return None

    def __validate_desc(desc):
        if isinstance(desc, dict):
            for k, v in desc.items():
                if not isinstance(k, str):
                    raise Exception(prefix+"'version' must be a string: '%s'"%type(k))

                try:
                    version = str(float(k))
                except:
                    version = None

                if version != k:
                    raise Exception(prefix+"'version' invalid value: '%s'"%k)

                for x in ['[in]', '[out]', '[in,out]']:
                    if v.startswith(x):
                        return x
        else:
            for x in ['[in]', '[out]', '[in,out]']:
                if desc.startswith(x):
                    return x
        return None

    def __validate_name(d, key, tags, case='lower', prefix=""):
        if not isinstance(d[key], str):
            raise Exception(prefix+"'%s' must be a string: '%s'"%(key, type(d[key])))

        if not __validate_tag(d, key, tags, case):
            raise Exception(prefix+"'%s' must start with {%s}: '%s'"%(key, ", ".join([x.upper() if case == 'upper' else x for x in tags if x != "$OneApi"]), d[key]))

        name = _subt(d[key], tags)
        if not is_iso(name):
            raise Exception(prefix+"'%s' must be ISO-C: '%s'"%(key, d[key]))

        if case == 'upper' and not name.isupper():
            raise Exception(prefix+"%s' must be upper case: '%s'"%(key, d[key]))
        elif case == 'lower' and not name.islower():
            raise Exception(prefix+"'%s' must be lower case: '%s'"%(key, d[key]))
        elif case == 'camel' and (name.isupper() or name.islower()):
            raise Exception(prefix+"'%s' must be camel case: '%s'"%(key, d[key]))

    def __validate_type(d, key, tags):
        __validate_name(d, key, tags)

        if not d[key].endswith("_t"):
            raise Exception("'%s' must end with '_t': '%s'"%(key, d[key]))

    def __validate_handle(d, key, tags):
        __validate_type(d, key, tags)

        if not d[key].endswith("handle_t"):
            raise Exception("'%s' must end with 'handle_t': '%s'"%(key, d[key]))

    def __validate_details(d):
        if 'details' in d:

            if not (isinstance(d['details'], list) or isinstance(d['details'], str)):
                raise Exception("'details' must be a string or a sequence")

            if isinstance(d['details'], list):
                for i, item in enumerate(d['details']):
                    prefix = "'details'[%s] "%i
                    if isinstance(item, dict):
                        for key in item:
                            if not isinstance(key, str):
                                raise Exception(prefix+"must be a string: '%s'"%type(key))

                            for j, val in enumerate(item[key]):
                                prefix2 = prefix[:-1]+"[%s] "%j
                                if not isinstance(val, str):
                                    raise Exception(prefix2+"must be a string: '%s'"%type(val))

                    elif not isinstance(item, str):
                        raise Exception(prefix+"must be a string: '%s'"%type(item))

    def extract_type(s):
        match = re.match(r'^\[(.+)\]\s', s)
        if match:
            return match.group(1)
        else:
            return None

    def __validate_etors(d, tags):
        if 'etors' not in d:
            raise Exception("'enum' requires the following sequence of mappings: {`etors`}")

        if not isinstance(d['etors'], list):
            raise Exception("'etors' must be a sequence: '%s'"%type(d['etors']))

        typed = d.get('typed_etors', False)

        value = -1
        d_ver = d.get('version', default_version)
        max_ver = float(d_ver)
        for i, item in enumerate(d['etors']):
            prefix="'etors'[%s] "%i
            if not isinstance(item, dict):
                raise Exception(prefix+"must be a mapping: '%s'"%(i, type(item)))

            if ('desc' not in item) or ('name' not in item):
                raise Exception(prefix+"requires the following scalar fields: {`desc`, `name`}")

            if 'extend' in d and d.get('extend') == True and 'value' not in item:
                raise Exception(prefix+"must include a value for experimental features: {`value`: `0xabcd`}")

            if typed:
                type = extract_type(item['desc'])
                if type is None:
                    raise Exception(prefix+"typed etor " + item['name'] + " must begin with a type identifier: [type]")
                type_name = _subt(type, tags)
                if not is_iso(type_name):
                    raise Exception(prefix+"type " + str(type) + " in a typed etor " + item['name'] + " must be a valid ISO C identifier")

            __validate_name(item, 'name', tags, case='upper', prefix=prefix)

            value = _get_etor_value(item.get('value'), value)
            if type_traits.is_flags(d['name']) and not value_traits.is_bit(item.get('value')):
                raise Exception(prefix+"'value' must use BIT macro: %s"%value)
            elif not type_traits.is_flags(d['name']) and value_traits.is_bit(item.get('value')):
                raise Exception(prefix+"'value' must not use BIT macro: %s"%value)

            if value >= 0x7fffffff:
                raise Exception(prefix+"'value' must be less than 0x7fffffff: %s"%value)

            ver = __validate_version(item, prefix=prefix, base_version=d_ver)
            if item.get('value'):
                max_ver = ver
            if ver < max_ver:
                raise Exception(prefix+"'version' must be increasing: %s"%item['version'])
            max_ver = ver

    def __validate_base(d):
        namespace = re.sub(r"(\$[a-z])\w+", r"\1", d['name'])
        valid_names = [
            "%s_base_desc_t"%namespace,
            "%s_base_properties_t"%namespace,
            "%s_driver_extension_properties_t"%namespace
            ]
        if d['name'] not in valid_names:
            if type_traits.is_descriptor(d['name']) and not d.get('base', "").endswith("base_desc_t"):
                raise Exception("'base' must be '%s_base_desc_t': %s"%(namespace, d['name']))

            elif type_traits.is_properties(d['name']) and not d.get('base', "").endswith("base_properties_t"):
                raise Exception("'base' must be '%s_base_properties_t': %s"%(namespace, d['name']))

    def __validate_struct_range_members(name, members, meta):
        def has_handle(members, meta):
            for m in members:
                if type_traits.is_handle(m):
                    return True
                if type_traits.is_struct(m, meta):
                    return has_handle(
                        type_traits.get_struct_members(m['type']), meta)
            return False

        for m in members:
            if param_traits.is_range(m) and type_traits.is_handle(m['type']):
                raise Exception(
                    f"struct range {name} must not contain range of object handles {m['name']}"
                )
            if type_traits.is_struct(m['type'], meta):
                member_members = type_traits.get_struct_members(
                    m['type'], meta)
                # We can't handle a range of structs with handles within a range of structs
                if param_traits.is_range(m) and has_handle(
                        member_members, meta):
                    raise Exception(
                        f"struct range {m['name']} is already within struct range {name}, and must not contain an object handle"
                    )
                # We keep passing the original name so we can report it in
                # exception messages.
                __validate_struct_range_members(name, member_members, meta)

    def __validate_members(d, tags, meta):
        if 'members' not in d:
            raise Exception("'%s' requires the following sequence of mappings: {`members`}"%d['type'])

        if not isinstance(d['members'], list):
            raise Exception("'members' must be a sequence: '%s'"%type(d['members']))

        d_ver = d.get('version', default_version)
        max_ver = float(d_ver)
        for i, item in enumerate(d['members']):
            prefix="'members'[%s] "%i
            if not isinstance(item, dict):
                raise Exception(prefix+"must be a mapping: '%s'"%(i, type(item)))

            if ('desc' not in item) or ('type' not in item) or ('name' not in item):
                raise Exception(prefix+"requires the following scalar fields: {`desc`, 'type', `name`}")

            annotation = __validate_desc(item['desc'])
            if not annotation:
                raise Exception(prefix+"'desc' must start with {'[in]', '[out]', '[in,out]'}")

            if type_traits.is_handle(item['type']):
                raise Exception(prefix+"'type' must not be '*_handle_t': %s"%item['type'])

            if item['type'].endswith("flag_t"):
                raise Exception(prefix+"'type' must not be '*_flag_t': %s"%item['type'])

            if d['type'] == 'union'and item.get('tag') is None:
                raise Exception(prefix + f"union member {item['name']} must include a 'tag' annotation")

            if type_traits.is_struct(item['type'],
                                     meta) and param_traits.is_range(item):
                member_members = type_traits.get_struct_members(
                    item['type'], meta)
                __validate_struct_range_members(item['name'], member_members,
                                                meta)

            ver = __validate_version(item, prefix=prefix, base_version=d_ver)
            if ver < max_ver:
                raise Exception(prefix+"'version' must be increasing: %s"%item['version'])
            max_ver = ver

    def __validate_params(d, tags, meta):
        if 'params' not in d:
            raise Exception("'function' requires the following sequence of mappings: {`params`}")

        if not isinstance(d['params'], list):
            raise Exception("'params' must be a sequence: '%s'"%type(d['params']))

        d_ver = d.get('version', default_version)
        max_ver = float(d_ver)
        min = {'[in]': None, '[out]': None, '[in,out]': None}
        for i, item in enumerate(d['params']):
            prefix="'params'[%s] "%i
            if not isinstance(item, dict):
                raise Exception(prefix+"must be a mapping: '%s'"%(i, type(item)))

            if ('desc' not in item) or ('type' not in item) or ('name' not in item):
                raise Exception(prefix+"requires the following scalar fields: {`desc`, 'type', `name`}")

            annotation = __validate_desc(item['desc'])
            if not annotation:
                raise Exception(prefix+"'desc' must start with {'[in]', '[out]', '[in,out]'}")

            if not min[annotation]:
                min[annotation] = i

            if min['[out]'] and ("[in]" == annotation or "[in,out]" == annotation):
                raise Exception(prefix+"'%s' must come before '[out]'"%annotation)

            if d.get('decl') != "static" and i == 0 and not type_traits.is_handle(item['type']):
                raise Exception(prefix+"'type' must be '*_handle_t': %s"%item['type'])

            if item['type'].endswith("flag_t"):
                raise Exception(prefix+"'type' must not be '*_flag_t': %s"%item['type'])

            if type_traits.is_pointer(item['type']) and "_handle_t" in item['type'] and "[in]" in item['desc']:
                if not param_traits.is_range(item):
                    raise Exception(prefix+"handle type must include a range(start, end) as part of 'desc'")

            if param_traits.is_bounds(item):
                has_queue = False
                for p in d['params']:
                    if re.match(r"hQueue$", p['name']):
                        has_queue = True

                if not has_queue:
                    raise Exception(prefix+"bounds must only be used on entry points which take a `hQueue` parameter")

            if type_traits.is_struct(item['type'],
                                     meta) and param_traits.is_range(item):
                members = type_traits.get_struct_members(item['type'], meta)
                __validate_struct_range_members(item['name'], members, meta)

            ver = __validate_version(item, prefix=prefix, base_version=d_ver)
            if ver < max_ver:
                raise Exception(prefix+"'version' must be increasing: %s"%item['version'])
            max_ver = ver

    def __validate_union_tag(d):
        if d.get('tag') is None:
            raise Exception(f"{d['name']} must include a 'tag' part of the union.")

    try:
        if 'type' not in d:
            raise Exception("every document must have 'type'")

        if not isinstance(d['type'], str):
            raise Exception("'type' must be a string: '%s'"%type(d['type']))

        if 'header' == d['type']:
            if 'desc' not in d:
                raise Exception("'header' requires the following scalar fields: {`desc`}")

            if not isinstance(d['desc'], str):
                raise Exception("'desc' must be a string")

            __validate_ordinal(d)
            __validate_version(d)

        elif 'macro' == d['type']:
            if ('desc' not in d) or ('name' not in d) or ('value' not in d):
                raise Exception("'macro' requires the following scalar fields: {`desc`, `name`, `value`}")

            if not isinstance(d['desc'], str):
                raise Exception("'desc' must be a string")

            __validate_name(d, 'name', tags, case='upper')
            __validate_details(d)
            __validate_ordinal(d)
            __validate_version(d)

        elif 'typedef' == d['type']:
            if ('desc' not in d) or ('name' not in d) or ('value' not in d):
                raise Exception("'typedef' requires the following scalar fields: {`desc`, `name`, `value`}")

            __validate_type(d, 'name', tags)
            __validate_details(d)
            __validate_ordinal(d)
            __validate_version(d)

        elif 'handle' == d['type']:
            if ('desc' not in d) or ('name' not in d):
                raise Exception("'handle' requires the following scalar fields: {`desc`, `name`}")

            __validate_handle(d, 'name', tags)
            __validate_details(d)
            __validate_ordinal(d)
            __validate_version(d)

        elif 'enum' == d['type']:
            if ('desc' not in d) or ('name' not in d):
                raise Exception("'enum' requires the following scalar fields: {`desc`, `name`}")

            __validate_type(d, 'name', tags)
            __validate_etors(d, tags)
            __validate_details(d)
            __validate_ordinal(d)
            __validate_version(d)

        elif 'struct' == d['type'] or 'union' == d['type']:
            if ('desc' not in d) or ('name' not in d):
                raise Exception("'%s' requires the following scalar fields: {`desc`, `name`}"%d['type'])

            if d['type'] == 'union':
                __validate_union_tag(d)
            __validate_type(d, 'name', tags)
            __validate_base(d)
            __validate_members(d, tags, meta)
            __validate_details(d)
            __validate_ordinal(d)
            __validate_version(d)

        elif 'function' == d['type']:
            if ('desc' not in d) or ('name' not in d):
                raise Exception("'function' requires the following scalar fields: {`desc`, `name`}")

            if 'class' in d:
                __validate_name({'name': d['class']+d['name']}, 'name', tags, case='camel')
            else:
                __validate_name(d, 'name', tags, case='camel')

            __validate_params(d, tags, meta)
            __validate_details(d)
            __validate_ordinal(d)
            __validate_version(d)

        elif 'class' == d['type']:
            if ('desc' not in d) or ('name' not in d):
                raise Exception("'class' requires the following scalar fields: {`desc`, `name`}")

            __validate_name(d, 'name', tags, case='camel')
            __validate_details(d)
            __validate_ordinal(d)
            __validate_version(d)

        return True

    except Exception as msg:
        print("Specification Validation Error:")
        print("%s(%s): %s!"%(os.path.abspath(f), line_num, msg))
        print("-- Function Info --")
        print(d)
        raise

"""
    filters object by version
"""
def _filter_version(d, max_ver):
    ver = float(d.get('version', default_version))
    if ver > max_ver:
        return None

    def __filter_desc(d):
        if 'desc' in d and isinstance(d['desc'], dict):
            for k, v in d['desc'].items():
                if float(k) <= max_ver:
                    desc = v
            d['desc'] = desc
        return d

    flt = []
    type = d['type']
    if 'enum' == type:
        for e in d['etors']:
            ver = float(e.get('version', default_version))
            if ver <= max_ver:
                flt.append(__filter_desc(e))
        if d['name'].endswith('version_t'):
            flt.append({
                'name': d['name'][:-1].upper() + "CURRENT",
                'value': flt[-1]['value'],
                'desc': "latest known version"
                })
        d['etors'] = flt

    elif 'function' == type:
        for p in d['params']:
            ver = float(p.get('version', default_version))
            if ver <= max_ver:
                flt.append(__filter_desc(p))
        d['params'] = flt

    elif 'struct' == type or 'union' == type or 'class' == type:
        for m in d.get('members',[]):
            ver = float(m.get('version', default_version))
            if ver <= max_ver:
                flt.append(__filter_desc(m))
        d['members'] = flt

    return __filter_desc(d)

"""
    creates docs per version
"""
def _make_versions(d, max_ver):
    docs = []
    type = d['type']
    if 'function' == type or 'struct' == type:
        for ver in all_versions:
            if float(ver) > max_ver:
                break

            dv = _filter_version(copy.deepcopy(d), float(ver))
            if not dv:
                continue

            if len(docs) > 0:
                if dv == docs[-1]:
                    continue

                dv['name'] += ver
            docs.append(dv)
    else:
        docs.append(d)
    return docs

"""
    generates meta-data on all objects
"""
def _generate_meta(d, ordinal, meta):
    type = d['type']
    name = re.sub(r"(\w+)\(.*\)", r"\1", d['name']) # removes '()' part of macros

    # create dict if typename is not already known...
    if type not in meta:
        meta[type] = {}

    if 'class' not in meta:
        meta['class'] = {}

    cls = d.get('class')
    if cls:
        # create dict if class name is not already known...
        if cls not in meta['class']:
            meta['class'][cls] = {}
            meta['class'][cls]['ordinal'] = 0

        # create list if object-type is not already known for class...
        if type not in meta['class'][cls]:
            meta['class'][cls][type] = []

        # append if object-name is not already known for object-type in class...
        if name not in meta['class'][cls][type]:
            meta['class'][cls][type].append(name)

        else:
            print("Error - duplicate entries for %s found!"%name)
            raise

    if 'class' != type:
        # create list if name is not already known for type...
        if 'function' == type and cls:
            name = cls+name

        if name not in meta[type]:
            meta[type][name] = {'class': ""}

        # add values to list
        if 'enum' == type:
            value = -1
            max_value = -1
            bit_mask = 0
            meta[type][name]['etors'] = []
            for idx, etor in enumerate(d['etors']):
                meta[type][name]['etors'].append(etor['name'])
                value = _get_etor_value(etor.get('value'), value)
                if not etor.get('value'):
                    etor['value'] = str(value)
                if type_traits.is_flags(name):
                    bit_mask |= value
                if value > max_value:
                    max_value = value
                    max_index = idx
            if type_traits.is_flags(name):
                meta[type][name]['max'] = hex((max_value << 1)-1) if max_value else '0'
                if bit_mask != 0:
                    meta[type][name]['bit_mask'] = hex(ctypes.c_uint32(~bit_mask).value)
            else:
                meta[type][name]['max'] = d['etors'][max_index]['name']

        elif 'macro' == type:
            meta[type][name]['values'] = []
            if 'value' in d:
                meta[type][name]['values'].append(d['value'])

            if 'altvalue' in d:
                meta[type][name]['values'].append(d['altvalue'])

        elif 'function' == type:
            meta[type][name]['params'] = []
            for p in d['params']:
                meta[type][name]['params'].append({
                    'type':p['type']
                    })

        elif 'struct' == type or 'union' == type:
            meta[type][name]['members'] = []
            for m in d['members']:
                meta[type][name]['members'].append({
                    'type': m['type'],
                    'name': m['name'],
                    'desc': m['desc'],
                    'init': m.get('init')
                    })

        if cls:
            meta[type][name]['class'] = cls

    else:
        if name not in meta['class']:
            meta['class'][name] = {}

        meta['class'][name]['ordinal'] = ordinal

        if 'base' in d:
            base = d['base']
            if 'child' not in meta['class'][base]:
                meta['class'][base]['child'] = []

            meta['class'][base]['child'].append(name)

            if 'handle' not in meta['class'][name]:
                meta['class'][name]['handle'] = []

            if name[:2] == base[:2]:
                meta['class'][name]['handle'].extend(meta['class'][base]['handle'])

        if 'members' in d:
            meta['class'][name]['members'] = []

            for m in d['members']:
                meta['class'][name]['members'].append(m['type'])

        if 'owner' in d:
            owner = d['owner']
            meta['class'][name]['owner'] = owner

            if owner not in meta['class']:
                meta['class'][owner] = {}

            if 'owns' not in meta['class'][owner]:
                meta['class'][owner]['owns'] = []

            meta['class'][owner]['owns'].append(name)

    return meta

"""
    generates SHA512 string for the given object
"""
def _generate_hash(obj):
    # functions-only (for now)...
    if re.match(r"function", obj['type']):
        hash = hashlib.sha256()
        # hashcode of function signature...
        hash.update(obj['name'].encode())
        for p in obj['params']:
            hash.update(p['type'].encode())
        # hashcode of class
        if 'class' in obj:
            hash.update(obj['class'].encode())
        # digest into string
        obj['hash'] = hash.hexdigest()
    return obj

"""
    generates structure members from base
"""
def _inline_base(obj, meta):
    if re.match(r"struct|union", obj['type']):
        base = obj.get('base')
        if base in meta['struct']:
            for i, m in enumerate(meta['struct'][base]['members']):
                m = copy.deepcopy(m)
                if m['name'] == "stype":
                    m['init'] = re.sub(r"(\$[a-z]+)(\w+)_t", r"\1_STRUCTURE_TYPE\2", obj['name']).upper()
                    m['desc'] += ', must be %s' % m['init']
                obj['members'].insert(i, m)
    return obj

"""
    generate complete return permutations
"""
def _generate_returns(obj, meta):
    if re.match(r"function", obj['type']):
        # default results for all functions
        rets = [
            {"$X_RESULT_SUCCESS":[]},
            {"$X_RESULT_ERROR_UNINITIALIZED":[]},
            {"$X_RESULT_ERROR_DEVICE_LOST":[]},
            {"$X_RESULT_ERROR_ADAPTER_SPECIFIC": []}
            ]

        # special function for appending to our list of dicts; avoiding duplicates
        def _append(lst, key, val):
            idx = next((i for i, v in enumerate(lst) if v.get(key)), len(lst))
            if idx == len(lst):
                rets.append({key:[]})
            if val and val not in rets[idx][key]:
                rets[idx][key].append(val)

        def append_nullchecks(param, accessor: str):
            if type_traits.is_pointer(param['type']):
                _append(rets, "$X_RESULT_ERROR_INVALID_NULL_POINTER", "`NULL == %s`" % accessor)

            elif type_traits.is_funcptr(param['type'], meta):
                _append(rets, "$X_RESULT_ERROR_INVALID_NULL_POINTER", "`NULL == %s`" % accessor)

            elif type_traits.is_handle(param['type']) and not type_traits.is_ipc_handle(item['type']):
                _append(rets, "$X_RESULT_ERROR_INVALID_NULL_HANDLE", "`NULL == %s`" % accessor)

        def append_enum_checks(param, accessor: str):
            ptypename = type_traits.base(param['type'])

            prefix = "`"
            if param_traits.is_optional(item):
                prefix = "`NULL != %s && " % item['name']

            if re.match(r"stype", param['name']):
                _append(rets, "$X_RESULT_ERROR_UNSUPPORTED_VERSION", prefix + "%s != %s`"%(re.sub(r"(\$\w)_(.*)_t.*", r"\1_STRUCTURE_TYPE_\2", typename).upper(), accessor))
            else:
                if type_traits.is_flags(param['type']) and 'bit_mask' in meta['enum'][ptypename].keys():
                    _append(rets, "$X_RESULT_ERROR_INVALID_ENUMERATION", prefix + "%s & %s`"%(ptypename.upper()[:-2]+ "_MASK", accessor))
                else:
                    _append(rets, "$X_RESULT_ERROR_INVALID_ENUMERATION", prefix + "%s < %s`"%(meta['enum'][ptypename]['max'], accessor))

        # generate results based on parameters
        for item in obj['params']:
            if param_traits.is_nocheck(item):
                continue

            if not param_traits.is_optional(item):
                append_nullchecks(item, item['name'])

            if type_traits.is_enum(item['type'], meta) and not type_traits.is_pointer(item['type']):
                append_enum_checks(item, item['name'])

            if type_traits.is_descriptor(item['type']) or type_traits.is_properties(item['type']):
                typename = type_traits.base(item['type'])
                # walk each entry in the desc for pointers and enums
                for i, m in enumerate(meta['struct'][typename]['members']):
                    if param_traits.is_nocheck(m):
                        continue

                    if not param_traits.is_optional(m):
                        append_nullchecks(m, "%s->%s" % (item['name'], m['name']))

                    if type_traits.is_enum(m['type'], meta) and not type_traits.is_pointer(m['type']):
                        append_enum_checks(m, "%s->%s" % (item['name'], m['name']))

        # finally, append all user entries
        for item in obj.get('returns', []):
            if isinstance(item, dict):
                for key, values in item.items():
                    for val in values:
                        _append(rets, key, val)
            else:
                _append(rets, item, None)

        # update doc
        obj['returns'] = rets
    return obj


def _inline_extended_structs(specs, meta):
    for s in specs:
        for i, obj in enumerate(s['objects']):
            obj = _inline_base(obj, meta)
            s['objects'][i] = obj

"""
    generates extra content
"""
def _generate_extra(specs, meta):
    for s in specs:
        for i, obj in enumerate(s['objects']):
            obj = _generate_hash(obj)
            obj = _generate_returns(obj, meta)
            s['objects'][i] = obj

"""
    generates reference-data on all objects
"""
def _generate_ref(specs, tags, ref):
    for s in specs:
        for obj in s['objects']:
            # create dict if typename is not already known...
            type = obj['type']
            if type not in ref:
                ref[type] = {}

            name = _subt(obj['name'], tags)

            # convert dict to json-string
            dstr = json.dumps(obj)

            # replace all tags with values
            for key, value in tags.items():
                dstr = re.sub(r"-%s"%re.escape(key), "-"+value, dstr)
                dstr = re.sub(re.escape(key), value, dstr)
                dstr = re.sub(re.escape(key.upper()), value.upper(), dstr)

            # convert json-string back to dict
            obj = json.loads(dstr)

            # update ref
            ref[type][name] = obj

    return ref

def _refresh_enum_meta(obj, meta):
    ## remove the existing meta records
    if obj.get('class'):
        meta['class'][obj['class']]['enum'].remove(obj['name'])

    if meta['enum'].get(obj['name']):
        del meta['enum'][obj['name']]
    ## re-generate meta
    meta = _generate_meta(obj, None, meta)


def _validate_ext_enum_range(extension, enum) -> bool:
    try:
        existing_values = [_get_etor_value(etor.get('value'), None) for etor in enum['etors']]
        for ext in extension['etors']:
            value = _get_etor_value(ext.get('value'), None)
            if value in existing_values:
                return False
            return True
    except:
        return False

def _extend_enums(enum_extensions, specs, meta):
    enum_extensions = sorted(enum_extensions, key= lambda x : x['name'])
    enum_groups = [(k, list(g)) for k, g in itertools.groupby(enum_extensions, key=lambda x : x['name'])]

    for k, group in enum_groups:
        matching_enum = [obj for s in specs for obj in s['objects'] if obj['type'] == 'enum' and k == obj['name'] and not obj.get('extend')][0]
        for i, extension in enumerate(group):
            if not _validate_ext_enum_range(extension, matching_enum):
                raise Exception(f"Invalid enum values.")
            matching_enum['etors'].extend(extension['etors'])

        _refresh_enum_meta(matching_enum, meta)

        ## Sort the etors
        value = -1
        def sort_etors(x):
            nonlocal value
            value = _get_etor_value(x.get('value'), value)
            return value
        matching_enum['etors'] = sorted(matching_enum['etors'], key=sort_etors)

"""
Entry-point:
    Reads each YML file and extracts data
    Returns list of data per file
"""
def parse(section, version, tags, meta, ref):
    path = os.path.join("./", section)
    specs = []

    files = util.findFiles(path, "*.yml")
    files.sort(key = lambda f: 0 if f.endswith('common.yml') else 1)
    registry = [f for f in files if f.endswith('registry.yml')][0]

    enum_extensions = []
    for f in files:

        print("Parsing %s..."%f)
        docs = util.yamlRead(f)
        line_nums = _get_line_nums(f)

        header = None
        name = None
        objects = []

        for i, d in enumerate(docs):
            d = _preprocess(d)
            if not _validate_doc(f, d, tags, line_nums[i], meta):
                continue

            d = _filter_version(d, float(version))
            if not d:
                continue

            if d['type'] == "enum" and d.get("extend") == True:
                # enum extensions are resolved later
                enum_extensions.append(d)
                continue

            # extract header from objects
            if re.match(r"header", d['type']):
                header = d
                header['ordinal'] = int(int(header.get('ordinal',"1000")) * float(header.get('version',"1.0")))
                header['ordinal'] *= 1000 if re.match(r"extension", header.get('desc',"").lower()) else 1
                header['ordinal'] *= 1000 if re.match(r"experimental", header.get('desc',"").lower()) else 1
                basename = os.path.splitext(os.path.basename(f))[0]
                if 'name' in header:
                    name = header['name']
                elif basename.startswith('exp-'):
                    name = f'{basename[len("exp-"):]} (experimental)'
                else:
                    name = basename
                for c in '_-':
                    name = name.replace(c, ' ')
            elif header:
                for d in _make_versions(d, float(version)):
                    objects.append(d)
                    meta = _generate_meta(d, header['ordinal'], meta)

        if header:
            specs.append({
                'name'      : name,
                'header'    : header,
                'objects'   : objects,
            })

    specs = sorted(specs, key=lambda s: s['header']['ordinal'])
    _inline_extended_structs(specs, meta)
    generate_ids.generate_registry(registry, specs, meta, _refresh_enum_meta)
    _extend_enums(enum_extensions, specs, meta)
    _generate_extra(specs, meta)

    ref = _generate_ref(specs, tags, ref)

    print("Parsed %s files and found:"%len(specs))
    for key in meta:
        print(" - %s %s(s)"%(len(meta[key]),key))
    print("")
    return specs, meta, ref
