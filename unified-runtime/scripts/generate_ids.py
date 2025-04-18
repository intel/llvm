# Copyright (C) 2023 Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates a unique id for each spec function that doesn't have it."""

from fileinput import FileInput
import util
import yaml
import re
import copy

ENUM_NAME = "$x_function_t"


class quoted(str):
    pass


def quoted_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


def get_registry_header():
    return {
        "type": "header",
        "desc": quoted("Intel $OneApi Unified Runtime function registry"),
        "ordinal": quoted(-1),
    }


def write_registry(data, path):
    with open(path, "w") as fout:
        yaml.add_representer(quoted, quoted_presenter)
        yaml.dump_all(
            data, fout, default_flow_style=False, sort_keys=False, explicit_start=True
        )


def find_type_in_specs(specs, type):
    return [obj for s in specs for obj in s["objects"] if obj["name"] == type][0]


def get_max_enum(enum):
    return int(max(enum["etors"], key=lambda x: int(x["value"]))["value"])


def copy_and_strip_prefix_from_enums(enum, prefix):
    cpy = copy.deepcopy(enum)
    for etor in cpy["etors"]:
        etor["name"] = etor["name"][len(prefix) :]
    return cpy


def generate_function_type(specs, meta, update_fn) -> dict:
    existing_function_type = find_type_in_specs(specs, "$x_function_t")
    existing_etors = {
        etor["name"]: etor["value"] for etor in existing_function_type["etors"]
    }
    max_etor = get_max_enum(existing_function_type)
    functions = [
        obj["class"][len("$x") :] + obj["name"]
        for s in specs
        for obj in s["objects"]
        if obj["type"] == "function"
    ]
    registry = list()
    for fname in functions:
        etor_name = "$X_FUNCTION_" + util.to_snake_case(fname).upper()
        id = existing_etors.get(etor_name)
        if id is None:
            max_etor += 1
            id = max_etor
        registry.append(
            {"name": etor_name, "desc": f"Enumerator for $x{fname}", "value": str(id)}
        )
    registry = sorted(registry, key=lambda x: int(x["value"]))
    existing_function_type["etors"] = registry
    update_fn(existing_function_type, meta)

    ## create a copy to write back to registry.yml
    return copy_and_strip_prefix_from_enums(existing_function_type, "$X_FUNCTION_")


def generate_structure_type(specs, meta, refresh_fn) -> dict:
    structure_type = find_type_in_specs(specs, "$x_structure_type_t")
    extended_structs = [
        obj
        for s in specs
        for obj in s["objects"]
        if re.match(r"struct|union", obj["type"]) and "base" in obj
    ]
    max_enum = get_max_enum(structure_type)

    structure_type_etors = list()
    for struct in extended_structs:
        # skip experimental enumerations
        if struct["name"].startswith("$x_exp_"):
            continue

        etor = [mem for mem in struct["members"] if mem["name"] == "stype"][0]["init"]

        # try and match the etor
        matched_etor = [e for e in structure_type["etors"] if e["name"] == etor]

        out_etor = {"name": etor, "desc": struct["name"]}

        # if no match exists we assign it a new value
        if len(matched_etor) == 0:
            max_enum += 1
            out_etor["value"] = str(max_enum)
        else:
            out_etor["value"] = matched_etor[0]["value"]

        structure_type_etors.append(out_etor)

    structure_type_etors = sorted(structure_type_etors, key=lambda x: int(x["value"]))
    structure_type["etors"] = structure_type_etors
    refresh_fn(structure_type, meta)

    ## create a copy to write back to registry.yml
    return copy_and_strip_prefix_from_enums(structure_type, "$X_STRUCTURE_TYPE_")


def generate_registry(path, specs, meta, update_fn):
    try:
        write_registry(
            [
                get_registry_header(),
                generate_function_type(specs, meta, update_fn),
                generate_structure_type(specs, meta, update_fn),
            ],
            path,
        )

    except BaseException as e:
        print("Failed to generate registry.yml... %s", e)
        raise e
