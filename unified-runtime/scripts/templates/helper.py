"""
Copyright (C) 2022-2024 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import re
from typing import Any, Dict, List, Tuple, Union
import util

from .print_helper import get_api_types_funcs
from version import Version


class obj_traits:
    """Extracts traits from a spec object"""

    @staticmethod
    def is_function(obj) -> bool:
        try:
            return obj["type"].startswith("function")
        except KeyError:
            return False

    @staticmethod
    def is_class(obj) -> bool:
        try:
            return obj["type"].startswith("class")
        except KeyError:
            return False

    @staticmethod
    def is_handle(obj) -> bool:
        try:
            return obj["type"].startswith("handle")
        except KeyError:
            return False

    @staticmethod
    def is_enum(obj) -> bool:
        try:
            return obj["type"].startswith("enum")
        except KeyError:
            return False

    @staticmethod
    def is_experimental(obj) -> bool:
        try:
            return obj["name"].endswith("Exp")
        except KeyError:
            return False

    @staticmethod
    def class_name(obj) -> Union[str, None]:
        try:
            return obj["class"]
        except KeyError:
            return None

    @staticmethod
    def is_loader_only(obj) -> bool:
        try:
            return obj["loader_only"]
        except KeyError:
            return False


class class_traits:
    """Extracts traits from a class name"""

    @staticmethod
    def is_global(name, tags) -> bool:
        try:
            return name in tags
        except BaseException:
            return False

    @staticmethod
    def is_namespace(name, namespace: str, tags) -> bool:
        try:
            return tags[name] == namespace
        except BaseException:
            return False

    @staticmethod
    def is_singleton(item) -> bool:
        try:
            return "singleton" == item["attribute"]
        except BaseException:
            return False

    @staticmethod
    def get_handle(item, meta) -> str:
        try:
            return meta["class"][item["name"]]["handle"][0]
        except BaseException:
            return ""


class type_traits:
    """Extracts traits from a type name"""

    RE_HANDLE = r"(.*)handle_t"
    RE_NATIVE_HANDLE = r"(.*)native_handle_t"
    RE_IPC = r"(.*)ipc(.*)handle_t"
    RE_POINTER = r"(.*\w+)\*+"
    RE_PPOINTER = r"(.*\w+)\*{2,}"
    RE_DESC = r"(.*)desc_t.*"
    RE_PROPS = r"(.*)properties_t.*"
    RE_FLAGS = r"(.*)flags_t"
    RE_ARRAY = r"(.*)\[([1-9][0-9]*)\]"

    @staticmethod
    def base(name) -> Union[str, bool]:
        try:
            return _remove_const_ptr(name)
        except BaseException:
            return False

    @classmethod
    def is_handle(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_HANDLE, name))
        except TypeError:
            return False

    @classmethod
    def is_native_handle(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_NATIVE_HANDLE, name))
        except TypeError:
            return False

    @classmethod
    def is_pointer_to_pointer(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_PPOINTER, name))
        except TypeError:
            return False

    @classmethod
    def is_ipc_handle(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_IPC, name))
        except TypeError:
            return False

    @staticmethod
    def is_class_handle(name, meta) -> bool:
        try:
            name = _remove_const_ptr(name)
            return len(meta["handle"][name]["class"]) > 0
        except BaseException:
            return False

    @classmethod
    def is_pointer(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_POINTER, name))
        except TypeError:
            return False

    @classmethod
    def is_descriptor(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_DESC, name))
        except TypeError:
            return False

    @classmethod
    def is_properties(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_PROPS, name))
        except TypeError:
            return False

    @classmethod
    def is_flags(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_FLAGS, name))
        except TypeError:
            return False

    @classmethod
    def get_flag_type(cls, name: str) -> str:
        return re.sub(r"(\w+)_flags_t", r"\1_flag_t", name)

    @staticmethod
    def is_known(name, meta) -> bool:
        try:
            name = _remove_const_ptr(name)
            for group in meta:
                if name in meta[group]:
                    return True
            return False
        except BaseException:
            return False

    @staticmethod
    def is_enum(name, meta) -> bool:
        try:
            name = _remove_const_ptr(name)
            if name in meta["enum"]:
                return True
            return False
        except BaseException:
            return False

    @staticmethod
    def is_funcptr(name, meta) -> bool:
        return name in meta["fptr_typedef"]

    @staticmethod
    def is_struct(name, meta) -> bool:
        try:
            name = _remove_const_ptr(name)
            if name in meta["struct"]:
                return True
            return False
        except BaseException:
            return False

    @staticmethod
    def find_class_name(name, meta) -> Union[str, None]:
        try:
            name = _remove_const_ptr(name)
            for group in meta:
                if name in meta[group]:
                    return meta[group][name]["class"]
            return None
        except BaseException:
            return None

    @classmethod
    def is_array(cls, name) -> bool:
        try:
            return True if re.match(cls.RE_ARRAY, name) else False
        except BaseException:
            return False

    @classmethod
    def get_array_length(cls, name) -> str:
        if not cls.is_array(name):
            raise Exception("Cannot find array length of non-array type.")
        match = re.match(cls.RE_ARRAY, name)
        assert match
        return match.groups()[1]

    @classmethod
    def get_array_element_type(cls, name) -> str:
        if not cls.is_array(name):
            raise Exception("Cannot find array type of non-array type.")
        match = re.match(cls.RE_ARRAY, name)
        assert match
        return match.groups()[0]

    @staticmethod
    def get_struct_members(type_name, meta) -> List[dict]:
        struct_type = _remove_const_ptr(type_name)
        if struct_type not in meta["struct"]:
            raise Exception(f"Cannot return members of non-struct type {struct_type}.")
        return meta["struct"][struct_type]["members"]


class value_traits:
    """Extracts traits from a value name"""

    RE_VERSION = r"\$X_MAKE_VERSION\(\s*(\d+)\s*\,\s*(\d+)\s*\)"
    RE_BIT = r".*BIT\(\s*(.*)\s*\)"
    RE_HEX = r"0x\w+"
    RE_MACRO = r"(\$\w+)\(.*\)"
    RE_ARRAY = r"(.*)\[(.*)\]"

    @classmethod
    def is_ver(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_VERSION, name))
        except BaseException:
            return False

    @classmethod
    def get_major_ver(cls, name) -> int:
        try:
            return int(re.sub(cls.RE_VERSION, r"\1", name))
        except BaseException:
            return 0

    @classmethod
    def get_minor_ver(cls, name) -> int:
        try:
            return int(re.sub(cls.RE_VERSION, r"\2", name))
        except BaseException:
            return 0

    @classmethod
    def is_bit(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_BIT, name))
        except BaseException:
            return False

    @classmethod
    def get_bit_count(cls, name) -> int:
        try:
            return int(re.sub(cls.RE_BIT, r"\1", name))
        except BaseException:
            return 0

    @classmethod
    def is_hex(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_HEX, name))
        except BaseException:
            return False

    @classmethod
    def is_macro(cls, name, meta) -> bool:
        try:
            name = cls.get_macro_name(name)
            name = cls.get_array_length(name)
            return name in meta["macro"]
        except BaseException:
            return False

    @classmethod
    def get_macro_name(cls, name) -> str:
        try:
            return re.sub(cls.RE_MACRO, r"\1", name)  # 'NAME()' -> 'NAME'
        except BaseException:
            return name

    @classmethod
    def is_array(cls, name) -> bool:
        try:
            return bool(re.match(cls.RE_ARRAY, name))
        except BaseException:
            return False

    @classmethod
    def get_array_name(cls, name) -> str:
        try:
            return re.sub(cls.RE_ARRAY, r"\1", name)  # 'name[len]' -> 'name'
        except BaseException:
            return name

    @classmethod
    def get_array_length(cls, name) -> str:
        try:
            return re.sub(cls.RE_ARRAY, r"\2", name)  # 'name[len]' -> 'len'
        except BaseException:
            return name

    @classmethod
    def find_enum_name(cls, name, meta) -> Union[list, None]:
        try:
            name = cls.get_array_name(name)
            # if the value is an etor, return the name of the enum
            for e in meta["enum"]:
                if name in meta["enum"][e]["etors"]:
                    return e
            return None
        except BaseException:
            return None


class param_traits:
    """Extracts traits from a parameter object"""

    RE_MBZ = r".*\[mbz\].*"
    RE_IN = r"^\[in\].*"
    RE_OUT = r"^\[out\].*"
    RE_INOUT = r"^\[in,out\].*"
    RE_OPTIONAL = r".*\[optional\].*"
    RE_NOCHECK = r".*\[nocheck\].*"
    RE_RANGE = r".*\[range\((.+),\s*(.+)\)\][\S\s]*"
    RE_RETAIN = r".*\[retain\].*"
    RE_RELEASE = r".*\[release\].*"
    RE_TYPENAME = r".*\[typename\((.+),\s(.+)\)\].*"
    RE_TAGGED = r".*\[tagged_by\((.+)\)].*"
    RE_BOUNDS = r".*\[bounds\((.+),\s*(.+)\)].*"

    @classmethod
    def is_mbz(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_MBZ, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_input(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_IN, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_output(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_OUT, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_inoutput(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_INOUT, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_optional(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_OPTIONAL, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_nocheck(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_NOCHECK, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_range(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_RANGE, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_tagged(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_TAGGED, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_bounds(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_BOUNDS, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def tagged_member(cls, item) -> Union[str, None]:
        try:
            return re.sub(cls.RE_TAGGED, r"\1", item["desc"])
        except BaseException:
            return None

    @classmethod
    def range_start(cls, item) -> Union[str, None]:
        try:
            return re.sub(cls.RE_RANGE, r"\1", item["desc"])
        except BaseException:
            return None

    @classmethod
    def range_end(cls, item) -> Union[str, None]:
        try:
            return re.sub(cls.RE_RANGE, r"\2", item["desc"])
        except BaseException:
            return None

    @classmethod
    def is_retain(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_RETAIN, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_release(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_RELEASE, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_typename(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_TYPENAME, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def typename(cls, item) -> Union[str, None]:
        match = re.match(cls.RE_TYPENAME, item["desc"])
        if match:
            return match.group(1)
        else:
            return None

    @classmethod
    def typename_size(cls, item) -> Union[str, None]:
        match = re.match(cls.RE_TYPENAME, item["desc"])
        if match:
            return match.group(2)
        else:
            return None

    @classmethod
    def bounds_offset(cls, item) -> Union[str, None]:
        match = re.match(cls.RE_BOUNDS, item["desc"])
        if match:
            return match.group(1)
        else:
            return None

    @classmethod
    def bounds_size(cls, item) -> Union[str, None]:
        match = re.match(cls.RE_BOUNDS, item["desc"])
        if match:
            return match.group(2)
        else:
            return None


class function_traits:
    """Extracts traits from a function object"""

    @staticmethod
    def is_static(item) -> bool:
        try:
            return bool(re.match(r"static", item["decl"]))
        except BaseException:
            return False

    @staticmethod
    def is_global(item, tags) -> bool:
        try:
            return item["class"] in tags
        except BaseException:
            return False


class etor_traits:
    """Extracts traits from an enumerator"""

    RE_OPTIONAL_QUERY = r".*\[optional-query\].*"
    RE_DEPRECATED = r".*\[deprecated-value\].*"

    @classmethod
    def is_optional_query(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_OPTIONAL_QUERY, item["desc"]))
        except BaseException:
            return False

    @classmethod
    def is_deprecated_etor(cls, item) -> bool:
        try:
            return bool(re.match(cls.RE_DEPRECATED, item["desc"]))
        except BaseException:
            return False


def subt(
    namespace, tags: dict, string: str, comment=False, remove_namespace=False
) -> str:
    """
    Public:
        substitutes each tag['key'] with tag['value']
        if comment, then insert doxygen '::' notation at beginning (for autogen links)
    """
    for key, value in tags.items():
        if remove_namespace:
            repl = ""  # remove namespace; e.g. "$x" -> ""
            string = re.sub(r"%s_?" % re.escape(key), repl, string)
            string = re.sub(r"%s_?" % re.escape(key.upper()), repl.upper(), string)
        else:
            string = re.sub(
                r"-%s" % re.escape(key), "-" + value, string
            )  # hack for compile options
            repl = (
                "::" + value if comment and "$OneApi" != key else value
            )  # replace tag; e.g., "$x" -> "ur"
            string = re.sub(re.escape(key), repl, string)
            string = re.sub(re.escape(key.upper()), repl.upper(), string)
    return string


def append_ws(string: str, count: int) -> str:
    """
    Public:
        appends whitespace (in multiples of 4) to the end of the string,
        until len(string) > count
    """
    while len(string) > count:
        count = count + 4
    string = "{str: <{width}}".format(str=string, width=count)
    return string


def split_line(line: str, ch_count: int) -> List[str]:
    """
    Public:
        split the line of text into a list of strings,
        where each length of each entry is less-than count
    """
    if not line:
        return [""]

    RE_NEWLINE = r"(.*)\n(.*)"

    words = line.split(" ")
    lines = []
    word_list = []

    for word in words:
        if re.match(RE_NEWLINE, word):
            prologue = re.sub(RE_NEWLINE, r"\1", word)
            epilogue = re.sub(RE_NEWLINE, r"\2", word)
            word_list.append(prologue)
            lines.append(" ".join(word_list))
            word_list = []
            if len(epilogue):
                word_list.append(epilogue)
        elif sum(map(len, word_list)) + len(word_list) + len(word) <= ch_count:
            word_list.append(word)

        else:
            lines.append(" ".join(word_list))
            word_list = [word]

    if len(word_list):
        lines.append(" ".join(word_list))
    return lines


def _camel_to_snake(name: str) -> str:
    """
    Private:
        converts string from camelCase to snake_case
    """
    return util.to_snake_case(name).lower()


def filter_items(lst: List[dict], key: str, filter: Any) -> List[dict]:
    """
    Public:
        removes items from the list with the key and whose value do not match filter
    """
    flst = []
    for item in lst:
        if key in item:
            if filter == item[key]:
                flst.append(item)
    return flst


def extract_items(lst: List[dict], key: str) -> List[dict]:
    """
    Public:
        returns a list of items with key from a list of dict
    """
    klst = []
    for item in lst:
        if key in item:
            klst.append(item[key])
    return klst


def extract_objs(specs: List[dict], value: str) -> List[dict]:
    """
    Public:
        returns a list of all objects of type in all specs
    """
    objs = []
    for s in specs:
        for obj in s["objects"]:
            if re.match(value, obj["type"]):
                objs.append(obj)
    return objs


def get_adapter_functions(specs: List[dict]) -> List[dict]:
    """
    Public:
        returns a list of all adapter functions
    """
    objs = []
    for s in specs:
        for obj in s["objects"]:
            if obj_traits.is_function(obj) and not obj_traits.is_loader_only(obj):
                objs.append(obj)
    return objs


def get_adapter_handles(specs: List[dict]) -> List[dict]:
    """
    Public:
        returns a list of all adapter handles
    """
    objs = []
    for s in specs:
        for obj in s["objects"]:
            if obj_traits.is_handle(obj) and not (
                obj_traits.is_loader_only(obj) or "native" in obj["name"]
            ):
                objs.append(obj)
    return objs


def get_adapter_manifests(specs: List[dict]) -> List[dict]:
    """
    Public:
        returns a list of all adapter manifests
    """
    objs = []
    for s in specs:
        for obj in s["objects"]:
            if "manifest" in obj["type"]:
                objs.append(obj)
    return objs


def get_loader_functions(
    specs: List[dict], meta: dict, namespace: str, tags
) -> List[str]:
    """
    Public:
        returns a list of all loader API functions' names
    """
    func_names = []

    # Main API functions
    for s in specs:
        for obj in s["objects"]:
            if obj_traits.is_function(obj):
                func_names.append(make_func_name(namespace, tags, obj))
    # Process address tables functions
    for tbl in get_pfntables(specs, meta, namespace, tags):
        func_names.append(tbl["export"]["name"])

    # Print functions
    api_types_funcs = get_api_types_funcs(specs, meta, namespace, tags)
    for func in api_types_funcs:
        func_names.append(func.c_name)
    func_names.append(f"{tags['$x']}PrintFunctionParams")

    return sorted(func_names)


def _remove_const(name: str) -> str:
    """
    Private:
        removes 'const' from c++ type
    """
    name = name.split(" ")[-1]
    return name


def _remove_ptr(name: str, last: bool = True) -> str:
    """
    Private:
        removes '*' from c++ type
    """
    if last:
        name = re.sub(r"(.*)\*", r"\1", name)  # removes only last '*'
    else:
        name = re.sub(r"\*", "", name)  # removes all '*'
    return name


def _remove_const_ptr(name: str) -> str:
    """
    Private:
        removes 'const' and '*' from c++ type
    """
    name = _remove_ptr(_remove_const(name))
    return name


def make_macro_name(namespace: str, tags: dict, obj: dict, params: bool = True) -> str:
    """
    Public:
        returns c/c++ name of macro
    """
    if params:
        return subt(namespace, tags, obj["name"])
    else:
        name = re.sub(r"(.*)\(.*", r"\1", obj["name"])  # remove '()' part
        return subt(namespace, tags, name)


def make_type_name(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns c/c++ name of enums, structs, unions, typedefs...
    """
    return subt(namespace, tags, obj["name"])


def make_enum_name(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns c/c++ name of enums...
    """
    name = make_type_name(namespace, tags, obj)
    if type_traits.is_flags(obj["name"]):
        name = re.sub(r"flags", r"flag", name)
    return name


def make_flags_bitmask(namespace: str, tags: dict, obj: dict, meta) -> str:
    """Returns c/c++ definition of flags_t bit mask."""
    etor_meta = meta[obj["type"]][obj["name"]]
    if "bit_mask" not in etor_meta.keys():
        return ""
    macro_def = "#define"
    macro_name = make_type_name(namespace, tags, obj).upper()[:-2] + "_MASK"
    mask = etor_meta["bit_mask"]
    return "%s %s %s" % (macro_def, macro_name, mask)


def make_etor_name(namespace: str, tags: dict, enum, etor, meta=None) -> str:
    """
    Public:
        returns c/c++ name of etor
    """
    return subt(namespace, tags, etor)


def etor_get_associated_type(namespace: str, tags: dict, item) -> Union[str, None]:
    """
    Private:
        returns the associated type of an etor from a typed enum
    """
    match = re.match(r"^\[([$A-Za-z0-9_*[\] ]+)\]", item["desc"])
    if match:
        associated_type = match.group(1)
        return subt(namespace, tags, associated_type)
    else:
        return None


def _get_value_name(namespace: str, tags: dict, value) -> str:
    """
    Private:
        returns c/c++ name of value
    """
    return subt(namespace, tags, value)


def make_etor_lines(namespace: str, tags: dict, obj: dict, meta=None) -> List[str]:
    """
    Public:
        returns a list of strings for declaring each enumerator in an enumeration
        c++ format: "ETOR_NAME = VALUE, ///< DESCRIPTION"
    """
    lines = []
    for item in obj["etors"]:
        name = make_etor_name(namespace, tags, obj["name"], item["name"], meta)
        deprecated = " [[deprecated]]" if etor_traits.is_deprecated_etor(item) else ""

        if "value" in item:
            delim = ","
            value = _get_value_name(namespace, tags, item["value"])
            prologue = "%s%s = %s%s" % (name, deprecated, value, delim)
        else:
            prologue = "%s%s," % (name, deprecated)

        for line in split_line(subt(namespace, tags, item["desc"], True), 70):
            lines.append(" /// %s" % line)
        lines.append(prologue)

    lines += [
        "/// @cond",
        "%sFORCE_UINT32 = 0x7fffffff"
        % make_enum_name(namespace, tags, obj)[:-1].upper(),
        "/// @endcond",
    ]

    return lines


def _get_type_name(namespace: str, tags: dict, obj: dict, item) -> str:
    """
    Private:
        returns c/c++ name of any type
    """
    type = item["type"]
    if type_traits.is_array(type):
        type = type_traits.get_array_element_type(type)
    return subt(namespace, tags, type)


def make_member_name(
    namespace: str, tags: dict, item: dict, prefix="", remove_array=False
) -> str:
    """
    Public:
        returns c/c++ name of member of struct/class
    """
    name = subt(namespace, tags, prefix + item["name"])
    if remove_array:
        name = value_traits.get_array_name(name)
    return name


def make_member_lines(
    namespace: str, tags: dict, obj: dict, prefix="", meta=None
) -> List[str]:
    """
    Public:
        returns a list of strings for each member of a structure or class
        c++ format: "TYPE NAME = INIT, ///< DESCRIPTION"
    """
    lines = []
    if "members" not in obj:
        return lines

    for item in obj["members"]:
        name = make_member_name(namespace, tags, item, prefix)
        tname = _get_type_name(namespace, tags, obj, item)

        array_suffix = (
            f"[{type_traits.get_array_length(item['type'])}]"
            if type_traits.is_array(item["type"])
            else ""
        )
        prologue = "%s %s %s;" % (tname, name, array_suffix)

        for line in split_line(subt(namespace, tags, item["desc"], True), 70):
            lines.append(" /// %s" % line)
        lines.append(prologue)
    return lines


def _get_param_name(namespace: str, tags: dict, item: dict) -> str:
    """
    Private:
        returns c/c++ name of parameter
    """
    return subt(namespace, tags, item["name"])


def make_param_lines(
    namespace: str,
    tags: dict,
    obj: dict,
    decl: bool = False,
    meta: Union[dict, None] = None,
    format: List[str] = ["type", "name", "delim", "desc"],
    delim: str = ",",
    replacements: dict = {},
):
    """
    Public:
        returns a list of c++ strings for each parameter of a function
        format: "TYPE NAME = INIT, ///< DESCRIPTION"
    """
    lines = []

    params = obj["params"]

    for i, item in enumerate(params):
        name = _get_param_name(namespace, tags, item)
        if replacements.get(name):
            name = replacements[name]

        tname = _get_type_name(namespace, tags, obj, item)

        words = []
        if "type*" in format:
            words.append(tname + "*")
            name = "p" + name
        elif "type" in format:
            words.append(tname)
        if "name" in format:
            words.append(name)

        prologue = " ".join(words)
        if "delim" in format:
            if i < len(params) - 1:
                prologue += delim

        if "desc" in format:
            desc = item["desc"]
            for line in split_line(subt(namespace, tags, desc, True), 70):
                lines.append(" /// %s" % line)
            lines.append(prologue)
        else:
            lines.append(prologue)

    if "type" in format and len(lines) == 0:
        lines = ["void"]
    return lines


def find_param_name(name_re, namespace: str, tags: dict, obj) -> Union[str, None]:
    """
    Public:
        searches params of function `obj` for a match to the given regex and
        returns its full C++ name
    """
    for param in obj["params"]:
        param_cpp_name = _get_param_name(namespace, tags, param)
        print("searching {0} for pattner {1}".format(param_cpp_name, name_re))
        if re.search(name_re, param_cpp_name):
            return param_cpp_name
    return None


def make_desc_lines(namespace: str, tags: dict, obj) -> List[str]:
    """
    Public:
        returns a list of strings for the description
        format: "@brief DESCRIPTION"
    """
    lines = []
    prologue = "@brief"
    for line in split_line(subt(namespace, tags, obj["desc"], True), 70):
        lines.append("%s %s" % (prologue, line))
        prologue = "      "
    return lines


def make_details_lines(namespace: str, tags: dict, obj) -> List[str]:
    """
    Public:
        returns a list of strings for the detailed description
        format: "@details DESCRIPTION"
    """
    lines = []
    if "details" in obj:
        lines.append("")
        lines.append("@details")

        # If obj['details'] is a list of bullet points, add the bullet point formatting to the lines.
        if isinstance(obj["details"], list):
            for item in obj["details"]:
                if isinstance(item, dict):
                    for key, values in item.items():
                        prologue = "    -"
                        for line in split_line(subt(namespace, tags, key, True), 70):
                            lines.append("%s %s" % (prologue, line))
                            prologue = "     "
                        for val in values:
                            prologue = "        +"
                            for line in split_line(
                                subt(namespace, tags, val, True), 66
                            ):
                                lines.append("%s %s" % (prologue, line))
                                prologue = "         "
                else:
                    prologue = "    -"
                    for line in split_line(subt(namespace, tags, item, True), 70):
                        lines.append("%s %s" % (prologue, line))
                        prologue = "     "
        # If obj['details'] is a string, then assume that it's already correctly formatted using markdown.
        else:
            for line in obj["details"].splitlines(False):
                lines.append(subt(namespace, tags, line, True))

    if "analogue" in obj:
        lines.append("")
        lines.append("@remarks")
        lines.append("  _Analogues_")
        for line in obj["analogue"]:
            lines.append("    - %s" % line)
    return lines


def make_returns_lines(namespace: str, tags: dict, obj: dict, meta=None) -> List[str]:
    """
    Public:
        returns a list of strings for possible return values
    """
    lines = []
    lines.append("@returns")
    for item in obj.get("returns", []):
        for key, values in item.items():
            lines.append("    - %s" % subt(namespace, tags, key, True))
            for val in values:
                lines.append("        + %s" % subt(namespace, tags, val, True))

    return lines


def make_func_name(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns the name of a function
    """
    cname = obj_traits.class_name(obj)
    if not cname:  # If can't find the class name append nothing
        cname = ""
    return subt(namespace, tags, "%s%s" % (cname, obj["name"]))


def make_func_name_with_prefix(prefix, name) -> str:
    """
    Public:
        returns the name of a function from a given name with prefix
    """
    func_name = re.sub(r"^[^_]+_", "", name)
    func_name = re.sub("_t$", "", func_name).capitalize()
    func_name = re.sub(r"_([a-z])", lambda match: match.group(1).upper(), func_name)
    func_name = f"{prefix}{func_name}"
    return func_name


def make_func_etor(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns the etor of a function
    """
    etags = tags.copy()
    etags["$x"] += "Function"
    return util.to_snake_case(make_func_name(namespace, etags, obj)).upper()


def make_pfn_name(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns the name of a function pointer
    """
    return subt(namespace, tags, "pfn%s" % obj["name"])


def make_pfncb_name(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns the name of a function pointer
    """
    return subt(namespace, tags, "pfn%sCb" % obj["name"])


def make_pfn_type(namespace: str, tags: dict, obj: dict, epilogue="") -> str:
    """
    Public:
        returns the name of a function pointer
    """
    newtags = dict()
    for key, value in tags.items():
        if re.match(namespace, value):
            newtags[key] = "pfn"
    return "%s_%s%s_t" % (namespace, make_func_name(namespace, newtags, obj), epilogue)


def make_pfncb_type(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns the name of a function pointer
    """
    return make_pfn_type(namespace, tags, obj, epilogue="Cb")


def make_pfncb_param_type(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns the name of a function pointer
    """
    return "%s_params_t" % _camel_to_snake(make_func_name(namespace, tags, obj))


def get_bounds_check(param, bounds_error) -> str:
    """
    Public:
        returns an appropriate bounds helper function call for an entry point
        parameter with the [bounds] tag
    """
    # Images need their own helper, since function signature wise they would be
    # identical to buffer rect
    bounds_function = "boundsImage" if "image" in param["name"].lower() else "bounds"
    bounds_check = "auto {0} = {1}({2}, {3}, {4})".format(
        bounds_error,
        bounds_function,
        param["name"],
        param_traits.bounds_offset(param),
        param_traits.bounds_size(param),
    )
    bounds_check += "; {0} != UR_RESULT_SUCCESS".format(bounds_error)

    # USM bounds checks need the queue handle parameter to be able to use the
    # GetMemAllocInfo entry point
    if type_traits.is_pointer(param["type"]):
        # If no `hQueue` parameter exists that should have been caught at spec
        # generation.
        return re.sub(r"bounds\(", "bounds(hQueue, ", bounds_check)

    return bounds_check


def make_param_checks(
    namespace: str, tags: dict, obj: dict, cpp=False, meta=None
) -> dict:
    """
    Public:
        returns a dict of auto-generated c++ parameter validation checks for the
        given function (specified by `obj`)
    """
    checks = {}
    for item in obj.get("returns", []):
        for key, values in item.items():
            key = subt(namespace, tags, key, False, cpp)
            for val in values:
                code = re.match(r"^\`([^`]*)\`$", val)
                if code:
                    if key not in checks:
                        checks[key] = []
                    checks[key].append(subt(namespace, tags, code.group(1), False, cpp))

    for p in obj.get("params", []):
        if param_traits.is_bounds(p):
            if "boundsError" not in checks:
                checks["boundsError"] = []
            checks["boundsError"].append(get_bounds_check(p, "boundsError"))

    return checks


def get_class_function_objs(specs: List[dict], cname, version=None) -> List[dict]:
    """
    Public:
        returns a list of all function objs for the specified class.
    """
    objects = []
    for s in specs:
        for obj in s["objects"]:
            is_function = obj_traits.is_function(obj)
            match_cls = cname == obj_traits.class_name(obj)
            if is_function and match_cls:
                if version is None:
                    objects.append(obj)
                elif Version(obj.get("version", "1.0")) <= version:
                    objects.append(obj)
    return sorted(
        objects,
        key=lambda obj: (Version(obj.get("version", "1.0")).major * 10000)
        + int(obj.get("ordinal", "100")),
    )


def get_class_function_objs_exp(
    specs: List[dict], cname
) -> Tuple[List[dict], List[dict]]:
    """
    Public:
        returns a list of all non-experimental function objs and a list of experimental function objs for the specified class
    """
    objects = []
    exp_objects = []
    for s in specs:
        for obj in s["objects"]:
            is_function = obj_traits.is_function(obj)
            match_cls = cname == obj_traits.class_name(obj)
            if is_function and match_cls:
                if obj_traits.is_experimental(obj):
                    exp_objects.append(obj)
                else:
                    objects.append(obj)
    objects = sorted(
        objects,
        key=lambda obj: (Version(obj.get("version", "1.0")).major * 10000)
        + int(obj.get("ordinal", "100")),
    )
    exp_objects = sorted(
        exp_objects,
        key=lambda obj: (Version(obj.get("version", "1.0")).major * 10000)
        + int(obj.get("ordinal", "100")),
    )
    return objects, exp_objects


def get_table_name(namespace: str, tags: dict, obj) -> str:
    """
    Public:
        returns string name of table for function object
    """
    cname = obj_traits.class_name(obj)
    assert cname
    if obj_traits.is_experimental(obj):
        cname = cname + "Exp"
    name = subt(namespace, tags, cname, remove_namespace=True)  # i.e., "$x" -> ""
    name = name if len(name) > 0 else "Global"
    return name


def get_pfntables(specs: List[dict], meta: dict, namespace: str, tags) -> List[dict]:
    """
    Public:
        returns a list of dict of each pfntables needed
    """
    tables = []
    for cname in sorted(meta["class"], key=lambda x: meta["class"][x]["ordinal"]):
        objs, exp_objs = get_class_function_objs_exp(specs, cname)
        objs = list(filter(lambda obj: not obj_traits.is_loader_only(obj), objs))
        exp_objs = list(
            filter(lambda obj: not obj_traits.is_loader_only(obj), exp_objs)
        )

        if len(objs) > 0:
            name = get_table_name(namespace, tags, objs[0])
            table = "%s_%s_dditable_t" % (namespace, _camel_to_snake(name))

            params = []
            params.append(
                {
                    "type": "$x_api_version_t",
                    "name": "version",
                    "desc": "[in] API version requested",
                }
            )
            params.append(
                {
                    "type": "%s*" % table,
                    "name": "pDdiTable",
                    "desc": "[in,out] pointer to table of DDI function pointers",
                }
            )
            export = {
                "name": "%sGet%sProcAddrTable" % (namespace, name),
                "params": params,
            }

            pfn = "%s_pfnGet%sProcAddrTable_t" % (namespace, name)

            tables.append(
                {
                    "name": name,
                    "type": table,
                    "export": export,
                    "pfn": pfn,
                    "functions": objs,
                    "experimental": False,
                }
            )
        if len(exp_objs) > 0:
            name = get_table_name(namespace, tags, exp_objs[0])
            table = "%s_%s_dditable_t" % (namespace, _camel_to_snake(name))

            params = []
            params.append(
                {
                    "type": "$x_api_version_t",
                    "name": "version",
                    "desc": "[in] API version requested",
                }
            )
            params.append(
                {
                    "type": "%s*" % table,
                    "name": "pDdiTable",
                    "desc": "[in,out] pointer to table of DDI function pointers",
                }
            )
            export = {
                "name": "%sGet%sProcAddrTable" % (namespace, name),
                "params": params,
            }

            pfn = "%s_pfnGet%sProcAddrTable_t" % (namespace, name)

            tables.append(
                {
                    "name": name,
                    "type": table,
                    "export": export,
                    "pfn": pfn,
                    "functions": exp_objs,
                    "experimental": True,
                }
            )

    return tables


def get_initial_null_set(obj) -> str:
    """
    Public:
        returns an expression setting required output parameters to null on entry
    """
    cname = obj_traits.class_name(obj)
    assert cname
    lvalue = {
        ("$xProgram", "Link"): "phProgram",
        ("$xProgram", "LinkExp"): "phProgram",
    }.get((cname, obj["name"]))
    if lvalue is not None:
        return "if (nullptr != {0}) {{*{0} = nullptr;}}".format(lvalue)
    return ""


def get_pfncbtables(specs: List[dict], meta: dict, namespace: str, tags) -> List[dict]:
    """
    Public:
        returns a list of dict of each pfntables needed
    """
    tables = []
    for cname in sorted(meta["class"], key=lambda x: meta["class"][x]["ordinal"]):
        objs = get_class_function_objs(specs, cname, Version("1.0"))
        if len(objs) > 0:
            name = get_table_name(namespace, tags, {"class": cname})
            table = "%s_%s_callbacks_t" % (namespace, _camel_to_snake(name))
            tables.append({"name": name, "type": table, "functions": objs})
    return tables


def strip_deref(string_to_strip) -> str:
    """
    Public:
        Strips a string of all dereferences.

        This is useful in layer templates for creating unique variable names. For
        instance if we need to copy pMyStruct->member.hObject out of a function
        parameter into a local variable we use this to get the unique (or at least
        distinct from any other parameter we might copy) variable name
        pMyStructmemberhObject.
    """
    string_to_strip = string_to_strip.replace(".", "")
    return string_to_strip.replace("->", "")


def get_enum_by_name(
    specs, namespace: str, tags, name, only_typed
) -> Union[dict, None]:
    """
    Public:
        returns an enum object with the given name
    """
    for s in specs:
        for obj in s["objects"]:
            if obj_traits.is_enum(obj) and make_enum_name(namespace, tags, obj) == name:
                typed = obj.get("typed_etors", False) is True
                if only_typed:
                    if typed:
                        return obj
                    else:
                        return None
                else:
                    return obj
    return None


def get_loader_epilogue(
    specs: List[dict], namespace: str, tags: dict, obj: dict, meta
) -> List[str]:
    """
    Public:
        returns a list of dict for converting loader output parameters
    """

    epilogue = []

    for item in obj["params"]:
        if param_traits.is_mbz(item):
            continue

        name = subt(namespace, tags, item["name"])
        tname = _remove_const_ptr(subt(namespace, tags, item["type"]))

        obj_name = re.sub(r"(\w+)_handle_t", r"\1_object_t", tname)
        fty_name = re.sub(r"(\w+)_handle_t", r"\1_factory", tname)

        if (
            param_traits.is_retain(item)
            or param_traits.is_release(item)
            or param_traits.is_output(item)
            or param_traits.is_inoutput(item)
        ):
            if type_traits.is_class_handle(item["type"], meta):
                if param_traits.is_range(item):
                    range_start = param_traits.range_start(item)
                    range_end = param_traits.range_end(item)
                    epilogue.append(
                        {
                            "name": name,
                            "type": tname,
                            "obj": obj_name,
                            "factory": fty_name,
                            "retain": param_traits.is_retain(item),
                            "release": param_traits.is_release(item),
                            "range": (range_start, range_end),
                        }
                    )
                else:
                    epilogue.append(
                        {
                            "name": name,
                            "type": tname,
                            "obj": obj_name,
                            "factory": fty_name,
                            "retain": param_traits.is_retain(item),
                            "release": param_traits.is_release(item),
                            "optional": param_traits.is_optional(item),
                        }
                    )
            elif param_traits.is_typename(item):
                typename = param_traits.typename(item)
                underlying_type = None
                for inner in obj["params"]:
                    iname = _get_param_name(namespace, tags, inner)
                    if iname == typename:
                        underlying_type = _get_type_name(namespace, tags, obj, inner)
                if underlying_type is None:
                    continue

                prop_size = param_traits.typename_size(item)
                enum = get_enum_by_name(specs, namespace, tags, underlying_type, True)
                assert enum
                handle_etors = []
                for etor in enum["etors"]:
                    associated_type = etor_get_associated_type(namespace, tags, etor)
                    assert associated_type
                    if "handle" in associated_type:
                        is_array = False
                        if value_traits.is_array(associated_type):
                            associated_type = value_traits.get_array_name(
                                associated_type
                            )
                            is_array = True

                        etor_name = make_etor_name(
                            namespace, tags, enum["name"], etor["name"]
                        )
                        obj_name = re.sub(
                            r"(\w+)_handle_t", r"\1_object_t", associated_type
                        )
                        fty_name = re.sub(
                            r"(\w+)_handle_t", r"\1_factory", associated_type
                        )
                        handle_etors.append(
                            {
                                "name": etor_name,
                                "type": associated_type,
                                "obj": obj_name,
                                "factory": fty_name,
                                "is_array": is_array,
                            }
                        )

                if handle_etors:
                    epilogue.append(
                        {
                            "name": name,
                            "obj": obj_name,
                            "retain": False,
                            "release": False,
                            "typename": typename,
                            "size": prop_size,
                            "etors": handle_etors,
                        }
                    )

    return epilogue


def get_event_wait_list_functions(
    specs: List[dict], namespace: str, tags
) -> List[dict]:
    funcs = []
    for s in specs:
        for obj in s["objects"]:
            if re.match(r"function", obj["type"]):
                if any(x["name"] == "phEventWaitList" for x in obj["params"]) and any(
                    x["name"] == "numEventsInWaitList" for x in obj["params"]
                ):
                    funcs.append(make_func_name(namespace, tags, obj))
    return funcs


def _get_create_get_retain_release_functions(
    specs: List[dict], namespace: str, tags
) -> Dict[str, List[dict]]:
    """
    Private:
        returns a dictionary with lists of create, get, retain and release functions
    """
    funcs = []
    for s in specs:
        for obj in s["objects"]:
            if re.match(r"function", obj["type"]):
                funcs.append(make_func_name(namespace, tags, obj))

    create_suffixes = r"(Create[A-Za-z]*){1}$"
    enqueue_prefixes = r"(Enqueue){1}"
    get_suffixes = r"(Get){1}$"
    retain_suffixes = r"(Retain){1}$"
    release_suffixes = r"(Release){1}$"
    common_prefix = r"^" + namespace

    create_exp = common_prefix + r"[A-Za-z]+" + create_suffixes
    enqueue_exp = common_prefix + enqueue_prefixes + r"[A-Za-z]+$"
    get_exp = common_prefix + r"[A-Za-z]+" + get_suffixes
    retain_exp = common_prefix + r"[A-Za-z]+" + retain_suffixes
    release_exp = common_prefix + r"[A-Za-z]+" + release_suffixes

    create_funcs, get_funcs, retain_funcs, release_funcs = (
        list(
            filter(lambda f: re.match(create_exp, f) or re.match(enqueue_exp, f), funcs)
        ),
        list(filter(lambda f: re.match(get_exp, f), funcs)),
        list(filter(lambda f: re.match(retain_exp, f), funcs)),
        list(filter(lambda f: re.match(release_exp, f), funcs)),
    )

    return {
        "create": create_funcs,
        "get": get_funcs,
        "retain": retain_funcs,
        "release": release_funcs,
    }


def get_handle_create_get_retain_release_functions(
    specs: List[dict], namespace: str, tags
) -> List[dict]:
    """
    Public:
        returns a list of dictionaries containing handle types and the corresponding create, get, retain and release functions
    """
    # Handles without release function
    excluded_handles = ["$x_platform_handle_t", "$x_native_handle_t"]
    # Handles from experimental features
    exp_prefix = "$x_exp"

    funcs = _get_create_get_retain_release_functions(specs, namespace, tags)
    records = []
    for h in get_adapter_handles(specs):
        if h["name"] in excluded_handles or h["name"].startswith(exp_prefix):
            continue

        class_type = subt(namespace, tags, h["class"])

        prefixes = [class_type]
        if class_type == namespace + "Event":
            prefixes.append(namespace + "Enqueue")
            # Functions prefixed with $xEnqueue are also 'create' functions for event handles

        def has_prefix(f):
            return any(p in f for p in prefixes)

        create_funcs = list(filter(has_prefix, funcs["create"]))
        get_funcs = list(filter(has_prefix, funcs["get"]))
        retain_funcs = list(filter(has_prefix, funcs["retain"]))
        release_funcs = list(filter(has_prefix, funcs["release"]))

        record = {}
        record["handle"] = subt(namespace, tags, h["name"])
        record["create"] = create_funcs
        record["get"] = get_funcs
        record["retain"] = retain_funcs
        record["release"] = release_funcs

        records.append(record)

    return records


def get_queue_related_functions(specs: List[dict], namespace: str, tags) -> List[dict]:
    """
    Public:
        returns a list of objects representing functions that accept $x_queue_handle_t as a first param
    """
    funcs = []
    for s in specs:
        for obj in s["objects"]:
            if re.match(r"function", obj["type"]):
                if obj["params"] and obj["params"][0]["type"] == "$x_queue_handle_t":
                    funcs.append(obj)
    return funcs


def transform_queue_related_function_name(
    namespace, tags, obj, format=["name", "type"]
) -> str:
    """
    Public:
        transform a queue related function using following rules:
        - remove $x prefix
        - make first letter lowercase
        - remove first param (queue)
    """
    function_name = make_func_name(namespace, tags, obj).replace(namespace, "")
    function_name = function_name[0].lower() + function_name[1:]

    if obj["params"][0]["type"] != "$x_queue_handle_t":
        raise ValueError("First parameter is not a queue handle.")

    params = make_param_lines(namespace, tags, obj, format=format)
    params = params[1:]

    return "{}({})".format(function_name, ", ".join(params))


def get_optional_queries(specs: List[dict], namespace: str, tags) -> Dict[str, dict]:
    """
    Public:
        Returns a dictionary mapping info enum types to the list of optional queries
        within that enum. If an enum type doesn't have any optional queries it will
        not appear in the dictionary as a key.
    """
    optional_queries = {}
    for s in specs:
        for obj in s["objects"]:
            if obj["type"] == "enum":
                optional_etors = []
                for e in obj["etors"]:
                    if etor_traits.is_optional_query(e):
                        name = make_enum_name(namespace, tags, e)
                        optional_etors.append(name)
                if optional_etors:
                    type_name = make_type_name(namespace, tags, obj)
                    optional_queries[type_name] = optional_etors
    return optional_queries


def get_etors(obj):
    """
    Public:
        Generator returning non-deprecated enum values.
    """
    if not obj_traits.is_enum(obj):
        raise TypeError("obj parameter is not an enum.")
    for item in obj["etors"]:
        if etor_traits.is_deprecated_etor(item):
            continue
        yield item


def get_dditable_field(obj) -> dict:
    """
    Public:
        Returns the first non-optional non-native handle for the given function.

        If it is a range, `name[0]` will be returned instead of `name`.
    """
    for p in obj["params"]:
        if param_traits.is_optional(p):
            continue
        if "native_handle_t" in p["type"]:
            continue

        if param_traits.is_range(p):
            if not p["type"].endswith("_handle_t*"):
                continue
            return p["name"] + "[0]"
        else:
            if not p["type"].endswith("_handle_t"):
                continue
            return p["name"]
    obj_class = obj["class"]
    name = obj["name"]
    raise RuntimeError(
        f"Function {obj_class}::{name} does not have a non-optional handle argument"
    )
