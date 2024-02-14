"""
 Copyright (C) 2023-2024 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

A helper script for generating Printing API code and HTML documentation that aligns with the code.
The script utilizes provided specifications to generate function objects and their associated details,
such as function names, arguments, and types.
The get_api_types() function generates all Printing API function objects, excluding "extras" functions.
"""
import re

from templates import helper as th
from typing import Dict, List, Union


class Arg:
    """
    Represents a function argument by storing its type and name.

    Args:
    type (str): The type of the argument.
    name (str): The name of the argument.
    """

    def __init__(self, type: str, name: str):
        self.type = type
        self.name = name

    def __repr__(self):
        return " ".join([self.type, self.name])


class PrintArg(Arg):
    """
    Represents an argument that is printed by an API function's call.

    Args:
    type (str): The type of the argument.
    type_name (str): The Unified Runtime type, such as 'ur_base_desc_t' for the argument's type 'const struct ur_base_desc_t'.
    name (str): The name of the argument.

    Attributes:
    type_name (str): The Unified Runtime type name of the argument.

    Properties:
    base_type (str): Returns the base type of the argument, which can be either an enum or a struct.
    """

    def __init__(self, type: str, type_name: str, name: str):
        super().__init__(type, name)
        self.type_name = type_name

    @property
    def base_type(self) -> str:
        match = re.search(r"enum|struct", self.type)
        return match.group(0) if match else ""


class Func:
    """
    A function object storing function's name and arguments
    both in C and C++ versions

    Args:
    namespace (str): The namespace name
    tags (Dict[str, str]): A dictionary of tags from the spec
    obj (Dict[...]): A dictionary representing the function object, retrieved from the spec

    Properties:
    c_name (str): The generated C function name
    c_args (str): The generated C arguments
    cpp_name (str): The generated C++ function name
    print_arg (PrintArg): The argument printed by a Printing API function call
    """

    def __init__(
        self,
        namespace: str,
        tags: Dict[str, str],
        obj: Dict[str, Union[str, List[Dict[str, str]]]],
    ):
        """
        _x (str): The prefix for API functions
        _c_common_args (List[Arg]): A list of common arguments for printing C API functions
        _obj_type (str): The type of the function object, retrieved from the spec
        _name (str): The name of the function object, retrieved from the spec
        _type_name (str): The type name generated based on the spec
        """
        self._x = tags["$x"]
        self._c_common_args = [
            Arg("char *", "buffer"),
            Arg("const size_t", "buff_size"),
            Arg("size_t *", "out_size"),
        ]
        self._obj_type = obj["type"]
        self._name = obj["name"]
        self._type_name = self._make_type_name(namespace, tags, obj)

    @property
    def c_name(self) -> str:
        return self._make_c_func_name()

    @property
    def c_args(self) -> str:
        return self._make_c_args()

    @property
    def cpp_name(self) -> str:
        return "operator<<"

    @property
    def cpp_args(self) -> str:
        attribute = (
            "[[maybe_unused]]" if re.match("const struct", self.print_arg.type) else ""
        )
        return str(
            [
                Arg("std::ostream &", "os"),
                Arg(" ".join([attribute, self.print_arg.type]), self.print_arg.name),
            ]
        ).strip("[]")

    @property
    def print_arg(self) -> PrintArg:
        return self._make_print_arg()

    def _make_type_name(
        self,
        namespace: str,
        tags: Dict[str, str],
        obj: Dict[str, Union[str, List[Dict[str, str]]]],
    ) -> str:
        """
        Generates a Unified Runtime API object type name based on the spec
        """
        if re.match("function", self._obj_type):
            return th.make_pfncb_param_type(namespace, tags, obj)
        elif re.match("enum", self._obj_type):
            return th.make_enum_name(namespace, tags, obj)
        elif re.match("struct", self._obj_type):
            return th.make_type_name(namespace, tags, obj)
        return ""

    def _make_c_func_name(self) -> str:
        """
        Generates the C function name
        """
        if re.match("function", self._obj_type):
            return th.make_func_name_with_prefix(f"{self._x}Print", self._type_name)
        elif re.match(r"enum|struct", self._obj_type):
            return th.make_func_name_with_prefix(f"{self._x}Print", self._name)
        return ""

    def _make_c_args(self) -> str:
        """
        Generates the C arguments
        """
        c_args = [self.print_arg]
        c_args.extend(self._c_common_args)
        return str(c_args).strip("[]")

    def _make_print_arg(self) -> PrintArg:
        """
        Generates the argument object printed by a printing API function call
        """
        if re.match("enum", self._obj_type):
            return PrintArg(
                " ".join(["enum", self._type_name]),
                self._type_name,
                "value",
            )
        elif re.match("struct", self._obj_type):
            return PrintArg(
                " ".join(["const", "struct", self._type_name]),
                self._type_name,
                "params",
            )
        elif re.match("function", self._obj_type):
            return PrintArg(
                " ".join(["const", "struct", self._type_name, "*"]),
                self._type_name,
                "params",
            )


def _get_simple_types_funcs(
    specs: List[
        Dict[
            str,
            Union[
                str,
                Dict[str, Union[str, int]],
                List[Dict[str, Union[str, List[Dict[str, str]]]]],
            ],
        ]
    ],
    namespace: str,
    tags: Dict[str, str],
) -> List[Func]:
    """
    Retrieves function objects for printing Unified Runtime API objects based on the provided specifications.
    """
    return [
        Func(namespace, tags, obj)
        for spec in specs
        for obj in spec["objects"]
        if re.match(r"enum|struct", obj["type"])
    ]


def _get_param_types_funcs(
    specs: List[
        Dict[
            str,
            Union[
                str,
                Dict[str, Union[str, int]],
                List[Dict[str, Union[str, List[Dict[str, str]]]]],
            ],
        ]
    ],
    meta: Dict[str, Dict[str, Dict[str, Union[str, List[str]]]]],
    namespace: str,
    tags: Dict[str, str],
) -> List[Func]:
    """
    Retrieves function objects for printing Unified Runtime *_params_t parameter types
    based on the provided specifications.
    """
    return [
        Func(namespace, tags, obj)
        for tbl in th.get_pfncbtables(specs, meta, namespace, tags)
        for obj in tbl["functions"]
    ]


def get_api_types_funcs(
    specs: List[
        Dict[
            str,
            Union[
                str,
                Dict[str, Union[str, int]],
                List[Dict[str, Union[str, List[Dict[str, str]]]]],
            ],
        ]
    ],
    meta: Dict[str, Dict[str, Dict[str, Union[str, List[str]]]]],
    namespace: str,
    tags: Dict[str, str],
) -> List[Func]:
    """
    Retrieve all printing API function objects, excluding "extras" functions.
    """
    api_types_funcs = _get_simple_types_funcs(specs, namespace, tags)
    api_types_funcs.extend(_get_param_types_funcs(specs, meta, namespace, tags))
    return api_types_funcs
