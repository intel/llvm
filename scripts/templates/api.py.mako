<%
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
%>"""
 Copyright (C) 2022 Intel Corporation

 SPDX-License-Identifier: MIT

 @file ${n}.py
 @version v${ver}-r${rev}

 """
import platform
from ctypes import *
from enum import *

${"###############################################################################"}
__version__ = "1.0"

%for s in specs:
%for obj in s['objects']:
%if not re.match(r"class", obj['type']) and not re.match(r"function", obj['type']):
${"###############################################################################"}
%for line in th.make_desc_lines(n, tags, obj):
${"##"} ${line}
%endfor
%for line in th.make_details_lines(n, tags, obj):
${"##"} ${line}
%endfor
## MACRO ######################################################################
%if re.match(r"macro", obj['type']):
%if re.match(r".*\(.*\)", obj['name']):
def ${th.make_macro_name(n, tags, obj)}:
    return ${th.subt(n, tags, obj['value'])}
%elif 'altvalue' not in obj and not obj['value'].startswith("__"):
${th.make_macro_name(n, tags, obj)} = ${th.subt(n, tags, obj['value'])}
%else:
# ${th.make_macro_name(n, tags, obj)} not required for python
%endif
## TYPEDEF ####################################################################
%elif re.match(r"typedef", obj['type']):
class ${th.make_type_name(n, tags, obj)}(${th.get_ctype_name(n, tags, {'type': obj['value']})}):
    pass
## ENUM #######################################################################
%elif re.match(r"enum", obj['type']):
class ${re.sub(r"(\w+)_t", r"\1_v", th.make_type_name(n, tags, obj))}(IntEnum):
    %for line in th.make_etor_lines(n, tags, obj, py=True, meta=meta):
    ${line}
    %endfor

class ${th.make_type_name(n, tags, obj)}(c_int):
    def __str__(self):
    %if th.type_traits.is_flags(obj['name']):
        return hex(self.value)
    %else:
        return str(${re.sub(r"(\w+)_t", r"\1_v", th.make_type_name(n, tags, obj))}(self.value))
    %endif

## STRUCT/UNION ###############################################################
%elif re.match(r"struct|union", obj['type']):
class ${th.make_type_name(n, tags, obj)}(Structure):
    _fields_ = [
        %for line in th.make_member_lines(n, tags, obj, py=True, meta=meta):
        ${line}
        %endfor
    ]
## HANDLE #####################################################################
%elif re.match(r"handle", obj['type']):
class ${th.make_type_name(n, tags, obj)}(c_void_p):
    pass
%endif

%endif # !class && !function
%endfor # objects
%endfor # specs
${"###############################################################################"}
__use_win_types = "Windows" == platform.uname()[0]
<%
    tables = th.get_pfntables(specs, meta, n, tags)
%>
%for tbl in tables:
%for obj in tbl['functions']:
${"###############################################################################"}
${"##"} @brief Function-pointer for ${th.make_func_name(n, tags, obj)}
%if 'condition' not in obj:
if __use_win_types:
    _${th.make_func_name(n, tags, obj)}_t = WINFUNCTYPE( ${x}_result_t, ${", ".join(th.make_param_lines(n, tags, obj, py=True, meta=meta, format=["type"]))} )
else:
    _${th.make_func_name(n, tags, obj)}_t = CFUNCTYPE( ${x}_result_t, ${", ".join(th.make_param_lines(n, tags, obj, py=True, meta=meta, format=["type"]))} )
%endif # condition

%endfor # functions

${"###############################################################################"}
${"##"} @brief Table of ${tbl['name']} functions pointers
class _${tbl['type']}(Structure):
    _fields_ = [
        %for obj in tbl['functions']:
        %if 'condition' not in obj:
        %if loop.index < len(tbl['functions'])-1:
        ${th.append_ws("(\""+th.make_pfn_name(n, tags, obj)+"\", c_void_p),", 63)} ## _${th.make_func_name(n, tags, obj)}_t
        %else:
        ${th.append_ws("(\""+th.make_pfn_name(n, tags, obj)+"\", c_void_p)", 63)} ## _${th.make_func_name(n, tags, obj)}_t
        %endif
        %endif # condition
        %endfor
    ]

%endfor # tables
${"###############################################################################"}
class _${n}_dditable_t(Structure):
    _fields_ = [
        %for tbl in tables:
        %if loop.index < len(tables)-1:
        ("${tbl['name']}", _${tbl['type']}),
        %else:
        ("${tbl['name']}", _${tbl['type']})
        %endif
        %endfor
    ]

${"###############################################################################"}
${"##"} @brief ${n} device-driver interfaces
class ${N}_DDI:
    def __init__(self, version : ${x}_api_version_t):
        # load the ${x}_loader library
        if "Windows" == platform.uname()[0]:
            self.__dll = WinDLL("${x}_loader.dll")
        else:
            self.__dll = CDLL("${x}_loader.so")

        # fill the ddi tables
        self.__dditable = _${n}_dditable_t()

        %for tbl in tables:
        # call driver to get function pointers
        _${tbl['name']} = _${tbl['type']}()
        r = ${x}_result_v(self.__dll.${tbl['export']['name']}(version, byref(_${tbl['name']})))
        if r != ${x}_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.${tbl['name']} = _${tbl['name']}

        # attach function interface to function address
        %for obj in tbl['functions']:
        %if 'condition' not in obj:
        self.${th.make_func_name(n, tags, obj)} = _${th.make_func_name(n, tags, obj)}_t(self.__dditable.${tbl['name']}.${th.make_pfn_name(n, tags, obj)})
        %endif
        %endfor # functions

        %endfor # tables
        # success!
