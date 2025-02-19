<%!
import re
from templates import helper as th

def define_dbg(obj, tags):
    if re.match("class", obj['type']):
        return True
    if 'class' not in obj or obj['class'] in tags:
        return re.match("enum", obj['type']) or re.match("struct|union", obj['type'])
    return False

%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
%>/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 * @brief C++ library for ${n}
 *
 */
#include "${x}_lib.hpp"

extern "C" {

%for s in specs:
## FUNCTION ###################################################################
%for obj in th.filter_items(s['objects'], 'type', 'function'):
///////////////////////////////////////////////////////////////////////////////
%if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
%endif
%for line in th.make_desc_lines(n, tags, obj):
/// ${line}
%endfor
%for line in th.make_details_lines(n, tags, obj):
/// ${line}
%endfor
/// 
%for line in th.make_returns_lines(n, tags, obj, meta=meta):
/// ${line}
%endfor
${x}_result_t ${X}_APICALL
${th.make_func_name(n, tags, obj)}(
    %for line in th.make_param_lines(n, tags, obj):
    ${line}
    %endfor
    )
try {
%if th.obj_traits.is_loader_only(obj):
    return ur_lib::${th.make_func_name(n, tags, obj)}(${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
%else:
    ${th.get_initial_null_set(obj)}auto ${th.make_pfn_name(n, tags, obj)} = ${x}_lib::getContext()->${n}DdiTable.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};
    if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
        return ${X}_RESULT_ERROR_UNINITIALIZED;

    return ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
%endif
} catch(...) { return exceptionToResult(std::current_exception()); }
%if 'condition' in obj:
#endif // ${th.subt(n, tags, obj['condition'])}
%endif

%endfor
%endfor
} // extern "C"
