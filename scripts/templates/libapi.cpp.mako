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
 * SPDX-License-Identifier: MIT
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
{
%if re.match("Init", obj['name']):
    static ${x}_result_t result = ${X}_RESULT_SUCCESS;
    std::call_once(${x}_lib::context->initOnce, [device_flags]() {
        result = ${x}_lib::context->Init(device_flags);
    });

    if( ${X}_RESULT_SUCCESS != result )
        return result;

%endif
    auto ${th.make_pfn_name(n, tags, obj)} = ${x}_lib::context->${n}DdiTable.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};
    if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
        return ${X}_RESULT_ERROR_UNINITIALIZED;

    return ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
}
%if 'condition' in obj:
#endif // ${th.subt(n, tags, obj['condition'])}
%endif

%endfor
%endfor
} // extern "C"
