<%!
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
%>/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */

// Do not edit. This file is auto generated from a template: scripts/templates/queue_api.cpp.mako

#include "queue_api.hpp"
#include "queue_handle.hpp"
#include "ur_util.hpp"

namespace ur::level_zero::v2 {

ur_queue_t_::~ur_queue_t_() {}

## FUNCTION ###################################################################
%for obj in th.get_queue_related_functions(specs, n, tags):
%if not 'Release' in obj['name'] and not 'Retain' in obj['name']:
%if 'guard' in obj:
#if ${obj['guard']}
%endif
${x}_result_t
${th.make_func_name(n, tags, obj)}(
    %for line in th.make_param_lines(n, tags, obj, format=["name", "type", "delim"], global_handles=True):
    ${line}
    %endfor
    )
try {
    return v2_cast(${obj['params'][0]['name']})->get().${th.transform_queue_related_function_name(n, tags, obj, format=["name"], cast_handles=True)};
} catch(...) {
    return exceptionToResult(std::current_exception());
}
%if 'guard' in obj:
#endif // ${obj['guard']}
%endif
%else:
%if 'guard' in obj:
#endif // ${obj['guard']}
%endif
${x}_result_t
${th.make_func_name(n, tags, obj)}(
    %for line in th.make_param_lines(n, tags, obj, format=["name", "type", "delim"], global_handles=True):
    ${line}
    %endfor
    )
try {
    return v2_cast(${obj['params'][0]['name']})->${th.transform_queue_related_function_name(n, tags, obj, format=["name"], cast_handles=True)};
} catch(...) {
    return exceptionToResult(std::current_exception());
}
%if 'guard' in obj:
#endif // ${obj['guard']}
%endif
%endif
%endfor

} // namespace ur::level_zero::v2
