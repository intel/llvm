<%!
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
    Adapter=adapter.upper()
%>//===--------- ${n}_interface_loader.hpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <${n}_api.h>
#include <${n}_ddi.h>

namespace ${n}::${adapter} {
%for s in specs:
%for obj in th.filter_items(s['objects'], 'type', 'function'):
%if not th.obj_traits.is_loader_only(obj):
${x}_result_t ${th.make_func_name(n, tags, obj)}(
    %for line in th.make_param_lines(n, tags, obj, format=["type", "name", "delim"]):
    ${line}
    %endfor
    );
%endif
%endfor
%endfor
#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi);
#endif
}
