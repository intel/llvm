<%!
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
    Adapter=adapter.upper()

    # See ur_interface_loader.cpp.mako: map the Level Zero v1/v2 adapter ids to
    # their real nested C++ namespace.
    adapter_namespace={
        'level_zero': 'ur::level_zero::v1',
        'level_zero_v2': 'ur::level_zero::v2',
    }.get(adapter, 'ur::'+adapter)
%>//===--------- ${n}_interface_loader.hpp - Level Zero Adapter ------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <unified-runtime/${n}_api.h>
#include <unified-runtime/${n}_ddi.h>

namespace ${adapter_namespace} {
%for s in specs:
%for obj in th.filter_items(s['objects'], 'type', 'function'):
%if 'guard' in obj:
// ${obj['guard']}
%endif
%if not th.obj_traits.is_loader_only(obj):
${x}_result_t ${th.make_func_name(n, tags, obj)}(
    %for line in th.make_param_lines(n, tags, obj, format=["type", "name", "delim"], global_handles=True):
    ${line}
    %endfor
    );
%endif
%if 'guard' in obj:
// end ${obj['guard']}
%endif
%endfor
%endfor
#ifdef UR_STATIC_ADAPTER_${Adapter}
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi);
#endif

struct ddi_getter {
  const static ${x}_dditable_t *value();
};
} // namespace ${adapter_namespace}
