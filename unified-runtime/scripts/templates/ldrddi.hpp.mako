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
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.hpp
 *
 */
#ifndef UR_LOADER_LDRDDI_H
#define UR_LOADER_LDRDDI_H 1

#include "${x}_object.hpp"
#include "${x}_singleton.hpp"

namespace ur_loader
{
    ///////////////////////////////////////////////////////////////////////////////
    <%
        factories = []
    %>
    %for obj in th.get_adapter_handles(specs):
    %if 'class' in obj:
    <%
        _handle_t = th.subt(n, tags, obj['name'])
        _object_t = re.sub(r"(\w+)_handle_t", r"\1_object_t", _handle_t)
        _factory_t = re.sub(r"(\w+)_handle_t", r"\1_factory_t", _handle_t)
        _factory = re.sub(r"(\w+)_handle_t", r"\1_factory", _handle_t)
        factories.append((_factory_t, _factory))
    %>using ${th.append_ws(_object_t, 35)} = object_t < ${_handle_t} >;
    using ${th.append_ws(_factory_t, 35)} = singleton_factory_t < ${_object_t}, ${_handle_t} >;

    %endif
    %endfor

    struct handle_factories {
        %for (f_t, f) in factories:
            ${f_t} ${f};
        %endfor
    };

}

#endif /* UR_LOADER_LDRDDI_H */
