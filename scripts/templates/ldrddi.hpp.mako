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
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ${name}.hpp
 *
 */
#ifndef UR_LOADER_LDRDDI_H
#define UR_LOADER_LDRDDI_H 1

namespace loader
{
    ///////////////////////////////////////////////////////////////////////////////
    %for obj in th.extract_objs(specs, r"handle"):
    %if 'class' in obj:
    <%
        _handle_t = th.subt(n, tags, obj['name'])
        _object_t = re.sub(r"(\w+)_handle_t", r"\1_object_t", _handle_t)
        _factory_t = re.sub(r"(\w+)_handle_t", r"\1_factory_t", _handle_t)
    %>using ${th.append_ws(_object_t, 35)} = object_t < ${_handle_t} >;
    using ${th.append_ws(_factory_t, 35)} = singleton_factory_t < ${_object_t}, ${_handle_t} >;

    %endif
    %endfor
}

#endif /* UR_LOADER_LDRDDI_H */
