<%
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
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${n}_ddi.h
 * @version v${ver}-r${rev}
 *
 */
#ifndef ${N}_DDI_H_INCLUDED
#define ${N}_DDI_H_INCLUDED
#if defined(__cplusplus)
#pragma once
#endif
#include "${n}_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

%for tbl in th.get_pfntables(specs, meta, n, tags):
%for obj in tbl['functions']:
///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for ${th.make_func_name(n, tags, obj)} 
%if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
%endif
typedef ${x}_result_t (${X}_APICALL *${th.make_pfn_type(n, tags, obj)})(
    %for line in th.make_param_lines(n, tags, obj, format=["type", "delim"]):
    ${line}
    %endfor
    );
%if 'condition' in obj:
#endif // ${th.subt(n, tags, obj['condition'])}
%endif

%endfor
///////////////////////////////////////////////////////////////////////////////
/// @brief Table of ${tbl['name']} functions pointers
typedef struct ${tbl['type']}
{
    %for obj in tbl['functions']:
    %if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
    %endif
    ${th.append_ws(th.make_pfn_type(n, tags, obj), 59)} ${th.make_pfn_name(n, tags, obj)};
    %if 'condition' in obj:
#else
    ${th.append_ws("void*", 59)} ${th.make_pfn_name(n, tags, obj)};
#endif // ${th.subt(n, tags, obj['condition'])}
    %endif
    %endfor
} ${tbl['type']};

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ${tbl['name']} table
///        with current process' addresses
///
/// @returns
///     - ::${X}_RESULT_SUCCESS
///     - ::${X}_RESULT_ERROR_UNINITIALIZED
///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::${X}_RESULT_ERROR_UNSUPPORTED_VERSION
${X}_DLLEXPORT ${x}_result_t ${X}_APICALL
${tbl['export']['name']}(
    %for line in th.make_param_lines(n, tags, tbl['export']):
    ${line}
    %endfor
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for ${tbl['export']['name']}
typedef ${x}_result_t (${X}_APICALL *${tbl['pfn']})(
    %for line in th.make_param_lines(n, tags, tbl['export'], format=["type", "delim"]):
    ${line}
    %endfor
    );

%endfor
///////////////////////////////////////////////////////////////////////////////
/// @brief Container for all DDI tables
typedef struct ${n}_dditable_t
{
%for tbl in th.get_pfntables(specs, meta, n, tags):
    ${th.append_ws(tbl['type'], 35)} ${tbl['name']};
%endfor
} ${n}_dditable_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ${N}_DDI_H_INCLUDED
