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
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${n}_print.h
 *
 */
#ifndef ${X}_PRINT_H
#define ${X}_PRINT_H 1

#include "${x}_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

## Declarations ###############################################################
%for spec in specs:
%for obj in spec['objects']:
%if re.match(r"enum", obj['type']):
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Print ${th.make_enum_name(n, tags, obj)} enum
    /// @returns
    ///     - ::${X}_RESULT_SUCCESS
    ///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
    ///         - `NULL == buffer`
    ///     - ::${X}_RESULT_ERROR_INVALID_SIZE
    ///         - `buff_size < out_size`
    ${X}_APIEXPORT ${x}_result_t ${X}_APICALL ${th.make_func_name_with_prefix(f'{x}Print', obj['name'])}(enum ${th.make_enum_name(n, tags, obj)} value, char *buffer, const size_t buff_size, size_t *out_size);

%elif re.match(r"struct", obj['type']):
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Print ${th.make_type_name(n, tags, obj)} struct
    /// @returns
    ///     - ::${X}_RESULT_SUCCESS
    ///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
    ///         - `NULL == buffer`
    ///     - ::${X}_RESULT_ERROR_INVALID_SIZE
    ///         - `buff_size < out_size`
    ${X}_APIEXPORT ${x}_result_t ${X}_APICALL ${th.make_func_name_with_prefix(f'{x}Print', obj['name'])}(const ${obj['type']} ${th.make_type_name(n, tags, obj)} params, char *buffer, const size_t buff_size, size_t *out_size);

%endif
%endfor # obj in spec['objects']
%endfor

%for tbl in th.get_pfncbtables(specs, meta, n, tags):
%for obj in tbl['functions']:
<%
    name = th.make_pfncb_param_type(n, tags, obj)
%>
///////////////////////////////////////////////////////////////////////////////
/// @brief Print ${th.make_pfncb_param_type(n, tags, obj)} params struct
/// @returns
///     - ::${X}_RESULT_SUCCESS
///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::${X}_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
${X}_APIEXPORT ${x}_result_t ${X}_APICALL ${th.make_func_name_with_prefix(f'{x}Print', name)}(const struct ${th.make_pfncb_param_type(n, tags, obj)} *params, char *buffer, const size_t buff_size, size_t *out_size);
%endfor
%endfor

///////////////////////////////////////////////////////////////////////////////
/// @brief Print function parameters
/// @returns
///     - ::${X}_RESULT_SUCCESS
///     - ::${X}_RESULT_ERROR_INVALID_ENUMERATION
///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == params`
///         - `NULL == buffer`
///     - ::${X}_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
${X}_APIEXPORT ${x}_result_t ${X}_APICALL ${x}PrintFunctionParams(enum ${x}_function_t function, const void *params, char *buffer, const size_t buff_size, size_t *out_size);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* ${X}_PRINT_H */
