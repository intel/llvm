<%!
import re
from templates import print_helper as tph
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
%>/*
 *
 * Copyright (C) 2023-2024 Intel Corporation
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

<%
    api_types_funcs = tph.get_api_types_funcs(specs, meta, n, tags)
%>
## Declarations ###############################################################
%for func in api_types_funcs:
///////////////////////////////////////////////////////////////////////////////
/// @brief Print ${func.print_arg.type_name} ${func.print_arg.base_type}
/// @returns
///     - ::${X}_RESULT_SUCCESS
///     - ::${X}_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
${X}_APIEXPORT ${x}_result_t ${X}_APICALL ${func.c_name}(${func.c_args});

%endfor

///////////////////////////////////////////////////////////////////////////////
/// @brief Print function parameters
/// @returns
///     - ::${X}_RESULT_SUCCESS
///     - ::${X}_RESULT_ERROR_INVALID_ENUMERATION
///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == params`
///     - ::${X}_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
${X}_APIEXPORT ${x}_result_t ${X}_APICALL ${x}PrintFunctionParams(enum ${x}_function_t function, const void *params, char *buffer, const size_t buff_size, size_t *out_size);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* ${X}_PRINT_H */
