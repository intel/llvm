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
 * @file ${n}_print.cpp
 *
 */

#include "${n}_print.h"
#include "${n}_print.hpp"

#include <algorithm>
#include <sstream>
#include <string.h>

<%def name="ss_copy(item_name)">\
    std::stringstream ss;
    ss << ${item_name};
    return str_copy(&ss, buffer, buff_size, out_size);
</%def>

${x}_result_t str_copy(std::stringstream *ss, char *buff, const size_t buff_size, size_t *out_size) {
    size_t c_str_size = strlen(ss->str().c_str()) + 1;
    if (out_size) {
        *out_size = c_str_size;
    }

    if (buff) {
        if (buff_size < c_str_size) {
            return ${X}_RESULT_ERROR_INVALID_SIZE;
        }

#if defined(_WIN32)
        strncpy_s(buff, buff_size, ss->str().c_str(), c_str_size);
#else
        strncpy(buff, ss->str().c_str(), std::min(buff_size, c_str_size));
#endif
    }
    return ${X}_RESULT_SUCCESS;
}

<%
    api_types_funcs = tph.get_api_types_funcs(specs, meta, n, tags)
%>
%for func in api_types_funcs:
${x}_result_t ${func.c_name}(${func.c_args}) {
    ${ss_copy(func.print_arg.name)}
}

%endfor

${x}_result_t ${x}PrintFunctionParams(enum ${x}_function_t function, const void *params, char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ${x}_result_t result = ${x}::extras::printFunctionParams(ss, function, params);
    if (result != ${X}_RESULT_SUCCESS) {
        return result;
    }
    return str_copy(&ss, buffer, buff_size, out_size);
}
