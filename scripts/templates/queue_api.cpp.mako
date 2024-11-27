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
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */

#include "queue_api.hpp"
#include "ur_util.hpp"

ur_queue_handle_t_::~ur_queue_handle_t_() {}

## FUNCTION ###################################################################
namespace ${x}::level_zero {
%for obj in th.get_queue_related_functions(specs, n, tags):
${x}_result_t
${th.make_func_name(n, tags, obj)}(
    %for line in th.make_param_lines(n, tags, obj, format=["name", "type", "delim"]):
    ${line}
    %endfor
    )
try {
    return ${obj['params'][0]['name']}->${th.transform_queue_related_function_name(n, tags, obj, format=["name"])};
} catch(...) {
    return exceptionToResult(std::current_exception());
}
%endfor
}