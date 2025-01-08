<%!
import re
from templates import helper as th
%><%
optional_queries = th.get_optional_queries(specs, namespace, tags)
%>/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${filename}.h
 *
 */

 // Auto-generated file, do not edit.

#pragma once

#include <algorithm>
#include <array>
#include <ur_api.h>

namespace uur {

template <class T> bool isQueryOptional(T) { return false; }

%for type, names in optional_queries.items():
constexpr std::array optional_${type} = {
%for name in names:
    ${name},
%endfor
};

template <> inline bool isQueryOptional(${type} query) {
    return std::find(optional_${type}.begin(),
                     optional_${type}.end(),
                     query) != optional_${type}.end();
}

%endfor

}
