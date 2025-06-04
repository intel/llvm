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
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */
#include "${x}_lib.hpp"
#ifndef DYNAMIC_LOAD_LOADER
#include "${n}_ddi.h"
#endif

namespace ${x}_lib
{
    ///////////////////////////////////////////////////////////////////////////////


    __${x}dlllocal ${x}_result_t context_t::ddiInit()
    {
        ${x}_result_t result = ${X}_RESULT_SUCCESS;

    %for tbl in th.get_pfntables(specs, meta, n, tags):
        if( ${X}_RESULT_SUCCESS == result )
        {
            result = ${tbl['export']['name']}( ${X}_API_VERSION_CURRENT, &${n}DdiTable.${tbl['name']} );
        }

    %endfor
        return result;
    }

} // namespace ${x}_lib
