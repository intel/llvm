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
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */
#include "${x}_lib.hpp"
#ifndef DYNAMIC_LOAD_LOADER
#include "unified-runtime/${n}_ddi.h"
#endif

namespace ${x}_lib
{
    ///////////////////////////////////////////////////////////////////////////////


    __${x}dlllocal ${x}_result_t context_t::ddiInit()
    {
        ${x}_result_t result = ${X}_RESULT_SUCCESS;

    %for tbl in th.get_pfntables(specs, meta, n, tags):
%if 'guard' in tbl:
#if ${tbl['guard']}
%endif
        if( ${X}_RESULT_SUCCESS == result )
        {
            result = ${tbl['export']['name']}( ${X}_API_VERSION_CURRENT, &${n}DdiTable.${tbl['name']} );
        }
%if 'guard' in tbl:
#endif // ${tbl['guard']}
%endif

    %endfor
        return result;
    }

} // namespace ${x}_lib
