<%!
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
%>
/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 *
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.def
 * @version v${ver}-r${rev}
 *
 */

 // Auto-generated file, do not edit.

%for tbl in th.get_pfntables(specs, meta, n, tags):
%for obj in tbl['functions']:
_UR_API(${th.make_func_name(n, tags, obj)})
%endfor
%endfor
%for obj in th.get_loader_functions(specs, meta, n, tags):
%if n + "Loader" in obj:
_UR_API(${obj})
%endif
%endfor
