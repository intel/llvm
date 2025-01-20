<%!
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
    Adapter=adapter.upper()
%>//===--------- ${n}_interface_loader.cpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <${n}_api.h>
#include <${n}_ddi.h>

#include "ur_interface_loader.hpp"

static ur_result_t validateProcInputs(ur_api_version_t version, void *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }
  // Pre 1.0 we enforce loader and adapter must have same version.
  // Post 1.0 only major version match should be required.
  if (version != UR_API_VERSION_CURRENT) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }
  return UR_RESULT_SUCCESS;
}

#ifdef UR_STATIC_ADAPTER_${Adapter}
namespace ${n}::${adapter} {
#elif defined(__cplusplus)
extern "C" {
#endif

%for tbl in th.get_pfntables(specs, meta, n, tags):
${X}_APIEXPORT ${x}_result_t ${X}_APICALL ${tbl['export']['name']}(
    %for line in th.make_param_lines(n, tags, tbl['export'], format=["type", "name", "delim"]):
    ${line}
    %endfor
    )
{
    auto result = validateProcInputs(version, pDdiTable);
    if (UR_RESULT_SUCCESS != result) {
        return result;
    }

    %for obj in tbl['functions']:
    pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = ${n}::${adapter}::${th.make_func_name(n, tags, obj)};
    %endfor

    return result;
}

%endfor

#ifdef UR_STATIC_ADAPTER_${Adapter}
} // namespace ur::${adapter}
#elif defined(__cplusplus)
} // extern "C"
#endif

#ifdef UR_STATIC_ADAPTER_${Adapter}
namespace ur::${adapter} {
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi) {
  if (ddi == nullptr) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  ur_result_t result;

%for tbl in th.get_pfntables(specs, meta, n, tags):
  result = ${n}::${adapter}::${tbl['export']['name']}( ${X}_API_VERSION_CURRENT, &ddi->${tbl['name']} );
  if (result != UR_RESULT_SUCCESS)
    return result;
%endfor

  return result;
}
}
#endif
