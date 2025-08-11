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
#include "${x}_lib_loader.hpp"
#include "${x}_loader.hpp"

namespace ur_loader
{
    %for obj in th.get_adapter_functions(specs):
%if 'guard' in obj:
#if ${obj['guard']}
%endif
    <%
        func_name = th.make_func_name(n, tags, obj)
        if func_name.startswith(x):
            func_basename = func_name[len(x):]
        else:
            func_basename = func_name
    %>
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for ${func_name}
    %if 'condition' in obj:
    #if ${th.subt(n, tags, obj['condition'])}
    %endif
    __${x}dlllocal ${x}_result_t ${X}_APICALL ${func_name}(
        %for line in th.make_param_lines(n, tags, obj):
        ${line}
        %endfor
        )
    {
        ${th.get_initial_null_set(obj)}
        %if func_basename == "AdapterGet":
        auto context = getContext();

        size_t adapterIndex = 0;
        if( nullptr != ${obj['params'][1]['name']} && ${obj['params'][0]['name']} !=0)
        {
            for( auto& platform : context->platforms )
            {
                if(platform.initStatus != ${X}_RESULT_SUCCESS)
                    continue;
                platform.dditable.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( 1, &${obj['params'][1]['name']}[adapterIndex], nullptr );
                adapterIndex++;
                if (adapterIndex == NumEntries) {
                    break;
                }
            }
        }

        if( ${obj['params'][2]['name']} != nullptr )
        {
            *${obj['params'][2]['name']} = static_cast<uint32_t>(context->platforms.size());
        }

        return ${X}_RESULT_SUCCESS;
        %else:
        auto *dditable = *reinterpret_cast<${x}_dditable_t **>(${th.get_dditable_field(obj)});

        auto *${th.make_pfn_name(n, tags, obj)} = dditable->${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};
        if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
            return ${X}_RESULT_ERROR_UNINITIALIZED;

        // forward to device-platform
        return ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
        %endif
    }
    %if 'condition' in obj:
    #endif // ${th.subt(n, tags, obj['condition'])}
    %endif
%if 'guard' in obj:
#endif // ${obj['guard']}
%endif

    %endfor
} // namespace ur_loader

#if defined(__cplusplus)
extern "C" {
#endif

%for tbl in th.get_pfntables(specs, meta, n, tags):
%if 'guard' in tbl:
#if ${tbl['guard']}
%endif
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
    )
{
    if( nullptr == pDdiTable )
        return ${X}_RESULT_ERROR_INVALID_NULL_POINTER;

    if( ur_loader::getContext()->version < version )
        return ${X}_RESULT_ERROR_UNSUPPORTED_VERSION;

    ${x}_result_t result = ${X}_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for( auto& platform : ur_loader::getContext()->platforms )
    {
        // statically linked adapter inside of the loader
        if (platform.handle == nullptr)
            continue;

        if(platform.initStatus != ${X}_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<${tbl['pfn']}>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(), "${tbl['export']['name']}"));
        if(!getTable)
            continue;
        platform.initStatus = getTable( version, &platform.dditable.${tbl['name']});
    }

    if( ${X}_RESULT_SUCCESS == result )
    {
        if( ur_loader::getContext()->platforms.size() != 1 || ur_loader::getContext()->forceIntercept )
        {
            // return pointers to loader's DDIs
            %for obj in tbl['functions']:
%if 'guard' in obj and 'guard' not in tbl:
#if ${obj['guard']}
%endif
            %if 'condition' in obj:
        #if ${th.subt(n, tags, obj['condition'])}
            %endif
            pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = ur_loader::${th.make_func_name(n, tags, obj)};
            %if 'condition' in obj:
        #else
            pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = nullptr;
        #endif
            %endif
%if 'guard' in obj and 'guard' not in tbl:
#endif // ${obj['guard']}
%endif
            %endfor
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = ur_loader::getContext()->platforms.front().dditable.${tbl['name']};
        }
    }

    return result;
}
%if 'guard' in tbl:
#endif // ${obj['guard']}
%endif

%endfor

#if defined(__cplusplus)
}
#endif
