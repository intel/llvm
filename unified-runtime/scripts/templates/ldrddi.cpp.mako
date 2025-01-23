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
 * Copyright (C) 2022-2023 Intel Corporation
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
        ${x}_result_t result = ${X}_RESULT_SUCCESS;
        ${th.get_initial_null_set(obj)}

        [[maybe_unused]] auto context = getContext();
        %if func_basename == "AdapterGet":
        
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

        %elif func_basename == "PlatformGet":
        uint32_t total_platform_handle_count = 0;

        for( uint32_t adapter_index = 0; adapter_index < ${obj['params'][1]['name']}; adapter_index++)
        {
            // extract adapter's function pointer table
            auto *dditable =
                *reinterpret_cast<${n}_dditable_t **>( ${obj['params'][0]['name']}[adapter_index]);

            if( ( 0 < ${obj['params'][2]['name']} ) && ( ${obj['params'][2]['name']} == total_platform_handle_count))
                break;

            uint32_t library_platform_handle_count = 0;

            result = dditable->${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( &${obj['params'][0]['name']}[adapter_index], 1, 0, nullptr, &library_platform_handle_count );
            if( ${X}_RESULT_SUCCESS != result ) break;

            if( nullptr != ${obj['params'][3]['name']} && ${obj['params'][2]['name']} !=0)
            {
                if( total_platform_handle_count + library_platform_handle_count > ${obj['params'][2]['name']}) {
                    library_platform_handle_count = ${obj['params'][2]['name']} - total_platform_handle_count;
                }
                result = dditable->${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( &${obj['params'][0]['name']}[adapter_index], 1, library_platform_handle_count, &${obj['params'][3]['name']}[ total_platform_handle_count ], nullptr );
                if( ${X}_RESULT_SUCCESS != result ) break;
            }

            total_platform_handle_count += library_platform_handle_count;
        }

        if( ${X}_RESULT_SUCCESS == result && ${obj['params'][4]['name']} != nullptr )
            *${obj['params'][4]['name']} = total_platform_handle_count;

        %else:
        <%
            ddi_generated=False
        %>
        %for i, item in enumerate(th.get_loader_prologue(n, tags, obj, meta)):
        %if not ddi_generated and ('optional' not in item or not item['optional']) and not '_native_object_' in item['obj']:
        %if 'range' in item:
            auto *dditable = *reinterpret_cast<${x}_dditable_t **>(${item['name']}[ 0 ]);
        %else:
            auto *dditable = *reinterpret_cast<${x}_dditable_t **>(${item['name']});
        %endif
        <%break%>
        %endif

        %endfor

        auto *${th.make_pfn_name(n, tags, obj)} = dditable->${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};
        if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
            return ${X}_RESULT_ERROR_UNINITIALIZED;

        <%
        epilogue = th.get_loader_epilogue(specs, n, tags, obj, meta)
        has_typename = False
        for item in epilogue:
            if 'typename' in item:
                has_typename = True
                break
        %>

        %if has_typename:
            // this value is needed for converting adapter handles to loader handles
            size_t sizeret = 0;
            if (pPropSizeRet == NULL)
                pPropSizeRet = &sizeret;
        %endif

        // forward to device-platform
        result = ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );

        %for i, item in enumerate(epilogue):
        %if 0 == i and not item['release'] and not item['retain']:
        ## TODO: Remove once we have a concrete way for submitting warnings in place.
        %if re.match(r"Enqueue\w+", func_basename):
        // In the event of ERROR_ADAPTER_SPECIFIC we should still attempt to wrap any output handles below.
        if( ${X}_RESULT_SUCCESS != result && ${X}_RESULT_ERROR_ADAPTER_SPECIFIC != result )
            return result;
        %else:
        if( ${X}_RESULT_SUCCESS != result)
            return result;

        %endif
        %endif

        %endfor
        %endif
        return result;
    }
    %if 'condition' in obj:
    #endif // ${th.subt(n, tags, obj['condition'])}
    %endif

    %endfor
} // namespace ur_loader

#if defined(__cplusplus)
extern "C" {
#endif

%for tbl in th.get_pfntables(specs, meta, n, tags):
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
            %if 'condition' in obj:
        #if ${th.subt(n, tags, obj['condition'])}
            %endif
            pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = ur_loader::${th.make_func_name(n, tags, obj)};
            %if 'condition' in obj:
        #else
            pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = nullptr;
        #endif
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

%endfor

#if defined(__cplusplus)
}
#endif
