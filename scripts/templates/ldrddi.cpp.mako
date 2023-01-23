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
 * SPDX-License-Identifier: MIT
 *
 * @file ${name}.cpp
 *
 */
#include "${x}_loader.hpp"

namespace loader
{
    ///////////////////////////////////////////////////////////////////////////////
    %for obj in th.extract_objs(specs, r"handle"):
    %if 'class' in obj:
    <%
        _handle_t = th.subt(n, tags, obj['name'])
        _factory_t = re.sub(r"(\w+)_handle_t", r"\1_factory_t", _handle_t)
        _factory = re.sub(r"(\w+)_handle_t", r"\1_factory", _handle_t)
    %>${th.append_ws(_factory_t, 35)} ${_factory};
    %endif
    %endfor

    %for obj in th.extract_objs(specs, r"function"):
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for ${th.make_func_name(n, tags, obj)}
    %if 'condition' in obj:
    #if ${th.subt(n, tags, obj['condition'])}
    %endif
    __${x}dlllocal ${x}_result_t ${X}_APICALL
    ${th.make_func_name(n, tags, obj)}(
        %for line in th.make_param_lines(n, tags, obj):
        ${line}
        %endfor
        )
    {
        ${x}_result_t result = ${X}_RESULT_SUCCESS;<%
        add_local = False
        arrays_to_delete = []
    %>

        %if re.match(r"Init", obj['name']):
        bool atLeastOneplatformValid = false;
        for( auto& platform : context->platforms )
        {
            if(platform.initStatus != ${X}_RESULT_SUCCESS)
                continue;
            platform.initStatus = platform.dditable.${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
            if(platform.initStatus == ${X}_RESULT_SUCCESS)
                atLeastOneplatformValid = true;
        }

        if(!atLeastOneplatformValid)
            result=${X}_RESULT_ERROR_UNINITIALIZED;

        %elif re.match(r"\w+TearDown$", th.make_func_name(n, tags, obj)):

        for( auto& platform : context->platforms )
        {
            platform.dditable.${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
        }

        %elif re.match(r"\w+PlatformGet$", th.make_func_name(n, tags, obj)):
        uint32_t total_platform_handle_count = 0;

        for( auto& platform : context->platforms )
        {
            if(platform.initStatus != ${X}_RESULT_SUCCESS)
                continue;

            if( ( 0 < ${obj['params'][0]['name']} ) && ( ${obj['params'][0]['name']} == total_platform_handle_count))
                break;

            uint32_t library_platform_handle_count = 0;

            result = platform.dditable.${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( 0, nullptr, &library_platform_handle_count );
            if( ${X}_RESULT_SUCCESS != result ) break;

            if( nullptr != ${obj['params'][1]['name']} && ${obj['params'][0]['name']} !=0)
            {
                if( total_platform_handle_count + library_platform_handle_count > ${obj['params'][0]['name']}) {
                    library_platform_handle_count = ${obj['params'][0]['name']} - total_platform_handle_count;
                }
                result = platform.dditable.${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( library_platform_handle_count, &${obj['params'][1]['name']}[ total_platform_handle_count ], nullptr );
                if( ${X}_RESULT_SUCCESS != result ) break;

                try
                {
                    for( uint32_t i = 0; i < library_platform_handle_count; ++i ) {
                        uint32_t platform_index = total_platform_handle_count + i;
                        ${obj['params'][1]['name']}[ platform_index ] = reinterpret_cast<${n}_platform_handle_t>(
                            ${n}_platform_factory.getInstance( ${obj['params'][1]['name']}[ platform_index ], &platform.dditable ) );
                    }
                }
                catch( std::bad_alloc& )
                {
                    result = ${X}_RESULT_ERROR_OUT_OF_HOST_MEMORY;
                }
            }

            total_platform_handle_count += library_platform_handle_count;
        }

        if( ${X}_RESULT_SUCCESS == result && ${obj['params'][2]['name']} != nullptr )
            *${obj['params'][2]['name']} = total_platform_handle_count;

        %else:
        %for i, item in enumerate(th.get_loader_prologue(n, tags, obj, meta)):
        %if 0 == i:
        // extract platform's function pointer table
        auto dditable = reinterpret_cast<${item['obj']}*>( ${item['pointer']}${item['name']} )->dditable;
        auto ${th.make_pfn_name(n, tags, obj)} = dditable->${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};
        if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
            return ${X}_RESULT_ERROR_UNINITIALIZED;

        %endif
        %if 'range' in item:
        <%
        add_local = True%>// convert loader handles to platform handles
        auto ${item['name']}Local = new ${item['type']} [${item['range'][1]}];
        <%
        arrays_to_delete.append(item['name']+ 'Local')
        %>for( size_t i = ${item['range'][0]}; ( nullptr != ${item['name']} ) && ( i < ${item['range'][1]} ); ++i )
            ${item['name']}Local[ i ] = reinterpret_cast<${item['obj']}*>( ${item['name']}[ i ] )->handle;
        %else:
        // convert loader handle to platform handle
        %if item['optional']:
        ${item['name']} = ( ${item['name']} ) ? reinterpret_cast<${item['obj']}*>( ${item['name']} )->handle : nullptr;
        %else:
        ${item['name']} = reinterpret_cast<${item['obj']}*>( ${item['name']} )->handle;
        %endif
        %endif

        %endfor
        // forward to device-platform
        %if add_local:
        result = ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name", "local"]))} );
        %for array_name in arrays_to_delete:
        delete []${array_name};
        %endfor
        %else:
        result = ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
        %endif
<%
        del arrays_to_delete
        del add_local%>
        %for i, item in enumerate(th.get_loader_epilogue(n, tags, obj, meta)):
        %if 0 == i:
        if( ${X}_RESULT_SUCCESS != result )
            return result;

        %endif
        %if item['release']:
        // release loader handle
        ${item['factory']}.release( ${item['name']} );
        %else:
        try
        {
            %if 'range' in item:
            // convert platform handles to loader handles
            for( size_t i = ${item['range'][0]}; ( nullptr != ${item['name']} ) && ( i < ${item['range'][1]} ); ++i )
                ${item['name']}[ i ] = reinterpret_cast<${item['type']}>(
                    ${item['factory']}.getInstance( ${item['name']}[ i ], dditable ) );
            %else:
            // convert platform handle to loader handle
            %if item['optional']:
            if( nullptr != ${item['name']} )
                *${item['name']} = reinterpret_cast<${item['type']}>(
                    ${item['factory']}.getInstance( *${item['name']}, dditable ) );
            %else:
            *${item['name']} = reinterpret_cast<${item['type']}>(
                ${item['factory']}.getInstance( *${item['name']}, dditable ) );
            %endif
            %endif
        }
        catch( std::bad_alloc& )
        {
            result = ${X}_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        %endif

        %endfor
        %endif
        return result;
    }
    %if 'condition' in obj:
    #endif // ${th.subt(n, tags, obj['condition'])}
    %endif

    %endfor
} // namespace loader

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
    if( loader::context->platforms.size() < 1 )
        return ${X}_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return ${X}_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return ${X}_RESULT_ERROR_UNSUPPORTED_VERSION;

    ${x}_result_t result = ${X}_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != ${X}_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<${tbl['pfn']}>(
            GET_FUNCTION_PTR( platform.handle, "${tbl['export']['name']}") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.${n}.${tbl['name']});
        if(getTableResult == ${X}_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        %if tbl['experimental'] is False:
        else
            platform.initStatus = getTableResult;
        %endif
    }

    %if tbl['experimental'] is False: #//Experimental Tables may not be implemented in platform
    if(!atLeastOneplatformValid)
        result = ${X}_RESULT_ERROR_UNINITIALIZED;
    else
        result = ${X}_RESULT_SUCCESS;
    %endif

    if( ${X}_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            %for obj in tbl['functions']:
            %if 'condition' in obj:
        #if ${th.subt(n, tags, obj['condition'])}
            %endif
            pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = loader::${th.make_func_name(n, tags, obj)};
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
            *pDdiTable = loader::context->platforms.front().dditable.${n}.${tbl['name']};
        }
    }

    return result;
}

%endfor

#if defined(__cplusplus)
};
#endif
