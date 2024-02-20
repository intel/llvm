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
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */
#include "${x}_lib_loader.hpp"
#include "${x}_loader.hpp"

namespace ur_loader
{
    ///////////////////////////////////////////////////////////////////////////////
    %for obj in th.get_adapter_handles(specs):
    %if 'class' in obj:
    <%
        _handle_t = th.subt(n, tags, obj['name'])
        _factory_t = re.sub(r"(\w+)_handle_t", r"\1_factory_t", _handle_t)
        _factory = re.sub(r"(\w+)_handle_t", r"\1_factory", _handle_t)
    %>${th.append_ws(_factory_t, 35)} ${_factory};
    %endif
    %endfor

    %for obj in th.get_adapter_functions(specs):
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
    %>

        %if re.match(r"\w+AdapterGet$", th.make_func_name(n, tags, obj)):
        
        size_t adapterIndex = 0;
        if( nullptr != ${obj['params'][1]['name']} && ${obj['params'][0]['name']} !=0)
        {
            for( auto& platform : context->platforms )
            {
                platform.dditable.${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( 1, &${obj['params'][1]['name']}[adapterIndex], nullptr );
                try
                {
                    ${obj['params'][1]['name']}[adapterIndex] = reinterpret_cast<${n}_adapter_handle_t>(${n}_adapter_factory.getInstance(
                        ${obj['params'][1]['name']}[adapterIndex], &platform.dditable
                    ));
                }
                catch( std::bad_alloc &)
                {
                    result = ${X}_RESULT_ERROR_OUT_OF_HOST_MEMORY;
                    break;
                }
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

        %elif re.match(r"\w+PlatformGet$", th.make_func_name(n, tags, obj)):
        uint32_t total_platform_handle_count = 0;

        for( uint32_t adapter_index = 0; adapter_index < ${obj['params'][1]['name']}; adapter_index++)
        {
            // extract adapter's function pointer table
            auto dditable =
                reinterpret_cast<${n}_platform_object_t *>( ${obj['params'][0]['name']}[adapter_index])->dditable;

            if( ( 0 < ${obj['params'][2]['name']} ) && ( ${obj['params'][2]['name']} == total_platform_handle_count))
                break;

            uint32_t library_platform_handle_count = 0;

            result = dditable->${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( &${obj['params'][0]['name']}[adapter_index], 1, 0, nullptr, &library_platform_handle_count );
            if( ${X}_RESULT_SUCCESS != result ) break;

            if( nullptr != ${obj['params'][3]['name']} && ${obj['params'][2]['name']} !=0)
            {
                if( total_platform_handle_count + library_platform_handle_count > ${obj['params'][2]['name']}) {
                    library_platform_handle_count = ${obj['params'][2]['name']} - total_platform_handle_count;
                }
                result = dditable->${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)}( &${obj['params'][0]['name']}[adapter_index], 1, library_platform_handle_count, &${obj['params'][3]['name']}[ total_platform_handle_count ], nullptr );
                if( ${X}_RESULT_SUCCESS != result ) break;

                try
                {
                    for( uint32_t i = 0; i < library_platform_handle_count; ++i ) {
                        uint32_t platform_index = total_platform_handle_count + i;
                        ${obj['params'][3]['name']}[ platform_index ] = reinterpret_cast<${n}_platform_handle_t>(
                            ${n}_platform_factory.getInstance( ${obj['params'][3]['name']}[ platform_index ], dditable ) );
                    }
                }
                catch( std::bad_alloc& )
                {
                    result = ${X}_RESULT_ERROR_OUT_OF_HOST_MEMORY;
                }
            }

            total_platform_handle_count += library_platform_handle_count;
        }

        if( ${X}_RESULT_SUCCESS == result && ${obj['params'][4]['name']} != nullptr )
            *${obj['params'][4]['name']} = total_platform_handle_count;

        %else:
        <%param_replacements={}%>
        %for i, item in enumerate(th.get_loader_prologue(n, tags, obj, meta)):
        %if not '_native_object_' in item['obj'] or th.make_func_name(n, tags, obj) == 'urPlatformCreateWithNativeHandle':
        // extract platform's function pointer table
        auto dditable = reinterpret_cast<${item['obj']}*>( ${item['pointer']}${item['name']} )->dditable;
        auto ${th.make_pfn_name(n, tags, obj)} = dditable->${n}.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};
        if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
            return ${X}_RESULT_ERROR_UNINITIALIZED;

        <%break%>
        %endif
        %endfor
        %for i, item in enumerate(th.get_loader_prologue(n, tags, obj, meta)):
        %if 'range' in item:
        <%
        add_local = True
        param_replacements[item['name']] = item['name'] + 'Local.data()'%>// convert loader handles to platform handles
        auto ${item['name']}Local = std::vector<${item['type']}>(${item['range'][1]});
        for( size_t i = ${item['range'][0]}; i < ${item['range'][1]}; ++i )
            ${item['name']}Local[ i ] = reinterpret_cast<${item['obj']}*>( ${item['name']}[ i ] )->handle;
        %else:
        %if not '_native_object_' in item['obj'] or th.make_func_name(n, tags, obj) == 'urPlatformCreateWithNativeHandle':
        // convert loader handle to platform handle
        %if item['optional']:
        ${item['name']} = ( ${item['name']} ) ? reinterpret_cast<${item['obj']}*>( ${item['name']} )->handle : nullptr;
        %else:
        ${item['name']} = reinterpret_cast<${item['obj']}*>( ${item['name']} )->handle;
        %endif
        %endif
        %endif

        %endfor

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

        ## Here we deal with handles buried inside struct type parameters. First
        ## we create a local copy of the struct, then we convert all the handles
        ## in that local copy and set the parameter to point to it before forwarding
        ## it to the final API call.
        <% handle_structs = th.get_object_handle_structs_to_convert(n, tags, obj, meta) %>
        %if handle_structs:
        // Deal with any struct parameters that have handle members we need to convert.
        %for struct in handle_structs:
            %if struct['optional']:
            ${struct['type']} ${struct['name']}Local = {};
            if(${struct['name']})
                ${struct['name']}Local = *${struct['name']};
            %else:
            auto ${struct['name']}Local = *${struct['name']};
            %endif
        %endfor

        %for struct in handle_structs:
        %for member in struct['members']:
            ## If this member has a handle_members field that means it's a range of
            ## structs which each contain a handle to convert.
            %if 'handle_members' in member:
                ## we use the parent info stripped of derefs for a unique variable name
                <%
                parent_no_deref = th.strip_deref(member['parent'])
                range_vector_name = struct['name'] + parent_no_deref + member['name']
                ## we need to check if range bounds are literals or variables: variables
                ## need the full reference chain prepended to them
                range_start = member['range_start']
                if not re.match(r"[0-9]+$", range_start):
                    range_start = struct['name'] + "->" + member['parent'] + range_start
                range_end = member['range_end']
                if not re.match(r"[0-9]+$", range_end):
                    range_end = struct['name'] + "->" + member['parent'] + range_end %>
                std::vector<${member['type']}> ${range_vector_name};
                for(uint32_t i = ${range_start}; i < ${range_end}; i++) {
                    ${member['type']} NewRangeStruct = ${struct['name']}Local.${member['parent']}${member['name']}[i];
                    %for handle_member in member['handle_members']:
                    %if handle_member['optional']:
                    if(NewRangeStruct.${handle_member['parent']}${handle_member['name']})
                    %endif
                    NewRangeStruct.${handle_member['parent']}${handle_member['name']} =
                        reinterpret_cast<${handle_member['obj_name']}*>(
                            NewRangeStruct.${handle_member['parent']}${handle_member['name']})
                            ->handle;
                    %endfor

                    ${range_vector_name}.push_back(NewRangeStruct);
                }
                ${struct['name']}Local.${member['parent']}${member['name']} = ${range_vector_name}.data();
            ## If the member has range_start then its a range of handles
            %elif 'range_start' in member:
                ## we use the parent info stripped of derefs for a unique variable name
                <%
                parent_no_deref = th.strip_deref(member['parent'])
                range_vector_name = struct['name'] + parent_no_deref + member['name'] %>
                std::vector<${member['type']}> ${range_vector_name};
                for(uint32_t i = 0;i < ${struct['name']}->${member['parent']}${member['range_end']};i++) {
                    ${range_vector_name}.push_back(reinterpret_cast<${member['obj_name']}*>(${struct['name']}->${member['parent']}${member['name']}[i])->handle);
                }
                ${struct['name']}Local.${member['parent']}${member['name']} = ${range_vector_name}.data();
            %else:
                %if member['optional']:
                if(${struct['name']}Local.${member['parent']}${member['name']})
                %endif
                ${struct['name']}Local.${member['parent']}${member['name']} =
                    reinterpret_cast<${member['obj_name']}*>(
                        ${struct['name']}Local.${member['parent']}${member['name']})->handle;
            %endif
        %endfor
        %endfor

        // Now that we've converted all the members update the param pointers
        %for struct in handle_structs:
            %if struct['optional']:
            if(${struct['name']})
            %endif
            ${struct['name']} = &${struct['name']}Local;
        %endfor
        %endif

        // forward to device-platform
        %if add_local:
        result = ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name", "local"], replacements=param_replacements))} );
        %else:
        result = ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
        %endif
<% 
        del param_replacements
        del add_local
        %>
        %for i, item in enumerate(epilogue):
        %if 0 == i:
        if( ${X}_RESULT_SUCCESS != result )
            return result;

        %endif
        %if item['release']:
        // release loader handle
        ${item['factory']}.release( ${item['name']} );
        %elif not '_native_object_' in item['obj'] or th.make_func_name(n, tags, obj) == 'urPlatformCreateWithNativeHandle':
        try
        {
            %if 'typename' in item:
            if (${item['name']} != nullptr) {
                switch (${item['typename']}) {
                    %for etor in item['etors']:
                        case ${etor['name']}: {
                            ${etor['type']} *handles = reinterpret_cast<${etor['type']} *>(${item['name']});
                            size_t nelements = *pPropSizeRet / sizeof(${etor['type']});
                            for (size_t i = 0; i < nelements; ++i) {
                                if (handles[i] != nullptr) {
                                    handles[i] = reinterpret_cast<${etor['type']}>(
                                        ${etor['factory']}.getInstance( handles[i], dditable ) );
                                }
                            }
                        } break;
                    %endfor
                    default: {} break;
                }
            }
            %elif 'range' in item:
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

    if( ur_loader::context->version < version )
        return ${X}_RESULT_ERROR_UNSUPPORTED_VERSION;

    ${x}_result_t result = ${X}_RESULT_SUCCESS;

    // Load the device-platform DDI tables
    for( auto& platform : ur_loader::context->platforms )
    {
        if(platform.initStatus != ${X}_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<${tbl['pfn']}>(
            ur_loader::LibLoader::getFunctionPtr(platform.handle.get(), "${tbl['export']['name']}"));
        if(!getTable)
            continue;
        platform.initStatus = getTable( version, &platform.dditable.${n}.${tbl['name']});
    }

    if( ${X}_RESULT_SUCCESS == result )
    {
        if( ur_loader::context->platforms.size() != 1 || ur_loader::context->forceIntercept )
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
            *pDdiTable = ur_loader::context->platforms.front().dditable.${n}.${tbl['name']};
        }
    }

    return result;
}

%endfor

#if defined(__cplusplus)
}
#endif
