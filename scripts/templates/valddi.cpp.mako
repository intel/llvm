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
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ${name}.cpp
 *
 */
#include "${x}_validation_layer.hpp"

namespace validation_layer
{
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
        auto ${th.make_pfn_name(n, tags, obj)} = context.${n}DdiTable.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};

        if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
            return ${X}_RESULT_ERROR_UNSUPPORTED_FEATURE;

        if( context.enableParameterValidation )
        {
            %for key, values in th.make_param_checks(n, tags, obj, meta=meta).items():
            %for val in values:
            if( ${val} )
                return ${key};

            %endfor
            %endfor
        }

        return ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );
    }
    %if 'condition' in obj:
    #endif // ${th.subt(n, tags, obj['condition'])}
    %endif

    %endfor
    %for tbl in th.get_pfntables(specs, meta, n, tags):
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Exported function for filling application's ${tbl['name']} table
    ///        with current process' addresses
    ///
    /// @returns
    ///     - ::${X}_RESULT_SUCCESS
    ///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
    ///     - ::${X}_RESULT_ERROR_UNSUPPORTED_VERSION
    ${X}_DLLEXPORT ${x}_result_t ${X}_APICALL
    ${tbl['export']['name']}(
        %for line in th.make_param_lines(n, tags, tbl['export']):
        ${line}
        %endfor
        )
    {
        auto& dditable = validation_layer::context.${n}DdiTable.${tbl['name']};

        if( nullptr == pDdiTable )
            return ${X}_RESULT_ERROR_INVALID_NULL_POINTER;

        if (UR_MAJOR_VERSION(validation_layer::context.version) != UR_MAJOR_VERSION(version) ||
            UR_MINOR_VERSION(validation_layer::context.version) > UR_MINOR_VERSION(version))
            return ${X}_RESULT_ERROR_UNSUPPORTED_VERSION;

        ${x}_result_t result = ${X}_RESULT_SUCCESS;

        %for obj in tbl['functions']:
        %if 'condition' in obj:
    #if ${th.subt(n, tags, obj['condition'])}
        %endif
        dditable.${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = pDdiTable->${th.make_pfn_name(n, tags, obj)};
        pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 41)} = validation_layer::${th.make_func_name(n, tags, obj)};
        %if 'condition' in obj:
    #else
        dditable.${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = nullptr;
        pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 41)} = nullptr;
    #endif
        %endif

        %endfor
        return result;
    }

    %endfor
    ${x}_result_t context_t::init(
        ${x}_dditable_t *dditable
        )
    {
        ${x}_result_t result = ${X}_RESULT_SUCCESS;

        %for tbl in th.get_pfntables(specs, meta, n, tags):
        if ( ${X}_RESULT_SUCCESS == result )
        {
            result = validation_layer::${tbl['export']['name']}( ${X}_API_VERSION_CURRENT, &dditable->${tbl['name']} );
        }

        %endfor
        return result;
    }

} // namespace validation_layer
