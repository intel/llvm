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
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */

#include <iostream>
#include "${x}_tracing_layer.hpp"
#include <stdio.h>

namespace ur_tracing_layer
{
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
    {${th.get_initial_null_set(obj)}
        auto ${th.make_pfn_name(n, tags, obj)} = getContext()->${n}DdiTable.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};

        if( nullptr == ${th.make_pfn_name(n, tags, obj)} )
            return ${X}_RESULT_ERROR_UNSUPPORTED_FEATURE;

        ${th.make_pfncb_param_type(n, tags, obj)} params = { &${",&".join(th.make_param_lines(n, tags, obj, format=["name"]))} };
        uint64_t instance = getContext()->notify_begin(${th.make_func_etor(n, tags, obj)}, "${th.make_func_name(n, tags, obj)}", &params);

        auto &logger = getContext()->logger;
        logger.info("   ---> ${th.make_func_name(n, tags, obj)}\n");

        ${x}_result_t result = ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );

        getContext()->notify_end(${th.make_func_etor(n, tags, obj)}, "${th.make_func_name(n, tags, obj)}", &params, &result, instance);

        if (logger.getLevel() <= logger::Level::INFO) {
            std::ostringstream args_str;
            ur::extras::printFunctionParams(args_str, ${th.make_func_etor(n, tags, obj)}, &params);
            logger.info("   <--- ${th.make_func_name(n, tags, obj)}({}) -> {};\n", args_str.str(), result);
        }

        return result;
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
    __${x}dlllocal ${x}_result_t ${X}_APICALL
    ${tbl['export']['name']}(
        %for line in th.make_param_lines(n, tags, tbl['export']):
        ${line}
        %endfor
        )
    {
        auto& dditable = ur_tracing_layer::getContext()->${n}DdiTable.${tbl['name']};

        if( nullptr == pDdiTable )
            return ${X}_RESULT_ERROR_INVALID_NULL_POINTER;

        if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) != UR_MAJOR_VERSION(version) ||
            UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) > UR_MINOR_VERSION(version))
            return ${X}_RESULT_ERROR_UNSUPPORTED_VERSION;

        ${x}_result_t result = ${X}_RESULT_SUCCESS;

        %for obj in tbl['functions']:
        %if 'condition' in obj:
    #if ${th.subt(n, tags, obj['condition'])}
        %endif
        dditable.${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = pDdiTable->${th.make_pfn_name(n, tags, obj)};
        pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 41)} = ur_tracing_layer::${th.make_func_name(n, tags, obj)};
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

    ${x}_result_t
    context_t::init(ur_dditable_t *dditable,
                    const std::set<std::string> &enabledLayerNames,
                    codeloc_data codelocData) {
        ${x}_result_t result = ${X}_RESULT_SUCCESS;
        
        if(!enabledLayerNames.count(name)) {
            return result;
        }

        // Recreate the logger in case env variables have been modified between
        // program launch and the call to `urLoaderInit`
        logger = logger::create_logger("tracing", true, true);

        ur_tracing_layer::getContext()->codelocData = codelocData;

    %for tbl in th.get_pfntables(specs, meta, n, tags):
        if( ${X}_RESULT_SUCCESS == result )
        {
            result = ur_tracing_layer::${tbl['export']['name']}( ${X}_API_VERSION_CURRENT, &dditable->${tbl['name']} );
        }

    %endfor
        return result;
    }
} /* namespace ur_tracing_layer */
