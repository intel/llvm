<%!
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()

    handle_create_get_retain_release_funcs=th.get_handle_create_get_retain_release_functions(specs, n, tags)
%>/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */
#include "${x}_leak_check.hpp"
#include "${x}_validation_layer.hpp"

namespace ur_validation_layer
{
    %for obj in th.get_adapter_functions(specs):
    <%
        func_name=th.make_func_name(n, tags, obj)

        param_checks=th.make_param_checks(n, tags, obj, meta=meta).items()
        first_errors = [X + "_RESULT_ERROR_INVALID_NULL_POINTER", X + "_RESULT_ERROR_INVALID_NULL_HANDLE"]
        sorted_param_checks = sorted(param_checks, key=lambda pair: False if pair[0] in first_errors else True)

        tracked_params = list(filter(lambda p: any(th.subt(n, tags, p['type']) in [hf['handle'], hf['handle'] + "*"] for hf in handle_create_get_retain_release_funcs), obj['params']))
    %>
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for ${th.make_func_name(n, tags, obj)}
    %if 'condition' in obj:
    #if ${th.subt(n, tags, obj['condition'])}
    %endif
    __${x}dlllocal ${x}_result_t ${X}_APICALL
    ${func_name}(
        %for line in th.make_param_lines(n, tags, obj):
        ${line}
        %endfor
        )
    {
        auto ${th.make_pfn_name(n, tags, obj)} = context.${n}DdiTable.${th.get_table_name(n, tags, obj)}.${th.make_pfn_name(n, tags, obj)};

        if( nullptr == ${th.make_pfn_name(n, tags, obj)} ) {
            return ${X}_RESULT_ERROR_UNINITIALIZED;
        }

        if( context.enableParameterValidation )
        {
            %for key, values in sorted_param_checks:
            %for val in values:
            if( ${val} )
                return ${key};

            %endfor
            %endfor
            %if func_name in th.get_event_wait_list_functions(specs, n, tags):
            if (phEventWaitList != NULL && numEventsInWaitList > 0) {
                for (uint32_t i = 0; i < numEventsInWaitList; ++i) {
                    if (phEventWaitList[i] == NULL) {
                        return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
                    }
                }
            }
            %endif

        }

            %for tp in tracked_params:
            <%
                tp_input_handle_funcs = next((hf for hf in handle_create_get_retain_release_funcs if th.subt(n, tags, tp['type']) == hf['handle'] and "[in]" in tp['desc']), {})
                is_related_create_get_retain_release_func = any(func_name in funcs for funcs in tp_input_handle_funcs.values())
            %>
            %if tp_input_handle_funcs and not is_related_create_get_retain_release_func:
            if (context.enableLifetimeValidation && !refCountContext.isReferenceValid(${tp['name']})) {
                refCountContext.logInvalidReference(${tp['name']});
            }
            %endif
            %endfor

        ${x}_result_t result = ${th.make_pfn_name(n, tags, obj)}( ${", ".join(th.make_param_lines(n, tags, obj, format=["name"]))} );

        %for tp in tracked_params:
        <%
            tp_handle_funcs = next((hf for hf in handle_create_get_retain_release_funcs if th.subt(n, tags, tp['type']) in [hf['handle'], hf['handle'] + "*"]), None)
            is_handle_to_adapter = ("_adapter_handle_t" in tp['type'])
        %>
        %if func_name in tp_handle_funcs['create']:
        if( context.enableLeakChecking && result == UR_RESULT_SUCCESS )
        {
            refCountContext.createRefCount(*${tp['name']});
        }
        %elif func_name in tp_handle_funcs['get']:
        if( context.enableLeakChecking && ${tp['name']} && result == UR_RESULT_SUCCESS )
        {
            for (uint32_t i = ${th.param_traits.range_start(tp)}; i < ${th.param_traits.range_end(tp)}; i++) {
                refCountContext.createOrIncrementRefCount(${tp['name']}[i], ${str(is_handle_to_adapter).lower()});
            }
        }
        %elif func_name in tp_handle_funcs['retain']:
        if( context.enableLeakChecking && result == UR_RESULT_SUCCESS )
        {
            refCountContext.incrementRefCount(${tp['name']}, ${str(is_handle_to_adapter).lower()});
        }
        %elif func_name in tp_handle_funcs['release']:
        if( context.enableLeakChecking && result == UR_RESULT_SUCCESS )
        {
            refCountContext.decrementRefCount(${tp['name']}, ${str(is_handle_to_adapter).lower()});
        }
        %endif
        %endfor

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
    ${X}_DLLEXPORT ${x}_result_t ${X}_APICALL
    ${tbl['export']['name']}(
        %for line in th.make_param_lines(n, tags, tbl['export']):
        ${line}
        %endfor
        )
    {
        auto& dditable = ur_validation_layer::context.${n}DdiTable.${tbl['name']};

        if( nullptr == pDdiTable )
            return ${X}_RESULT_ERROR_INVALID_NULL_POINTER;

        if (UR_MAJOR_VERSION(ur_validation_layer::context.version) != UR_MAJOR_VERSION(version) ||
            UR_MINOR_VERSION(ur_validation_layer::context.version) > UR_MINOR_VERSION(version))
            return ${X}_RESULT_ERROR_UNSUPPORTED_VERSION;

        ${x}_result_t result = ${X}_RESULT_SUCCESS;

        %for obj in tbl['functions']:
        %if 'condition' in obj:
    #if ${th.subt(n, tags, obj['condition'])}
        %endif
        dditable.${th.append_ws(th.make_pfn_name(n, tags, obj), 43)} = pDdiTable->${th.make_pfn_name(n, tags, obj)};
        pDdiTable->${th.append_ws(th.make_pfn_name(n, tags, obj), 41)} = ur_validation_layer::${th.make_func_name(n, tags, obj)};
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
                    codeloc_data) {
        ${x}_result_t result = ${X}_RESULT_SUCCESS;

        if (enabledLayerNames.count(nameFullValidation)) {
            enableParameterValidation = true;
            enableLeakChecking = true;
            enableLifetimeValidation = true;
        } else {
            if (enabledLayerNames.count(nameParameterValidation)) {
                enableParameterValidation = true;
            }
            if (enabledLayerNames.count(nameLeakChecking)) {
                enableLeakChecking = true;
            }
            if (enabledLayerNames.count(nameLifetimeValidation)) {
                // Handle lifetime validation requires leak checking feature.
                enableLifetimeValidation = true;
                enableLeakChecking = true;
            }
        }

        if (!enableParameterValidation && !enableLeakChecking && !enableLifetimeValidation) {
            return result;
        }

        %for tbl in th.get_pfntables(specs, meta, n, tags):
        if ( ${X}_RESULT_SUCCESS == result )
        {
            result = ur_validation_layer::${tbl['export']['name']}( ${X}_API_VERSION_CURRENT, &dditable->${tbl['name']} );
        }

        %endfor
        return result;
    }

    ${x}_result_t context_t::tearDown() {
        ${x}_result_t result = ${X}_RESULT_SUCCESS;

        if (enableLeakChecking) {
            refCountContext.logInvalidReferences();
            refCountContext.clear();
        }
        return result;
    }

} // namespace ur_validation_layer
