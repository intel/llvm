<%
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
 * @file ${n}_api.h
 * @version v${ver}-r${rev}
 *
 */
#ifndef ${N}_API_H_INCLUDED
#define ${N}_API_H_INCLUDED
#if defined(__cplusplus)
#pragma once
#endif

%if n != x:
// 'core' API headers
#include "${x}_api.h"
%else:
// standard headers
#include <stdint.h>
#include <stddef.h>
%endif

#if defined(__cplusplus)
extern "C" {
#endif

%for spec in specs:
%if len(spec['objects']) and 'manifest' not in spec['name']:
%if 'guard' in spec['header']:
// ${spec['header']['guard']}
%endif
// ${th.subt(n, tags, spec['header']['desc'])}
#if !defined(__GNUC__)
#pragma region ${spec['name'].replace(' ', '_')}
#endif
%if 'guard' in spec:
// end ${obj['guard']}
%endif
%endif
%for obj in spec['objects']:
%if "manifest" in obj['type']:
<%continue%>
%elif not re.match(r"class", obj['type']):
%if 'guard' in obj and 'guard' not in spec['header']:
// ${obj['guard']}
%endif
///////////////////////////////////////////////////////////////////////////////
## MACRO ######################################################################
%if re.match(r"macro", obj['type']):
#ifndef ${th.make_macro_name(n, tags, obj, params=False)}
%endif
## CONDITION-START ############################################################
%if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
%endif
## DESCRIPTION ################################################################
%for line in th.make_desc_lines(n, tags, obj):
/// ${line}
%endfor
%for line in th.make_details_lines(n, tags, obj):
/// ${line}
%endfor
## MACRO ######################################################################
%if re.match(r"macro", obj['type']):
#define ${th.make_macro_name(n, tags, obj)}  ${th.subt(n, tags, obj['value'])}
%if 'altvalue' in obj:
#else
#define ${th.make_macro_name(n, tags, obj)}  ${th.subt(n, tags, obj['altvalue'])}
%endif
## TYPEDEF ####################################################################
%elif re.match(r"typedef", obj['type']):
typedef ${th.subt(n, tags, obj['value'])} ${th.make_type_name(n, tags, obj)};
## FPTR TYPEDEF ###############################################################
%elif re.match(r"fptr_typedef", obj['type']):
typedef ${th.subt(n, tags, obj['return'])} (*${th.make_func_name(n, tags, obj)})(
%if 'params' in obj:
%for line in th.make_param_lines(n, tags, obj):
    ${line}
%endfor
%endif
    );
## ENUM #######################################################################
%elif re.match(r"enum", obj['type']):
%if th.type_traits.is_flags(obj['name']):
typedef uint32_t ${th.make_type_name(n, tags, obj)};
%endif
typedef enum ${th.make_enum_name(n, tags, obj)}
{
    %for line in th.make_etor_lines(n, tags, obj):
    ${line}
    %endfor

} ${th.make_enum_name(n, tags, obj)};
%if th.type_traits.is_flags(obj['name']):
/// @brief Bit Mask for validating ${th.make_type_name(n, tags, obj)}
${th.make_flags_bitmask(n, tags, obj, meta)}
%endif
## STRUCT/UNION ###############################################################
%elif re.match(r"struct|union", obj['type']):
typedef ${obj['type']} ${th.make_type_name(n, tags, obj)}
{
    %for line in th.make_member_lines(n, tags, obj):
    ${line}
    %endfor

} ${th.make_type_name(n, tags, obj)};
## FUNCTION ###################################################################
%elif re.match(r"function", obj['type']):
/// 
%for line in th.make_returns_lines(n, tags, obj, meta=meta):
/// ${line}
%endfor
${X}_APIEXPORT ${x}_result_t ${X}_APICALL
${th.make_func_name(n, tags, obj)}(
    %for line in th.make_param_lines(n, tags, obj):
    ${line}
    %endfor
    );
## HANDLE #####################################################################
%elif re.match(r"handle", obj['type']):
%if th.type_traits.is_native_handle(obj['name']):
typedef uintptr_t ${th.subt(n, tags, obj['name'])};
%elif 'alias' in obj:
typedef ${th.subt(n, tags, obj['alias'])} ${th.subt(n, tags, obj['name'])};
%else:
typedef struct ${th.subt(n, tags, obj['name'])}_ *${th.subt(n, tags, obj['name'])};
%endif
%endif
## CONDITION-END ##############################################################
%if 'condition' in obj:
#endif // ${th.subt(n, tags, obj['condition'])}
%endif
## MACRO ######################################################################
%if re.match(r"macro", obj['type']):
#endif // ${th.make_macro_name(n, tags, obj, params=False)}
%endif
%if 'guard' in obj and 'guard' not in spec['header']:
// end ${obj['guard']}
%endif

%endif  # not re.match(r"class", obj['type'])
%endfor # obj in spec['objects']
%if len(spec['objects']) and 'manifest' not in spec['name']:
%if 'guard' in spec:
// ${obj['guard']}
%endif
#if !defined(__GNUC__)
#pragma endregion
#endif
%if 'guard' in spec['header']:
// end ${spec['header']['guard']}
%endif
%endif
%endfor # spec in specs
%if n not in ["zet", "zes"]:
// Intel ${tags['$OneApi']} Unified Runtime API function parameters
#if !defined(__GNUC__)
#pragma region callbacks
#endif
%for tbl in th.get_pfncbtables(specs, meta, n, tags):
%for obj in tbl['functions']:
%if 'guard' in obj:
// ${obj['guard']}
%endif
%if obj['params']:
///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for ${th.make_func_name(n, tags, obj)} 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
%if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
%endif
typedef struct ${th.make_pfncb_param_type(n, tags, obj)}
{
    %for line in th.make_param_lines(n, tags, obj, format=["type*", "name"]):
    ${line};
    %endfor
} ${th.make_pfncb_param_type(n, tags, obj)};
%if 'condition' in obj:
#endif // ${th.subt(n, tags, obj['condition'])}
%endif
%endif
%if 'guard' in obj:
// end ${obj['guard']}
%endif

%endfor
%endfor

#if !defined(__GNUC__)
#pragma endregion
#endif
%endif # not in ["zet", "zes"]:

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ${N}_API_H_INCLUDED
