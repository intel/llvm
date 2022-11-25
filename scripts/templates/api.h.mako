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
 * SPDX-License-Identifier: MIT
 *
 * @file ${n}_api.h
 * @version v${ver}-r${rev}
 *
 */
#ifndef _${N}_API_H
#define _${N}_API_H
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
%if len(spec['objects']):
// ${th.subt(n, tags, spec['header']['desc'])}
#if !defined(__GNUC__)
#pragma region ${spec['name']}
#endif
%endif
%for obj in spec['objects']:
%if not re.match(r"class", obj['type']):
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
typedef ${th.subt(n, tags, obj['return'])} (${th.make_func_name(n, tags, obj)})(
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
typedef enum _${th.make_enum_name(n, tags, obj)}
{
    %for line in th.make_etor_lines(n, tags, obj):
    ${line}
    %endfor

} ${th.make_enum_name(n, tags, obj)};
## STRUCT/UNION ###############################################################
%elif re.match(r"struct|union", obj['type']):
typedef ${obj['type']} _${th.make_type_name(n, tags, obj)}
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
%if 'alias' in obj:
typedef ${th.subt(n, tags, obj['alias'])} ${th.subt(n, tags, obj['name'])};
%else:
typedef struct _${th.subt(n, tags, obj['name'])} *${th.subt(n, tags, obj['name'])};
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

%endif  # not re.match(r"class", obj['type'])
%endfor # obj in spec['objects']
%if len(spec['objects']):
#if !defined(__GNUC__)
#pragma endregion
#endif
%endif
%endfor # spec in specs
%if n not in ["zet", "zes"]:
// Intel ${tags['$OneApi']} Level-Zero API Callbacks
#if !defined(__GNUC__)
#pragma region callbacks
#endif
%for tbl in th.get_pfncbtables(specs, meta, n, tags):
%for obj in tbl['functions']:
///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for ${th.make_func_name(n, tags, obj)} 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
%if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
%endif
typedef struct _${th.make_pfncb_param_type(n, tags, obj)}
{
    %for line in th.make_param_lines(n, tags, obj, format=["type*", "name"]):
    ${line};
    %endfor
} ${th.make_pfncb_param_type(n, tags, obj)};
%if 'condition' in obj:
#endif // ${th.subt(n, tags, obj['condition'])}
%endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for ${th.make_func_name(n, tags, obj)} 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
%if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
%endif
typedef void (${X}_APICALL *${th.make_pfncb_type(n, tags, obj)})(
    ${th.make_pfncb_param_type(n, tags, obj)}* params,
    ${x}_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );
%if 'condition' in obj:
#endif // ${th.subt(n, tags, obj['condition'])}
%endif

%endfor
///////////////////////////////////////////////////////////////////////////////
/// @brief Table of ${tbl['name']} callback functions pointers
typedef struct _${tbl['type']}
{
    %for obj in tbl['functions']:
    %if 'condition' in obj:
#if ${th.subt(n, tags, obj['condition'])}
    %endif
    ${th.append_ws(th.make_pfncb_type(n, tags, obj), 63)} ${th.make_pfncb_name(n, tags, obj)};
    %if 'condition' in obj:
#else
    ${th.append_ws("void*", 63)} ${th.make_pfncb_name(n, tags, obj)};
#endif // ${th.subt(n, tags, obj['condition'])}
    %endif
    %endfor
} ${tbl['type']};

%endfor
///////////////////////////////////////////////////////////////////////////////
/// @brief Container for all callbacks
typedef struct _${n}_callbacks_t
{
%for tbl in th.get_pfncbtables(specs, meta, n, tags):
    ${th.append_ws(tbl['type'], 35)} ${tbl['name']};
%endfor
} ${n}_callbacks_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
%endif # not in ["zet", "zes"]:

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _${N}_API_H