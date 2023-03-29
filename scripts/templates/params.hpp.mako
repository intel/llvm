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
 * @file ${name}.hpp
 *
 */
#ifndef ${X}_PARAMS_HPP
#define ${X}_PARAMS_HPP 1

#include "${x}_api.h"
#include <ostream>
#include <bitset>

namespace ${x}_params {

template <typename T> inline void serializePtr(std::ostream &os, T *ptr);

<%def name="member(iname, itype, loop)">
    %if iname == "pNext":
        serializeStruct(os, ${caller.body()});
    %elif th.type_traits.is_flags(itype):
        serializeFlag_${itype}(os, ${caller.body()});
    %elif not loop and th.type_traits.is_pointer(itype):
        serializePtr(os, ${caller.body()});
    %elif loop and th.type_traits.is_pointer_to_pointer(itype):
        serializePtr(os, ${caller.body()});
    %elif th.type_traits.is_handle(itype):
        serializePtr(os, ${caller.body()});
    %else:
        os << ${caller.body()};
    %endif
</%def>

<%def name="line(item, n, params)">
    <%
        iname = th._get_param_name(n, tags, item)
        prefix = "p" if params else ""
        pname = prefix + iname
        itype = th._get_type_name(n, tags, obj, item)
        access = "->" if params else "."
        deref = "*" if params else ""
    %>
    %if n != 0:
        os << ", ";
    %endif
    ## can't iterate over 'void *'...
    %if th.param_traits.is_range(item) and "void*" not in itype:
        os << ".${iname} = [";
        for (size_t i = ${th.param_traits.range_start(item)}; ${deref}(params${access}${pname}) != NULL && i < ${deref}params${access}${prefix + th.param_traits.range_end(item)}; ++i) {
            if (i != 0) {
                os << ", ";
            }
            <%call expr="member(iname, itype, True)">
                (${deref}(params${access}${pname}))[i]
            </%call>
        }
        os << "]";
    %else:
        os << ".${iname} = ";
        <%call expr="member(iname, itype, False)">
            ${deref}(params${access}${pname})
        </%call>
    %endif
</%def>

%for spec in specs:
%for obj in spec['objects']:
## ENUM #######################################################################
%if re.match(r"enum", obj['type']):
    %if "api_version" in obj['name']:
    inline std::ostream &operator<<(std::ostream &os, enum ${th.make_enum_name(n, tags, obj)} value) {
        os << UR_MAJOR_VERSION(value) << "." << UR_MINOR_VERSION(value);
        return os;
    }
    %else:
    inline std::ostream &operator<<(std::ostream &os, enum ${th.make_enum_name(n, tags, obj)} value) {
        switch (value) {
            %for n, item in enumerate(obj['etors']):
                <%
                ename = th.make_etor_name(n, tags, obj['name'], item['name'])
                %>
                case ${ename}:
                    os << "${ename}";
                    break;
            %endfor
                default:
                    os << "unknown enumerator";
                    break;
        }
        return os;
    }
    %endif
    %if "structure_type" in obj['name']:
    inline void serializeStruct(std::ostream &os, const void *ptr) {
        if (ptr == NULL) {
            serializePtr(os, ptr);
            return;
        }

        ## structure type enum value must be first
        enum ${th.make_enum_name(n, tags, obj)} *value = (enum ${th.make_enum_name(n, tags, obj)} *)ptr;
        switch (*value) {
            %for n, item in enumerate(obj['etors']):
                <%
                ename = th.make_etor_name(n, tags, obj['name'], item['name'])
                %>
                case ${ename}: {
                    const ${th.subt(n, tags, item['desc'])} *pstruct = (const ${th.subt(n, tags, item['desc'])} *)ptr;
                    serializePtr(os, pstruct);
                } break;
            %endfor
                default:
                    os << "unknown enumerator";
                    break;
        }
    }
    %endif
%if th.type_traits.is_flags(obj['name']):
inline void serializeFlag_${th.make_type_name(n, tags, obj)}(std::ostream &os, ${th.make_type_name(n, tags, obj)} flag) {
    uint32_t val = flag;
    bool first = true;
    %for n, item in enumerate(obj['etors']):
        <%
        ename = th.make_etor_name(n, tags, obj['name'], item['name'])
        %>
        if ((val & ${ename}) == (uint32_t)${ename}) {
            ## toggle the bits to avoid printing overlapping values
            ## instead of e.g., FLAG_FOO | FLAG_BAR | FLAG_ALL, this will just
            ## print FLAG_FOO | FLAG_BAR (or just FLAG_ALL, depending on order).
            val ^= (uint32_t)${ename};
            if (!first) {
                os << " | ";
            } else {
                first = false;
            }
            os << ${ename};
        }
    %endfor
    if (val != 0) {
        std::bitset<32> bits(val);
        if (!first) {
            os << " | ";
        }
        os << "unknown bit flags " << bits;
    } else if (first) {
        os << "0";
    }
}
%endif
## STRUCT/UNION ###############################################################
%elif re.match(r"struct|union", obj['type']):
inline std::ostream &operator<<(std::ostream &os, const ${obj['type']} ${th.make_type_name(n, tags, obj)} params) {
    os << "(${obj['type']} ${th.make_type_name(n, tags, obj)}){";
    <% mlist = obj['members'] %>
    %for n, item in enumerate(mlist):
        ${line(item, n, False)}
    %endfor
    os << "}";
    return os;
}
%endif
%endfor # obj in spec['objects']
%endfor

%for tbl in th.get_pfncbtables(specs, meta, n, tags):
%for obj in tbl['functions']:

inline std::ostream &operator<<(std::ostream &os, const struct ${th.make_pfncb_param_type(n, tags, obj)} *params) {
    %for n, item in enumerate(obj['params']):
        ${line(item, n, True)}
    %endfor
    return os;
}

%endfor
%endfor

## This is needed to avoid dereferencing forward declared handles
// https://devblogs.microsoft.com/oldnewthing/20190710-00/?p=102678
template<typename, typename = void>
constexpr bool is_type_complete_v = false;
template<typename T>
constexpr bool is_type_complete_v<T, std::void_t<decltype(sizeof(T))>> = true;

template <typename T> inline void serializePtr(std::ostream &os, T *ptr) {
    if (ptr == nullptr) {
        os << "nullptr";
    } else if constexpr (std::is_pointer_v<T>) {
        os << (void *)(ptr) << " (";
        serializePtr(os, *ptr);
        os << ")";
    } else if constexpr (std::is_void_v<T> || !is_type_complete_v<T>) {
        os << (void*)ptr;
    } else {
        os << (void *)(ptr) << " (";
        os << *ptr;
        os << ")";
    }
}

inline int serializeFunctionParams(std::ostream &os, uint32_t function, const void *params) {
    switch((enum ${x}_function_t)function) {
    %for tbl in th.get_pfncbtables(specs, meta, n, tags):
    %for obj in tbl['functions']:
        case ${th.make_func_etor(n, tags, obj)}: {
            os << (const struct ${th.make_pfncb_param_type(n, tags, obj)} *)params;
        } break;
    %endfor
    %endfor
        default: return -1;
    }
    return 0;
}

}

#endif /* ${X}_PARAMS_HPP */
