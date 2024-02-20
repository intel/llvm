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
 * Copyright (C) 2023-2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.hpp
 * @version v${ver}-r${rev}
 *
 */
#ifndef ${X}_PRINT_HPP
#define ${X}_PRINT_HPP 1

#include "${x}_api.h"
#include <ostream>
#include <bitset>

## Mako helper functions ######################################################
<%def name="member(iname, itype, loop)">
    %if iname == "pNext":
        ${x}::details::printStruct(os, ${caller.body()});
    %elif th.type_traits.is_flags(itype):
        ${x}::details::printFlag<${th.type_traits.get_flag_type(itype)}>(os, ${caller.body()});
    %elif not loop and th.type_traits.is_pointer(itype):
        ${x}::details::printPtr(os, ${caller.body()});
    %elif loop and th.type_traits.is_pointer_to_pointer(itype):
        ${x}::details::printPtr(os, ${caller.body()});
    %elif th.type_traits.is_handle(itype):
        ${x}::details::printPtr(os, ${caller.body()});
    %elif iname and iname.startswith("pfn"):
        os << reinterpret_cast<void*>(${caller.body()});
    %else:
        os << ${caller.body()};
    %endif
</%def>

<%
def findUnionTag(_union):
    tag = [_obj for _s in specs for _obj in _s['objects'] if _obj['name'] == _union['tag']]
    return tag[0] if len(tag) > 0 else None

def findMemberType(_item):
    query = [_o for _s in specs for _o in _s['objects'] if _o['name'] == _item['type']]
    return query[0] if len(query) > 0 else None
%>

<%def name="line(item, n, params, params_dict)">
    <%
        iname = th._get_param_name(n, tags, item)
        prefix = "p" if params else ""
        pname = prefix + iname
        itype = th._get_type_name(n, tags, obj, item)
        access = "->" if params else "."
        deref = "*" if params else ""
        typename = th.param_traits.typename(item)
        if typename is not None:
            typename_size = th.param_traits.typename_size(item)
            underlying_type = params_dict[typename]
    %>
    %if n != 0:
        os << ", ";
    %endif
## can't iterate over 'void *'...
    %if th.param_traits.is_range(item) and "void*" not in itype:
        os << ".${iname} = {";
        for (size_t i = ${th.param_traits.range_start(item)}; ${deref}(params${access}${pname}) != NULL && i < ${deref}params${access}${prefix + th.param_traits.range_end(item)}; ++i) {
            if (i != 0) {
                os << ", ";
            }
            <%call expr="member(iname, itype, True)">
                (${deref}(params${access}${pname}))[i]
            </%call>
        }
        os << "}";
    %elif findMemberType(item) is not None and findMemberType(item)['type'] == "union":
        os << ".${iname} = ";
        ${x}::details::printUnion(os, ${deref}(params${access}${item['name']}), params${access}${th.param_traits.tagged_member(item)});
    %elif th.type_traits.is_array(item['type']):
        os << ".${iname} = {";
        for(auto i = 0; i < ${th.type_traits.get_array_length(item['type'])}; i++){
            if(i != 0){
                os << ", ";
            }
            <%call expr="member(iname, itype, True)">
                ${deref}(params${access}${item['name']}[i])
            </%call>
        }
        os << "}";
    %elif typename is not None:
        os << ".${iname} = ";
        ${x}::details::printTagged(os, ${deref}(params${access}${pname}), ${deref}(params${access}${prefix}${typename}), ${deref}(params${access}${prefix}${typename_size}));
    %else:
        os << ".${iname} = ";
        <%call expr="member(iname, itype, False)">
            ${deref}(params${access}${pname})
        </%call>
    %endif
</%def>

## API functions declarations #################################################
namespace ${x}::details {
template <typename T> struct is_handle : std::false_type {};
%for spec in specs:
%for obj in spec['objects']:
%if re.match(r"handle", obj['type']):
template <> struct is_handle<${th.make_type_name(n, tags, obj)}> : std::true_type {};
%endif
%endfor
%endfor
template <typename T>
inline constexpr bool is_handle_v = is_handle<T>::value;
template <typename T> inline ${x}_result_t printPtr(std::ostream &os, const T *ptr);
template <typename T> inline ${x}_result_t printFlag(std::ostream &os, uint32_t flag);
template <typename T> inline ${x}_result_t printTagged(std::ostream &os, const void *ptr, T value, size_t size);

%for spec in specs:
%for obj in spec['objects']:
## ENUM #######################################################################
%if re.match(r"enum", obj['type']):
    %if obj.get('typed_etors', False) is True:
    template <> inline ${x}_result_t printTagged(std::ostream &os, const void *ptr, ${th.make_enum_name(n, tags, obj)} value, size_t size);
    %elif "structure_type" in obj['name']:
    inline ${x}_result_t printStruct(std::ostream &os, const void *ptr);
    %endif
%endif

## UNION ######################################################################
%if re.match(r"union", obj['type']) and obj['name']:
    <% tag = [_obj for _s in specs for _obj in _s['objects'] if _obj['name'] == obj['tag']][0] %>
    inline ${x}_result_t printUnion(
        std::ostream &os,
        const ${obj['type']} ${th.make_type_name(n, tags, obj)} params,
        const ${tag['type']} ${th.make_type_name(n, tags, tag)} tag
    );
%endif

## FLAG #######################################################################
%if th.type_traits.is_flags(obj['name']):
    template<> inline ${x}_result_t printFlag<${th.make_enum_name(n, tags, obj)}>(std::ostream &os, uint32_t flag);
%endif
%endfor # obj in spec['objects']
%endfor
} // namespace ${x}::details

%for spec in specs:
%for obj in spec['objects']:
%if re.match(r"enum", obj['type']):
    inline std::ostream &operator<<(std::ostream &os, enum ${th.make_enum_name(n, tags, obj)} value);
%elif re.match(r"struct", obj['type']):
    inline std::ostream &operator<<(std::ostream &os, [[maybe_unused]] const ${obj['type']} ${th.make_type_name(n, tags, obj)} params);
%endif
%endfor # obj in spec['objects']
%endfor

## API functions definitions ##################################################
%for spec in specs:
%for obj in spec['objects']:
## ENUM #######################################################################
%if re.match(r"enum", obj['type']):
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Print operator for the ${th.make_enum_name(n, tags, obj)} type
    /// @returns
    ///     std::ostream &
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
                %>case ${ename}:
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
    %if obj.get('typed_etors', False) is True:
    namespace ${x}::details {
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Print ${th.make_enum_name(n, tags, obj)} enum value
    template <>
    inline ${x}_result_t printTagged(std::ostream &os, const void *ptr, ${th.make_enum_name(n, tags, obj)} value, size_t size) {
        if (ptr == NULL) {
            return printPtr(os, ptr);
        }

        switch (value) {
            %for n, item in enumerate(obj['etors']):
                <%
                ename = th.make_etor_name(n, tags, obj['name'], item['name'])
                vtype = th.etor_get_associated_type(n, tags, item)
                %>case ${ename}: {
                    %if th.value_traits.is_array(vtype):
                    <% atype = th.value_traits.get_array_name(vtype) %>
                    %if 'void' in atype:
                    const ${atype} const *tptr = (const ${atype} const*)ptr;
                    %else:
                    const ${atype} *tptr = (const ${atype} *)ptr;
                    %endif
                        %if "char" in atype: ## print char* arrays as simple NULL-terminated strings
                            printPtr(os, tptr);
                        %else:
                            os << "{";
                            size_t nelems = size / sizeof(${atype});
                            for (size_t i = 0; i < nelems; ++i) {
                                if (i != 0) {
                                    os << ", ";
                                }
                                <%call expr="member(tptr, atype, True)">
                                    tptr[i]
                                </%call>
                            }
                            os << "}";
                        %endif
                    %else:
                    %if 'void' in vtype:
                    const ${vtype} const *tptr = (const ${vtype} const *)ptr;
                    %else:
                    const ${vtype} *tptr = (const ${vtype} *)ptr;
                    %endif
                    if (sizeof(${vtype}) > size) {
                        os << "invalid size (is: " << size << ", expected: >=" << sizeof(${vtype}) << ")";
                        return ${X}_RESULT_ERROR_INVALID_SIZE;
                    }
                    os << (const void *)(tptr) << " (";
                    <%call expr="member(tptr, vtype, False)">
                        *tptr
                    </%call>
                    os << ")";
                    %endif
                } break;
            %endfor
                default:
                    os << "unknown enumerator";
                    return ${X}_RESULT_ERROR_INVALID_ENUMERATION;
        }
        return ${X}_RESULT_SUCCESS;
    }
    } // namespace ${x}::details

    %elif "structure_type" in obj['name']:
    namespace ${x}::details {
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Print ${th.make_enum_name(n, tags, obj)} struct
    inline ${x}_result_t printStruct(std::ostream &os, const void *ptr) {
        if (ptr == NULL) {
            return printPtr(os, ptr);
        }

        ## structure type enum value must be first
        const enum ${th.make_enum_name(n, tags, obj)} *value = (const enum ${th.make_enum_name(n, tags, obj)} *)ptr;
        switch (*value) {
            %for n, item in enumerate(obj['etors']):
                <%
                ename = th.make_etor_name(n, tags, obj['name'], item['name'])
                %>
                case ${ename}: {
                    const ${th.subt(n, tags, item['desc'])} *pstruct = (const ${th.subt(n, tags, item['desc'])} *)ptr;
                    printPtr(os, pstruct);
                } break;
            %endfor
                default:
                    os << "unknown enumerator";
                    return ${X}_RESULT_ERROR_INVALID_ENUMERATION;
        }
        return ${X}_RESULT_SUCCESS;
    }
    } // namespace ${x}::details
    %endif
    %if th.type_traits.is_flags(obj['name']):

    namespace ${x}::details {
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Print ${th.make_enum_name(n, tags, obj)} flag
    template<>
    inline ${x}_result_t printFlag<${th.make_enum_name(n, tags, obj)}>(std::ostream &os, uint32_t flag) {
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
        return ${X}_RESULT_SUCCESS;
    }
    } // namespace ${x}::details
    %endif
## STRUCT #####################################################################
%elif re.match(r"struct", obj['type']):
///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ${th.make_type_name(n, tags, obj)} type
/// @returns
///     std::ostream &
inline std::ostream &operator<<(std::ostream &os, const ${obj['type']} ${th.make_type_name(n, tags, obj)} params) {
    os << "(${obj['type']} ${th.make_type_name(n, tags, obj)}){";
    <%
        mlist = obj['members']
        params_dict = dict()
        for item in mlist:
            iname = th._get_param_name(n, tags, item)
            itype = th._get_type_name(n, tags, obj, item)
            params_dict[iname] = itype
    %>
    %for n, item in enumerate(mlist):
        ${line(item, n, False, params_dict)}
    %endfor
    os << "}";
    return os;
}
## UNION ######################################################################
%elif re.match(r"union", obj['type']) and obj['name']:
namespace ${x}::details {
<% tag = findUnionTag(obj) %>
    ///////////////////////////////////////////////////////////////////////////////
    // @brief Print ${th.make_type_name(n, tags, obj)} union
    inline ${x}_result_t printUnion(
    std::ostream &os,
    const ${obj['type']} ${th.make_type_name(n, tags, obj)} params,
    const ${tag['type']} ${th.make_type_name(n, tags, tag)} tag
){
    os << "(${obj['type']} ${th.make_type_name(n, tags, obj)}){";
<%
params_dict = dict()
for item in obj['members']:
    iname = th._get_param_name(n, tags, item)
    itype = th._get_type_name(n, tags, obj, item)
    params_dict[iname] = itype
%>
    switch(tag){
%for mem in obj['members']:
    case ${th.subt(n, tags, mem['tag'])}:
        ${line(mem, 0, False, params_dict)}
        break;
%endfor
    default:
        os << "<unknown>";
        return ${X}_RESULT_ERROR_INVALID_ENUMERATION;
    }
    os << "}";
    return ${X}_RESULT_SUCCESS;
}
} // namespace ${x}::details
%endif
%endfor # obj in spec['objects']
%endfor

%for tbl in th.get_pfncbtables(specs, meta, n, tags):
%for obj in tbl['functions']:

///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the ${th.make_pfncb_param_type(n, tags, obj)} type
/// @returns
///     std::ostream &
inline std::ostream &operator<<(std::ostream &os, [[maybe_unused]] const struct ${th.make_pfncb_param_type(n, tags, obj)} *params) {
    <%
        params_dict = dict()
        for item in obj['params']:
            iname = th._get_param_name(n, tags, item)
            itype = th._get_type_name(n, tags, obj, item)
            params_dict[iname] = itype
    %>
    %for n, item in enumerate(obj['params']):
        ${line(item, n, True, params_dict)}
    %endfor
    return os;
}

%endfor
%endfor

namespace ${x}::details {
///////////////////////////////////////////////////////////////////////////////
// @brief Print pointer value
template <typename T> inline ${x}_result_t printPtr(std::ostream &os, const T *ptr) {
    if (ptr == nullptr) {
        os << "nullptr";
    } else if constexpr (std::is_pointer_v<T>) {
        os << (const void *)(ptr) << " (";
        printPtr(os, *ptr);
        os << ")";
    } else if constexpr (std::is_void_v<T> || is_handle_v<T *>) {
        os << (const void *)ptr;
    } else if constexpr (std::is_same_v<std::remove_cv_t< T >, char>) {
        os << (const void *)(ptr) << " (";
        os << ptr;
        os << ")";
    } else {
        os << (const void *)(ptr) << " (";
        os << *ptr;
        os << ")";
    }

    return ${X}_RESULT_SUCCESS;
}
} // namespace ${x}::details

namespace ${x}::extras {
///////////////////////////////////////////////////////////////////////////////
/// @brief Print function parameters
/// @returns
///     - ::${X}_RESULT_SUCCESS
///     - ::${X}_RESULT_ERROR_INVALID_ENUMERATION
///     - ::${X}_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == params`
inline ${x}_result_t ${X}_APICALL printFunctionParams(std::ostream &os, ur_function_t function, const void *params) {
    if (!params) {
        return ${X}_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    switch(function) {
    %for tbl in th.get_pfncbtables(specs, meta, n, tags):
    %for obj in tbl['functions']:
        case ${th.make_func_etor(n, tags, obj)}: {
            os << (const struct ${th.make_pfncb_param_type(n, tags, obj)} *)params;
        } break;
    %endfor
    %endfor
        default: return ${X}_RESULT_ERROR_INVALID_ENUMERATION;
    }
    return ${X}_RESULT_SUCCESS;
}
} // namespace ${x}::extras

#endif /* ${X}_PRINT_HPP */
