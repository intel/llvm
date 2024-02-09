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
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.cpp
 *
 */

#pragma once

#include <ur_api.h>
#include "utils.hpp"
#include <cstdlib>
#include <string_view>

namespace urinfo {
%for obj in th.extract_objs(specs, r"enum"):
%if obj["name"] == '$x_loader_config_info_t':
inline void printLoaderConfigInfos(${x}_loader_config_handle_t hLoaderConfig, std::string_view prefix = "  ") {
%for etor in obj['etors']:
%if 'REFERENCE_COUNT' not in etor['name']:
    <%etype = th.etor_get_associated_type(n, tags, etor)
    %>std::cout << prefix;
    printLoaderConfigInfo<${etype}>(hLoaderConfig, ${etor['name'].replace('$X', X)});
%endif
%endfor
}
%endif
%if obj["name"] == '$x_adapter_info_t':
inline void printAdapterInfos(${x}_adapter_handle_t hAdapter, std::string_view prefix = "  ") {
%for etor in obj['etors']:
%if 'REFERENCE_COUNT' not in etor['name']:
    <%etype = th.etor_get_associated_type(n, tags, etor)
    %>std::cout << prefix;
    printAdapterInfo<${etype}>(hAdapter, ${etor['name'].replace('$X', X)});
%endif
%endfor
}

%endif
%if obj["name"] == '$x_platform_info_t':
inline void printPlatformInfos(${x}_platform_handle_t hPlatform, std::string_view prefix = "    ") {
%for etor in obj['etors']:
    <%etype = th.etor_get_associated_type(n, tags, etor)
    %>std::cout << prefix;
    printPlatformInfo<${etype}>(hPlatform, ${etor['name'].replace('$X', X)});
%endfor
}

%endif
%if obj['name'] == '$x_device_info_t':
inline void printDeviceInfos(${x}_device_handle_t hDevice, std::string_view prefix = "      ") {
%for etor in obj['etors']:
    <%etype = th.etor_get_associated_type(n, tags, etor)
    %>std::cout << prefix;
%if etor['name'] == '$X_DEVICE_INFO_UUID':
    printDeviceUUID(hDevice, ${etor['name'].replace('$X', X)});
%else:
    printDeviceInfo<${etype}>(hDevice, ${etor['name'].replace('$X', X)});
%endif
%endfor
}
%endif
%endfor
}  // namespace urinfo
