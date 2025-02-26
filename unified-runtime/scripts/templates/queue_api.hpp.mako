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
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.hpp
 *
 */

// Do not edit. This file is auto generated from a template: scripts/templates/queue_api.hpp.mako

#pragma once

#include <ur_api.h>
#include <ze_api.h>

struct ur_queue_t_ {
    virtual ~ur_queue_t_();

    virtual void deferEventFree(ur_event_handle_t hEvent) = 0;

    %for obj in th.get_queue_related_functions(specs, n, tags):
    %if not 'Release' in obj['name'] and not 'Retain' in obj['name']:
    virtual ${x}_result_t ${th.transform_queue_related_function_name(n, tags, obj, format=["type"])} = 0;
    %endif
    %endfor

    virtual ur_result_t
    enqueueCommandBuffer(ze_command_list_handle_t, ur_event_handle_t *,
    uint32_t, const ur_event_handle_t *) = 0;
};
