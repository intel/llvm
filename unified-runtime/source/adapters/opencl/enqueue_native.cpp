//===--------- enqueue_native.cpp - OpenCL Adapter ------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueNativeCommandExp(
    ur_queue_handle_t, ur_exp_enqueue_native_command_function_t, void *,
    uint32_t, const ur_mem_handle_t *,
    const ur_exp_enqueue_native_command_properties_t *, uint32_t,
    const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
