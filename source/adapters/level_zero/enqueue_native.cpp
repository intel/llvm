//===--------- enqueue_native.cpp - LevelZero Adapter ---------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "queue.hpp"

ur_result_t ur_queue_handle_legacy_t_::enqueueNativeCommandExp(
    ur_exp_enqueue_native_command_function_t, void *, uint32_t,
    const ur_mem_handle_t *, const ur_exp_enqueue_native_command_properties_t *,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
