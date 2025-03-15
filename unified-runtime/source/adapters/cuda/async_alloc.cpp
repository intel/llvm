//===--------- async_alloc.cpp - CUDA Adapter -----------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "context.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "queue.hpp"
#include "usm.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMDeviceAllocExp(
    ur_queue_handle_t, ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMHostAllocExp(
    ur_queue_handle_t, ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMSharedAllocExp(
    ur_queue_handle_t, ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMFreeExp(ur_queue_handle_t, ur_usm_pool_handle_t, void *, uint32_t,
                    const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
