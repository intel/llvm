/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file queue_api.cpp
 *
 */

#include "queue_api.hpp"
#include "queue_immediate_in_order.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProperties, ur_queue_handle_t *phQueue) {
  // TODO: For now, always use immediate, in-order
  *phQueue =
      new v2::ur_queue_immediate_in_order_t(hContext, hDevice, pProperties);
  return UR_RESULT_SUCCESS;
}
