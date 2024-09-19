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

#include "logger/ur_logger.hpp"
#include "queue_api.hpp"
#include "queue_immediate_in_order.hpp"

#include <tuple>
#include <utility>

namespace ur::level_zero {
ur_result_t urQueueCreate(ur_context_handle_t hContext,
                          ur_device_handle_t hDevice,
                          const ur_queue_properties_t *pProperties,
                          ur_queue_handle_t *phQueue) {
  // TODO: For now, always use immediate, in-order
  *phQueue =
      new v2::ur_queue_immediate_in_order_t(hContext, hDevice, pProperties);
  return UR_RESULT_SUCCESS;
}

ur_result_t urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {
  std::ignore = hNativeQueue;
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pProperties;
  std::ignore = phQueue;
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
} // namespace ur::level_zero
