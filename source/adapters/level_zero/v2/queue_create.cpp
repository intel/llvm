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
                          ur_queue_handle_t *phQueue) try {
  if (!hContext->isValidDevice(hDevice)) {
    return UR_RESULT_ERROR_INVALID_DEVICE;
  }

  // TODO: For now, always use immediate, in-order
  *phQueue =
      new v2::ur_queue_immediate_in_order_t(hContext, hDevice, pProperties);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) try {
  // TODO: For now, always assume it's immediate, in-order

  bool ownNativeHandle = pProperties ? pProperties->isNativeHandleOwned : false;
  ur_queue_flags_t flags = 0;

  if (pProperties) {
    void *pNext = pProperties->pNext;
    while (pNext) {
      const ur_base_properties_t *extendedProperties =
          reinterpret_cast<const ur_base_properties_t *>(pNext);
      if (extendedProperties->stype == UR_STRUCTURE_TYPE_QUEUE_PROPERTIES) {
        const ur_queue_properties_t *pUrProperties =
            reinterpret_cast<const ur_queue_properties_t *>(extendedProperties);
        flags = pUrProperties->flags;
      }
      pNext = extendedProperties->pNext;
    }
  }

  *phQueue = new v2::ur_queue_immediate_in_order_t(
      hContext, hDevice, hNativeQueue, flags, ownNativeHandle);

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}
} // namespace ur::level_zero
