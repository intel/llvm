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
#include "queue_handle.hpp"
#include "queue_immediate_in_order.hpp"
#include "queue_immediate_out_of_order.hpp"

namespace v2 {

using queue_group_type = ur_device_handle_t_::queue_group_info_t::type;

static uint32_t getZeOrdinal(ur_device_handle_t hDevice) {
  return hDevice->QueueGroup[queue_group_type::Compute].ZeOrdinal;
}

static std::optional<int32_t> getZeIndex(const ur_queue_properties_t *pProps) {
  if (pProps && pProps->pNext) {
    const ur_base_properties_t *extendedDesc =
        reinterpret_cast<const ur_base_properties_t *>(pProps->pNext);
    if (extendedDesc->stype == UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES) {
      const ur_queue_index_properties_t *indexProperties =
          reinterpret_cast<const ur_queue_index_properties_t *>(extendedDesc);
      return indexProperties->computeIndex;
    }
  }
  return std::nullopt;
}

static ze_command_queue_priority_t getZePriority(ur_queue_flags_t flags) {
  if ((flags & UR_QUEUE_FLAG_PRIORITY_LOW) != 0)
    return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
  if ((flags & UR_QUEUE_FLAG_PRIORITY_HIGH) != 0)
    return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
  return ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
}

static event_flags_t eventFlagsFromQueueFlags(ur_queue_flags_t flags) {
  event_flags_t eventFlags = EVENT_FLAGS_COUNTER;
  if (flags & UR_QUEUE_FLAG_PROFILING_ENABLE)
    eventFlags |= EVENT_FLAGS_PROFILING_ENABLED;
  return eventFlags;
}

} // namespace v2

namespace ur::level_zero {
ur_result_t urQueueCreate(ur_context_handle_t hContext,
                          ur_device_handle_t hDevice,
                          const ur_queue_properties_t *pProperties,
                          ur_queue_handle_t *phQueue) try {
  if (!hContext->isValidDevice(hDevice)) {
    return UR_RESULT_ERROR_INVALID_DEVICE;
  }

  ur_queue_flags_t flags = 0;
  if (pProperties) {
    flags = pProperties->flags;
  }

  auto zeIndex = v2::getZeIndex(pProperties);

  if ((flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0 &&
      !zeIndex.has_value()) {
    *phQueue =
        ur_queue_handle_t_::create<v2::ur_queue_immediate_out_of_order_t>(
            hContext, hDevice, v2::getZeOrdinal(hDevice),
            v2::getZePriority(flags), v2::eventFlagsFromQueueFlags(flags),
            flags);
  } else {
    *phQueue = ur_queue_handle_t_::create<v2::ur_queue_immediate_in_order_t>(
        hContext, hDevice, v2::getZeOrdinal(hDevice), v2::getZePriority(flags),
        zeIndex, v2::eventFlagsFromQueueFlags(flags), flags);
  }

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

  auto commandListHandle = v2::raii::command_list_unique_handle(
      reinterpret_cast<ze_command_list_handle_t>(hNativeQueue),
      [ownNativeHandle](ze_command_list_handle_t hZeCommandList) {
        if (ownNativeHandle) {
          if (checkL0LoaderTeardown()) {
            ZE_CALL_NOCHECK(zeCommandListDestroy, (hZeCommandList));
          }
        }
      });

  *phQueue = ur_queue_handle_t_::create<v2::ur_queue_immediate_in_order_t>(
      hContext, hDevice, std::move(commandListHandle),
      v2::eventFlagsFromQueueFlags(flags), flags);

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}
} // namespace ur::level_zero
