//===--------- event_provider_counter.cpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <ur_api.h>
#include <ze_api.h>

#include "context.hpp"
#include "event_provider.hpp"
#include "event_provider_counter.hpp"
#include "loader/ze_loader.h"

#include "../device.hpp"
#include "../platform.hpp"

namespace v2 {

provider_counter::provider_counter(ur_platform_handle_t platform,
                                   ur_context_handle_t context,
                                   queue_type queueType,
                                   ur_device_handle_t device,
                                   event_flags_t flags)
    : queueType(queueType), flags(flags) {
  assert(flags & EVENT_FLAGS_COUNTER);

  ZE2UR_CALL_THROWS(zeDriverGetExtensionFunctionAddress,
                    (platform->ZeDriver, "zexCounterBasedEventCreate2",
                     (void **)&this->eventCreateFunc));
  ZE2UR_CALL_THROWS(zelLoaderTranslateHandle,
                    (ZEL_HANDLE_CONTEXT, context->getZeHandle(),
                     (void **)&translatedContext));
  ZE2UR_CALL_THROWS(
      zelLoaderTranslateHandle,
      (ZEL_HANDLE_DEVICE, device->ZeDevice, (void **)&translatedDevice));
}

static zex_counter_based_event_exp_flags_t createZeFlags(queue_type queueType,
                                                         event_flags_t flags) {
  zex_counter_based_event_exp_flags_t zeFlags =
      ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE;
  if (flags & EVENT_FLAGS_PROFILING_ENABLED) {
    zeFlags |= ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_TIMESTAMP;
  }

  if (queueType == QUEUE_IMMEDIATE) {
    zeFlags |= ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE;
  } else {
    zeFlags |= ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE;
  }

  return zeFlags;
}

raii::cache_borrowed_event provider_counter::allocate() {
  if (freelist.empty()) {
    zex_counter_based_event_desc_t desc = {};
    desc.stype = ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC;
    desc.flags = createZeFlags(queueType, flags);
    desc.signalScope = ZE_EVENT_SCOPE_FLAG_HOST;

    uint32_t equivalentFlags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    if (flags & EVENT_FLAGS_PROFILING_ENABLED) {
      equivalentFlags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    }
    UR_LOG(DEBUG, "ze_event_pool_desc_t flags set to: {}", equivalentFlags);

    ze_event_handle_t handle;

    // TODO: allocate host and device buffers to use here
    ZE2UR_CALL_THROWS(eventCreateFunc,
                      (translatedContext, translatedDevice, &desc, &handle));

    freelist.emplace_back(handle);
  }

  auto event = std::move(freelist.back());
  freelist.pop_back();

  return raii::cache_borrowed_event(
      event.release().first,
      [this](ze_event_handle_t handle) { freelist.push_back(handle); });
}

event_flags_t provider_counter::eventFlags() const { return flags; }

} // namespace v2
