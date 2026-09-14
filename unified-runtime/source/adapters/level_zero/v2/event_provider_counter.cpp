//===--------- event_provider_counter.cpp - Level Zero Adapter ------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <unified-runtime/ur_api.h>
#include <ze_api.h>

#include "context.hpp"
#include "event_provider.hpp"
#include "event_provider_counter.hpp"
#include "event_provider_normal.hpp"
#include "loader/ze_loader.h"

#include "../common/device.hpp"
#include "../common/platform.hpp"

namespace ur::level_zero::v2 {

provider_counter::provider_counter(ur_platform_handle_t platform,
                                   ur_context_handle_t context,
                                   queue_type queueType,
                                   ur_device_handle_t device,
                                   event_flags_t flags)
    : queueType(queueType), flags(flags),
      useCoreApi(platform->ZeCounterBasedEventsCoreApiSupported),
      zeContext(context->getZeHandle()), zeDevice(device->ZeDevice) {
  assert(flags & EVENT_FLAGS_COUNTER);

  if (useCoreApi) {
    // Probe that the driver actually supports the counter-based event core
    // API. This throws if it doesn't, allowing createProvider() to fall back
    // to the event-pool based provider_normal. Keep the created event around
    // so the probe isn't wasted.
    freelist.emplace_back(createZeEvent());
    return;
  }

  // Fallback for drivers older than Level Zero spec version 1.15: use the
  // deprecated ZEX_counter_based_event extension instead of the core API.
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

static ze_event_counter_based_flags_t createZeFlags(queue_type queueType,
                                                    event_flags_t flags) {
  ze_event_counter_based_flags_t zeFlags =
      ZE_EVENT_COUNTER_BASED_FLAG_HOST_VISIBLE;
  if (flags & EVENT_FLAGS_PROFILING_ENABLED) {
    zeFlags |= ZE_EVENT_COUNTER_BASED_FLAG_DEVICE_TIMESTAMP;
  }

  if (flags & EVENT_FLAGS_IPC) {
    zeFlags |= ZE_EVENT_COUNTER_BASED_FLAG_IPC;
  }

  if (queueType == QUEUE_IMMEDIATE) {
    zeFlags |= ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE;
  }
  // Always set non immediate flag for compatibility with graph record & replay
  zeFlags |= ZE_EVENT_COUNTER_BASED_FLAG_NON_IMMEDIATE;

  return zeFlags;
}

// Deprecated ZEX_counter_based_event extension flags, used only by the
// fallback path for drivers older than Level Zero spec version 1.15.
static zex_counter_based_event_exp_flags_t
createZeFlagsLegacy(queue_type queueType, event_flags_t flags) {
  zex_counter_based_event_exp_flags_t zeFlags =
      ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE;
  if (flags & EVENT_FLAGS_PROFILING_ENABLED) {
    zeFlags |= ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_TIMESTAMP;
  }

  if (flags & EVENT_FLAGS_IPC) {
    zeFlags |= ZEX_COUNTER_BASED_EVENT_FLAG_IPC;
  }

  if (queueType == QUEUE_IMMEDIATE) {
    zeFlags |= ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE;
  }
  // Always set non immediate flag for compatibility with graph record & replay
  zeFlags |= ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE;

  return zeFlags;
}

raii::ze_event_handle_t provider_counter::createZeEvent() {
  ze_event_counter_based_desc_t desc = {};
  desc.stype = ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC;
  desc.flags = createZeFlags(queueType, flags);
  desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;

  uint32_t equivalentFlags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  if (flags & EVENT_FLAGS_PROFILING_ENABLED) {
    equivalentFlags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
  }
  UR_LOG(DEBUG, "ze_event_pool_desc_t flags set to: {}", equivalentFlags);

  ze_event_handle_t handle;

  // TODO: allocate host and device buffers to use here
  ZE2UR_CALL_THROWS(zeEventCounterBasedCreate,
                    (zeContext, zeDevice, &desc, &handle));

  return raii::ze_event_handle_t(handle);
}

raii::ze_event_handle_t provider_counter::createZeEventLegacy() {
  zex_counter_based_event_desc_t desc = {};
  desc.stype = ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC;
  desc.flags = createZeFlagsLegacy(queueType, flags);
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

  return raii::ze_event_handle_t(handle);
}

raii::cache_borrowed_event provider_counter::allocate() {
  if (freelist.empty()) {
    freelist.emplace_back(useCoreApi ? createZeEvent() : createZeEventLegacy());
  }

  auto event = std::move(freelist.back());
  freelist.pop_back();

  return raii::cache_borrowed_event(
      event.release().first,
      [this](ze_event_handle_t handle) { freelist.push_back(handle); });
}

event_flags_t provider_counter::eventFlags() const { return flags; }

std::unique_ptr<event_provider> createProvider(ur_platform_handle_t platform,
                                               ur_context_handle_t context,
                                               queue_type queueType,
                                               ur_device_handle_t device,
                                               event_flags_t flags) {
  // Only try counter-based events if the flag is set
  if (flags & EVENT_FLAGS_COUNTER) {
    // Try to create a counter-based event provider first
    try {
      return std::make_unique<provider_counter>(platform, context, queueType,
                                                device, flags);
    } catch (...) {
      // If the counter-based event core API is not available, fall back to
      // normal provider which supports counter-based events using the
      // event-pool based API
      return std::make_unique<provider_normal>(context, queueType, flags);
    }
  }

  // Counter-based events not requested, use normal events
  return std::make_unique<provider_normal>(context, queueType, flags);
}

} // namespace ur::level_zero::v2
