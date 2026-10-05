//===--------- event_provider_counter.hpp - Level Zero Adapter ------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <stack>

#include <unified-runtime/ur_api.h>
#include <unordered_map>
#include <ur/ur.hpp>
#include <ze_api.h>

#include "common.hpp"
#include "event.hpp"
#include "event_provider.hpp"

#include "../common/device.hpp"

// Deprecated ZEX_counter_based_event extension, kept only as a fallback for
// Level Zero drivers older than spec version 1.15, which don't support the
// core zeEventCounterBasedCreate API yet.
#include <level_zero/driver_experimental/zex_event.h>
#include <level_zero/ze_intel_gpu.h>

namespace ur::level_zero::v2 {

// Function pointer type for the deprecated ZEX_counter_based_event
// extension's zexCounterBasedEventCreate2, used as a fallback on drivers
// older than Level Zero spec version 1.15.
typedef ze_result_t (*zexCounterBasedEventCreate)(
    ze_context_handle_t hContext, ze_device_handle_t hDevice,
    const zex_counter_based_event_desc_t *desc, ze_event_handle_t *phEvent);

class provider_counter : public event_provider {
public:
  provider_counter(ur_platform_handle_t platform, ur_context_handle_t,
                   queue_type, ur_device_handle_t, event_flags_t);

  raii::cache_borrowed_event allocate() override;
  event_flags_t eventFlags() const override;

private:
  // Creates a single counter-based event using the core Level Zero API.
  // Throws if the driver doesn't support it.
  raii::ze_event_handle_t createZeEvent();

  // Creates a single counter-based event using the deprecated
  // ZEX_counter_based_event extension. Used as a fallback for drivers older
  // than Level Zero spec version 1.15. Throws if the driver doesn't support
  // it.
  raii::ze_event_handle_t createZeEventLegacy();

  queue_type queueType;
  event_flags_t flags;

  // True if the driver supports the core counter-based-event API
  // (zeEventCounterBasedCreate), i.e. Level Zero spec version >= 1.15. If
  // false, the deprecated ZEX_counter_based_event extension is used instead.
  bool useCoreApi;

  ze_context_handle_t zeContext;
  ze_device_handle_t zeDevice;

  // Only used by the legacy (extension-based) fallback path.
  ze_context_handle_t translatedContext = nullptr;
  ze_device_handle_t translatedDevice = nullptr;
  zexCounterBasedEventCreate eventCreateFunc = nullptr;

  std::vector<raii::ze_event_handle_t> freelist;
};

// Factory function that creates a counter-based provider with fallback to
// normal provider
std::unique_ptr<event_provider> createProvider(ur_platform_handle_t platform,
                                               ur_context_handle_t context,
                                               queue_type queueType,
                                               ur_device_handle_t device,
                                               event_flags_t flags);

} // namespace ur::level_zero::v2
