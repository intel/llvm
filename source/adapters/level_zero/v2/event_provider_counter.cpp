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
                                   ur_device_handle_t device)
    : urDevice(device) {
  ZE2UR_CALL_THROWS(zeDriverGetExtensionFunctionAddress,
                    (platform->ZeDriver, "zexCounterBasedEventCreate",
                     (void **)&this->eventCreateFunc));
  ZE2UR_CALL_THROWS(zelLoaderTranslateHandle,
                    (ZEL_HANDLE_CONTEXT, context->getZeHandle(),
                     (void **)&translatedContext));
  ZE2UR_CALL_THROWS(
      zelLoaderTranslateHandle,
      (ZEL_HANDLE_DEVICE, device->ZeDevice, (void **)&translatedDevice));
}

event_allocation provider_counter::allocate() {
  if (freelist.empty()) {
    ZeStruct<ze_event_desc_t> desc;
    desc.index = 0;
    desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    desc.wait = 0;
    ze_event_handle_t handle;

    // TODO: allocate host and device buffers to use here
    ZE2UR_CALL_THROWS(eventCreateFunc, (translatedContext, translatedDevice,
                                        nullptr, nullptr, 0, &desc, &handle));

    freelist.emplace_back(handle);
  }

  auto event = std::move(freelist.back());
  freelist.pop_back();

  return {event_type::EVENT_COUNTER,
          raii::cache_borrowed_event(event.release(),
                                     [this](ze_event_handle_t handle) {
                                       freelist.push_back(handle);
                                     })};
}

ur_device_handle_t provider_counter::device() { return urDevice; }

} // namespace v2
