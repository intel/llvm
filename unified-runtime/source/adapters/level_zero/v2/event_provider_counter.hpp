//===--------- event_provider_counter.hpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <stack>

#include <unordered_map>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>

#include "common.hpp"
#include "event.hpp"
#include "event_provider.hpp"

#include "../device.hpp"

#include <level_zero/driver_experimental/zex_event.h>
#include <level_zero/ze_intel_gpu.h>

namespace v2 {

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
  queue_type queueType;
  event_flags_t flags;

  ze_context_handle_t translatedContext;
  ze_device_handle_t translatedDevice;

  zexCounterBasedEventCreate eventCreateFunc;

  std::vector<raii::ze_event_handle_t> freelist;
};

} // namespace v2
