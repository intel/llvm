//===--------- event_provider_normal.hpp - Level Zero Adapter -------------===//
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

#include "../device.hpp"

namespace v2 {

enum queue_type {
  QUEUE_REGULAR,
  QUEUE_IMMEDIATE,
};

class provider_pool {
public:
  provider_pool(ur_context_handle_t, ur_device_handle_t, event_type,
                queue_type);

  raii::cache_borrowed_event allocate();
  size_t nfree() const;

private:
  raii::ze_event_pool_handle_t pool;
  std::vector<raii::ze_event_handle_t> freelist;
};

class provider_normal : public event_provider {
public:
  provider_normal(ur_context_handle_t context, ur_device_handle_t device,
                  event_type etype, queue_type qtype)
      : producedType(etype), queueType(qtype), urContext(context),
        urDevice(device) {
    urDeviceRetain(device);
  }

  ~provider_normal() override { urDeviceRelease(urDevice); }

  event_allocation allocate() override;
  ur_device_handle_t device() override;

private:
  event_type producedType;
  queue_type queueType;
  ur_context_handle_t urContext;
  ur_device_handle_t urDevice;

  std::unique_ptr<provider_pool> createProviderPool();
  std::vector<std::unique_ptr<provider_pool>> pools;
};

} // namespace v2
