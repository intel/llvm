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
#include "../ur_interface_loader.hpp"

namespace v2 {

enum queue_type {
  QUEUE_REGULAR,
  QUEUE_IMMEDIATE,
};

class provider_pool {
public:
  provider_pool(ur_context_handle_t, queue_type, event_flags_t flags);

  raii::cache_borrowed_event allocate();
  size_t nfree() const;

private:
  raii::ze_event_pool_handle_t pool;
  std::vector<raii::ze_event_handle_t> freelist;
};

// supplies multi-device events for a given context
class provider_normal : public event_provider {
public:
  provider_normal(ur_context_handle_t context, queue_type qtype,
                  event_flags_t flags)
      : queueType(qtype), urContext(context), flags(flags) {}

  raii::cache_borrowed_event allocate() override;
  event_flags_t eventFlags() const override;

private:
  queue_type queueType;
  ur_context_handle_t urContext;
  event_flags_t flags;

  std::unique_ptr<provider_pool> createProviderPool();
  std::vector<std::unique_ptr<provider_pool>> pools;
};

} // namespace v2
