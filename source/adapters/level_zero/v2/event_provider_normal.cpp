//===--------- event_provider_normal.cpp - Level Zero Adapter -------------===//
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

#include <memory>

#include "context.hpp"
#include "event_provider.hpp"
#include "event_provider_normal.hpp"

#include "../common/latency_tracker.hpp"

#include "../common.hpp"

namespace v2 {
static constexpr int EVENTS_BURST = 64;

provider_pool::provider_pool(ur_context_handle_t context, queue_type queue,
                             event_flags_t flags) {
  ZeStruct<ze_event_pool_desc_t> desc;
  desc.count = EVENTS_BURST;
  desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

  ze_event_pool_counter_based_exp_desc_t counterBasedExt = {
      ZE_STRUCTURE_TYPE_COUNTER_BASED_EVENT_POOL_EXP_DESC, nullptr, 0};

  if (flags & EVENT_FLAGS_COUNTER) {
    counterBasedExt.flags =
        queue == queue_type::QUEUE_IMMEDIATE
            ? ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_IMMEDIATE
            : ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_NON_IMMEDIATE;
    desc.pNext = &counterBasedExt;
  }

  if (flags & EVENT_FLAGS_PROFILING_ENABLED) {
    desc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
  }

  std::vector<ze_device_handle_t> devices;
  for (auto &d : context->getDevices()) {
    devices.push_back(d->ZeDevice);
  }

  ZE2UR_CALL_THROWS(zeEventPoolCreate,
                    (context->getZeHandle(), &desc, devices.size(),
                     devices.data(), pool.ptr()));

  freelist.resize(EVENTS_BURST);
  for (int i = 0; i < EVENTS_BURST; ++i) {
    ZeStruct<ze_event_desc_t> desc;
    desc.index = i;
    desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    desc.wait = 0;
    ZE2UR_CALL_THROWS(zeEventCreate, (pool.get(), &desc, freelist[i].ptr()));
  }
}

raii::cache_borrowed_event provider_pool::allocate() {
  if (freelist.empty()) {
    return nullptr;
  }
  auto e = std::move(freelist.back());
  freelist.pop_back();
  return raii::cache_borrowed_event(
      e.release().first,
      [this](ze_event_handle_t handle) { freelist.push_back(handle); });
}

size_t provider_pool::nfree() const { return freelist.size(); }

std::unique_ptr<provider_pool> provider_normal::createProviderPool() {
  return std::make_unique<provider_pool>(urContext, queueType, flags);
}

raii::cache_borrowed_event provider_normal::allocate() {
  TRACK_SCOPE_LATENCY("provider_normal::allocate");

  if (pools.empty()) {
    pools.emplace_back(createProviderPool());
  }

  {
    auto &pool = pools.back();
    auto event = pool->allocate();
    if (event) {
      return event;
    }
  }

  std::sort(pools.begin(), pools.end(), [](auto &a, auto &b) {
    return a->nfree() < b->nfree(); // asceding
  });

  {
    auto &pool = pools.back();
    auto event = pool->allocate();
    if (event) {
      return event;
    }
  }

  pools.emplace_back(createProviderPool());

  return allocate();
}

event_flags_t provider_normal::eventFlags() const { return flags; }

} // namespace v2
