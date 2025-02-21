//===--------- event_pool_cache.cpp - Level Zero Adapter ------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "event_pool_cache.hpp"
#include "../device.hpp"
#include "../platform.hpp"

namespace v2 {

event_pool_cache::event_pool_cache(ur_context_handle_t hContext,
                                   size_t max_devices,
                                   ProviderCreateFunc ProviderCreate)
    : hContext(hContext), providerCreate(ProviderCreate) {
  pools.resize(max_devices * (1ULL << EVENT_FLAGS_USED_BITS));
}

raii::cache_borrowed_event_pool event_pool_cache::borrow(DeviceId id,
                                                         event_flags_t flags) {
  std::unique_lock<ur_mutex> Lock(mutex);

  event_descriptor event_desc{id, flags};

  if (event_desc.index() >= pools.size()) {
    return nullptr;
  }

  auto &vec = pools[event_desc.index()];
  if (vec.empty()) {
    vec.emplace_back(
        std::make_unique<event_pool>(hContext, providerCreate(id, flags)));
  }

  auto pool = vec.back().release();
  vec.pop_back();

  return raii::cache_borrowed_event_pool(
      pool, [this, id, flags](event_pool *pool) {
        std::unique_lock<ur_mutex> Lock(mutex);
        pools[event_descriptor{id, flags}.index()].emplace_back(pool);
      });
}

} // namespace v2
