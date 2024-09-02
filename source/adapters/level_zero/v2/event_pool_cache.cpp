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

event_pool_cache::event_pool_cache(size_t max_devices,
                                   ProviderCreateFunc ProviderCreate)
    : providerCreate(ProviderCreate) {
  pools.resize(max_devices);
}

event_pool_cache::~event_pool_cache() {}

raii::cache_borrowed_event_pool event_pool_cache::borrow(DeviceId id) {
  std::unique_lock<ur_mutex> Lock(mutex);

  if (id >= pools.size()) {
    return nullptr;
  }

  auto &vec = pools[id];
  if (vec.empty()) {
    vec.emplace_back(std::make_unique<event_pool>(providerCreate(id)));
  }

  auto pool = vec.back().release();
  vec.pop_back();

  return raii::cache_borrowed_event_pool(pool, [this](event_pool *pool) {
    std::unique_lock<ur_mutex> Lock(mutex);
    pools[pool->Id()].emplace_back(pool);
  });
}

} // namespace v2
