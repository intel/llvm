//===--------- standalone_event_factory.hpp - Level Zero Adapter
//------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "common.hpp"
#include "event_pool_cache.hpp"
#include "event_provider_counter.hpp"

namespace v2 {
struct standalone_event_factory {
  // Returns a standalone counter-based event, owned by the caller.
  // The event is not managed by the factory after creation.
  ur_event_handle_t create(ur_context_handle_t hContext,
                           ur_device_handle_t hDevice, event_flags_t flags);

private:
  std::unordered_map<event_pool_cache::event_descriptor,
                     std::unique_ptr<v2::event_provider>, event_descriptor_hash>
      providers;
  ur_mutex mutex;
};
} // namespace v2