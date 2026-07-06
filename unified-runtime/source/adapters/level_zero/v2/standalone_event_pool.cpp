//===--------- standalone_event_pool.cpp - Level Zero Adapter ------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "standalone_event_pool.hpp"

namespace v2 {
ur_event_handle_t standalone_event_pool::allocate(ur_context_handle_t hContext,
                                                  ur_device_handle_t hDevice,
                                                  event_flags_t flags) {
  const v2::event_descriptor id{hDevice->Id.value(), flags};
  std::lock_guard<ur_mutex> lock(mutex);

  if (!providers.count(id)) {
    auto const queueType = v2::QUEUE_IMMEDIATE;
    auto const platform = hDevice->Platform;

    auto provider =
        createProvider(platform, hContext, queueType, hDevice, flags);
    providers.emplace(id, std::move(provider));
  }

  auto event = raii::ze_event_handle_t{providers.at(id)->allocate().release()};
  return new ur_event_handle_t_(hContext, std::move(event), flags);
}
} // namespace v2
