//===--------- event_descriptor.hpp - Level Zero Adapter -----------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../device.hpp"
#include "event_provider.hpp"

namespace v2 {

struct event_descriptor {
  DeviceId device;
  event_flags_t flags;

  uint64_t index() const {
    return uint64_t(flags) | (uint64_t(device) << EVENT_FLAGS_USED_BITS);
  }

  bool operator==(const event_descriptor &other) const {
    return device == other.device && flags == other.flags;
  }
};

struct event_descriptor_hash {
  std::size_t operator()(const event_descriptor &key) const {
    return key.index();
  }
};

} // namespace v2
