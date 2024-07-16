//===--------- command_list_cache.hpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <memory>
#include <mutex>
#include <stack>

#include <unordered_map>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <vector>
#include <ze_api.h>

namespace v2 {

enum event_type { EVENT_REGULAR, EVENT_COUNTER };

using event_borrowed =
    std::unique_ptr<_ze_event_handle_t, std::function<void(ze_event_handle_t)>>;

struct event_allocation {
  event_type type;
  event_borrowed borrow;
};

class event_provider {
public:
  virtual ~event_provider() = default;
  virtual event_allocation allocate() = 0;
  virtual ur_device_handle_t device() = 0;
};

} // namespace v2
