/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_stackdepot.cpp
 *
 */

#include "sanitizer_stackdepot.hpp"

#include <atomic>
#include <unordered_map>

namespace ur_sanitizer_layer {

class StackDepot {
public:
  uint32_t Put(uint32_t Id, StackTrace Stack) {
    std::scoped_lock<ur_shared_mutex> Guard(_Mutex);
    _Depot[Id] = Stack;
    return Id;
  }

  StackTrace Get(uint32_t Id) {
    std::shared_lock<ur_shared_mutex> Guard(_Mutex);
    auto It = _Depot.find(Id);
    if (It != _Depot.end()) {
      return It->second;
    }
    return StackTrace();
  }

private:
  ur_shared_mutex _Mutex;
  std::unordered_map<uint32_t, StackTrace> _Depot;
};

static StackDepot TheDepot;

void StackDepotPut(uint32_t Id, StackTrace &Stack) { TheDepot.Put(Id, Stack); }

StackTrace StackDepotGet(uint32_t Id) { return TheDepot.Get(Id); }

} // namespace ur_sanitizer_layer
