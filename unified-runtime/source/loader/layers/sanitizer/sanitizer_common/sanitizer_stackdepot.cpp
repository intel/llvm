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

#pragma once

#include "sanitizer_stackdepot.hpp"
#include <atomic>
#include <unordered_map>

namespace ur_sanitizer_layer {

class StackDepot {
public:
  uint32_t Put(StackTrace Stack) {
    uint32_t Id = _NextId.fetch_add(1);
    _Depot[Id] = Stack;
    return Id;
  }

  StackTrace Get(uint32_t Id) {
    auto It = _Depot.find(Id);
    if (It != _Depot.end()) {
      return It->second;
    }
    return StackTrace();
  }

private:
  std::atomic_uint32_t _NextId{1};
  std::unordered_map<uint32_t, StackTrace> _Depot;
};

static StackDepot theDepot;

uint32_t StackDepotPut(StackTrace &Stack) { return theDepot.Put(Stack); }

StackTrace StackDepotGet(uint32_t Id) { return theDepot.Get(Id); }

} // namespace ur_sanitizer_layer
