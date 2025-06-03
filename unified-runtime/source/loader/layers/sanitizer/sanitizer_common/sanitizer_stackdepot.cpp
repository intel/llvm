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

uint32_t StackDepotPut(StackTrace &Stack, HeapType Type) {
  switch (Type) {
  case HeapType::DeviceUSM: {
    static StackDepot TheDepot;
    return TheDepot.Put(Stack);
  }
  case HeapType::HostUSM: {
    static StackDepot TheDepot;
    return TheDepot.Put(Stack);
  }
  case HeapType::SharedUSM: {
    static StackDepot TheDepot;
    return TheDepot.Put(Stack);
  }
  case HeapType::Local: {
    static StackDepot TheDepot;
    return TheDepot.Put(Stack);
  }
  default:
    assert(false && "Unknown heap type");
    return 0;
  }
}

StackTrace StackDepotGet(uint32_t Id, HeapType Type) {
  switch (Type) {
  case HeapType::DeviceUSM: {
    static StackDepot TheDepot;
    return TheDepot.Get(Id);
  }
  case HeapType::HostUSM: {
    static StackDepot TheDepot;
    return TheDepot.Get(Id);
  }
  case HeapType::SharedUSM: {
    static StackDepot TheDepot;
    return TheDepot.Get(Id);
  }
  case HeapType::Local: {
    static StackDepot TheDepot;
    return TheDepot.Get(Id);
  }
  default:
    assert(false && "Unknown heap type");
    return StackTrace();
  }
}

} // namespace ur_sanitizer_layer
