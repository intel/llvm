//===--- sanitizer_stackdepot.cpp - Stack traces keyed by origin id -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_stackdepot.hpp"

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace llvm::offload::sanitizer {

namespace {

class StackDepot {
public:
  uint32_t Put(uint32_t Id, StackTrace Stack) {
    std::scoped_lock<std::shared_mutex> Guard(_Mutex);
    _Depot[Id] = Stack;
    return Id;
  }

  StackTrace Get(uint32_t Id) {
    std::shared_lock<std::shared_mutex> Guard(_Mutex);
    auto It = _Depot.find(Id);
    if (It != _Depot.end()) {
      return It->second;
    }
    return StackTrace();
  }

private:
  std::shared_mutex _Mutex;
  std::unordered_map<uint32_t, StackTrace> _Depot;
};

// Leaked: reports may still be printed during process exit.
StackDepot &getDepot() {
  static StackDepot *TheDepot = new StackDepot;
  return *TheDepot;
}

} // namespace

void StackDepotPut(uint32_t Id, StackTrace &Stack) {
  getDepot().Put(Id, Stack);
}

StackTrace StackDepotGet(uint32_t Id) { return getDepot().Get(Id); }

} // namespace llvm::offload::sanitizer
