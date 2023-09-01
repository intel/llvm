//===--------- adapters.hpp - Level Zero Adapter --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <mutex>

struct ur_adapter_handle_t_ {
  std::atomic<uint32_t> RefCount = 0;
  std::mutex Mutex;
};

extern ur_adapter_handle_t_ Adapter;
