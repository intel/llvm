//===-------------- memory.hpp - Native CPU Adapter -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>

struct ur_mem_handle_t_ {
  ur_mem_handle_t_(size_t Size)
      : _mem{static_cast<char *>(malloc(Size))}, _ownsMem{true}, _refCount{1} {}

  ur_mem_handle_t_(void *HostPtr, size_t Size)
      : _mem{static_cast<char *>(malloc(Size))}, _ownsMem{true}, _refCount{1} {
    memcpy(_mem, HostPtr, Size);
  }

  ur_mem_handle_t_(void *HostPtr)
      : _mem{static_cast<char *>(HostPtr)}, _ownsMem{false}, _refCount{1} {}

  ~ur_mem_handle_t_() {
    if (_ownsMem) {
      free(_mem);
    }
  }

  void decrementRefCount() noexcept { _refCount--; }

  char *_mem;
  bool _ownsMem;
  std::atomic_uint32_t _refCount;
};
