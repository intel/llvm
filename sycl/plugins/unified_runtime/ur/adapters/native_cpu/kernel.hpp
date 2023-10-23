//===--------------- kernel.hpp - Native CPU Adapter ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
#include "nativecpu_state.hpp"
#include <ur_api.h>

namespace native_cpu {

struct NativeCPUArgDesc {
  void *MPtr;

  NativeCPUArgDesc(void *Ptr) : MPtr(Ptr){};
};

} // namespace native_cpu

using nativecpu_kernel_t = void(const native_cpu::NativeCPUArgDesc *,
                                native_cpu::state *);
using nativecpu_ptr_t = nativecpu_kernel_t *;
using nativecpu_task_t = std::function<nativecpu_kernel_t>;

struct local_arg_info_t {
  uint32_t argIndex;
  size_t argSize;
  local_arg_info_t(uint32_t argIndex, size_t argSize)
      : argIndex(argIndex), argSize(argSize) {}
};

struct ur_kernel_handle_t_ : RefCounted {

  ur_kernel_handle_t_(const char *name, nativecpu_task_t subhandler)
      : _name{name}, _subhandler{subhandler} {}

  const char *_name;
  nativecpu_task_t _subhandler;
  std::vector<native_cpu::NativeCPUArgDesc> _args;
  std::vector<local_arg_info_t> _localArgInfo;

  // To be called before enqueing the kernel.
  void handleLocalArgs() {
    updateMemPool();
    size_t offset = 0;
    for (auto &entry : _localArgInfo) {
      _args[entry.argIndex].MPtr =
          reinterpret_cast<char *>(_localMemPool) + offset;
      // update offset in the memory pool
      // Todo: update this offset computation when we have work-group
      // level parallelism.
      offset += entry.argSize;
    }
  }

  ~ur_kernel_handle_t_() {
    if (_localMemPool) {
      free(_localMemPool);
    }
  }

private:
  void updateMemPool() {
    // compute requested size.
    // Todo: currently we execute only one work-group at a time, so for each
    // local arg we can allocate just 1 * argSize local arg. When we implement
    // work-group level parallelism we should allocate N * argSize where N is
    // the number of work groups being executed in parallel (e.g. number of
    // threads in the thread pool).
    size_t reqSize = 0;
    for (auto &entry : _localArgInfo) {
      reqSize += entry.argSize;
    }
    if (reqSize == 0 || reqSize == _localMemPoolSize) {
      return;
    }
    // realloc handles nullptr case
    _localMemPool = realloc(_localMemPool, reqSize);
    _localMemPoolSize = reqSize;
  }
  void *_localMemPool = nullptr;
  size_t _localMemPoolSize = 0;
};
