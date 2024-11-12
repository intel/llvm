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
#include "program.hpp"
#include <array>
#include <ur_api.h>
#include <utility>

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

  ur_kernel_handle_t_(ur_program_handle_t hProgram, const char *name,
                      nativecpu_task_t subhandler)
      : hProgram(hProgram), _name{name}, _subhandler{std::move(subhandler)} {}

  ur_kernel_handle_t_(const ur_kernel_handle_t_ &other)
      : hProgram(other.hProgram), _name(other._name),
        _subhandler(other._subhandler), _args(other._args),
        _localArgInfo(other._localArgInfo), _localMemPool(other._localMemPool),
        _localMemPoolSize(other._localMemPoolSize),
        ReqdWGSize(other.ReqdWGSize) {
    incrementReferenceCount();
  }

  ~ur_kernel_handle_t_() {
    if (decrementReferenceCount() == 0) {
      free(_localMemPool);
    }
  }
  ur_kernel_handle_t_(ur_program_handle_t hProgram, const char *name,
                      nativecpu_task_t subhandler,
                      std::optional<native_cpu::WGSize_t> ReqdWGSize,
                      std::optional<native_cpu::WGSize_t> MaxWGSize,
                      std::optional<uint64_t> MaxLinearWGSize)
      : hProgram(hProgram), _name{name}, _subhandler{std::move(subhandler)},
        ReqdWGSize(ReqdWGSize), MaxWGSize(MaxWGSize),
        MaxLinearWGSize(MaxLinearWGSize) {}

  ur_program_handle_t hProgram;
  std::string _name;
  nativecpu_task_t _subhandler;
  std::vector<native_cpu::NativeCPUArgDesc> _args;
  std::vector<local_arg_info_t> _localArgInfo;

  std::optional<native_cpu::WGSize_t> getReqdWGSize() const {
    return ReqdWGSize;
  }

  std::optional<native_cpu::WGSize_t> getMaxWGSize() const { return MaxWGSize; }

  std::optional<uint64_t> getMaxLinearWGSize() const { return MaxLinearWGSize; }

  void updateMemPool(size_t numParallelThreads) {
    // compute requested size.
    size_t reqSize = 0;
    for (auto &entry : _localArgInfo) {
      reqSize += entry.argSize * numParallelThreads;
    }
    if (reqSize == 0 || reqSize == _localMemPoolSize) {
      return;
    }
    // realloc handles nullptr case
    _localMemPool = (char *)realloc(_localMemPool, reqSize);
    _localMemPoolSize = reqSize;
  }

  // To be called before executing a work group
  void handleLocalArgs(size_t numParallelThread, size_t threadId) {
    // For each local argument we have size*numthreads
    size_t offset = 0;
    for (auto &entry : _localArgInfo) {
      _args[entry.argIndex].MPtr =
          _localMemPool + offset + (entry.argSize * threadId);
      // update offset in the memory pool
      offset += entry.argSize * numParallelThread;
    }
  }

private:
  char *_localMemPool = nullptr;
  size_t _localMemPoolSize = 0;
  std::optional<native_cpu::WGSize_t> ReqdWGSize = std::nullopt;
  std::optional<native_cpu::WGSize_t> MaxWGSize = std::nullopt;
  std::optional<uint64_t> MaxLinearWGSize = std::nullopt;
};
