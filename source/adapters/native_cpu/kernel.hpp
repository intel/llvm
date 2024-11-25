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
#include <cstring>
#include <ur_api.h>
#include <utility>

using nativecpu_kernel_t = void(void *const *, native_cpu::state *);
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
      : Args(other.Args), hProgram(other.hProgram), _name(other._name),
        _subhandler(other._subhandler), _localArgInfo(other._localArgInfo),
        _localMemPool(other._localMemPool),
        _localMemPoolSize(other._localMemPoolSize),
        ReqdWGSize(other.ReqdWGSize) {
    incrementReferenceCount();
  }

  ~ur_kernel_handle_t_() {
    if (decrementReferenceCount() == 0) {
      free(_localMemPool);
      Args.deallocate();
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

  struct arguments {
    using args_index_t = std::vector<void *>;
    args_index_t Indices;
    std::vector<size_t> ParamSizes;
    std::vector<bool> OwnsMem;
    static constexpr size_t MaxAlign = 16 * sizeof(double);

    /// Add an argument to the kernel.
    /// If the argument existed before, it is replaced.
    /// Otherwise, it is added.
    /// Gaps are filled with empty arguments.
    /// Implicit offset argument is kept at the back of the indices collection.
    void addArg(size_t Index, size_t Size, const void *Arg) {
      if (Index + 1 > Indices.size()) {
        Indices.resize(Index + 1);
        OwnsMem.resize(Index + 1);
        ParamSizes.resize(Index + 1);

        // Update the stored value for the argument
        Indices[Index] = native_cpu::aligned_malloc(MaxAlign, Size);
        OwnsMem[Index] = true;
        ParamSizes[Index] = Size;
      } else {
        if (ParamSizes[Index] != Size) {
          Indices[Index] = realloc(Indices[Index], Size);
          ParamSizes[Index] = Size;
        }
      }
      std::memcpy(Indices[Index], Arg, Size);
    }

    void addPtrArg(size_t Index, void *Arg) {
      if (Index + 1 > Indices.size()) {
        Indices.resize(Index + 1);
        OwnsMem.resize(Index + 1);
        ParamSizes.resize(Index + 1);

        OwnsMem[Index] = false;
        ParamSizes[Index] = sizeof(uint8_t *);
      }
      Indices[Index] = Arg;
    }

    // This is called by the destructor of ur_kernel_handle_t_, since
    // ur_kernel_handle_t_ implements reference counting and we want
    // to deallocate only when the reference count is 0.
    void deallocate() {
      assert(OwnsMem.size() == Indices.size() && "Size mismatch");
      for (size_t Index = 0; Index < Indices.size(); Index++) {
        if (OwnsMem[Index])
          native_cpu::aligned_free(Indices[Index]);
      }
    }

    const args_index_t &getIndices() const noexcept { return Indices; }

  } Args;

  ur_program_handle_t hProgram;
  std::string _name;
  nativecpu_task_t _subhandler;
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
      Args.Indices[entry.argIndex] =
          _localMemPool + offset + (entry.argSize * threadId);
      // update offset in the memory pool
      offset += entry.argSize * numParallelThread;
    }
  }

  const std::vector<void *> &getArgs() const { return Args.getIndices(); }

  void addArg(const void *Ptr, size_t Index, size_t Size) {
    Args.addArg(Index, Size, Ptr);
  }

  void addPtrArg(void *Ptr, size_t Index) { Args.addPtrArg(Index, Ptr); }

private:
  char *_localMemPool = nullptr;
  size_t _localMemPoolSize = 0;
  std::optional<native_cpu::WGSize_t> ReqdWGSize = std::nullopt;
  std::optional<native_cpu::WGSize_t> MaxWGSize = std::nullopt;
  std::optional<uint64_t> MaxLinearWGSize = std::nullopt;
};
