//===--------------- kernel.hpp - Native CPU Adapter ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
#include "memory.hpp"
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
        ReqdWGSize(other.ReqdWGSize) {
    takeArgReferences(other);
  }

  ~ur_kernel_handle_t_() {
    removeArgReferences();
    native_cpu::aligned_free(_localMemPool);
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

    arguments() = default;

    arguments(const arguments &Other)
        : Indices(Other.Indices), ParamSizes(Other.ParamSizes),
          OwnsMem(Other.OwnsMem.size(), false) {
      for (size_t Index = 0; Index < Indices.size(); Index++) {
        if (!Other.OwnsMem[Index]) {
          continue;
        }
        addArg(Index, ParamSizes[Index], Indices[Index]);
      }
    }

    arguments(arguments &&Other) : arguments() {
      std::swap(Indices, Other.Indices);
      std::swap(ParamSizes, Other.ParamSizes);
      std::swap(OwnsMem, Other.OwnsMem);
    }

    ~arguments() {
      assert(OwnsMem.size() == Indices.size() && "Size mismatch");
      for (size_t Index = 0; Index < Indices.size(); Index++) {
        if (!OwnsMem[Index]) {
          continue;
        }
        native_cpu::aligned_free(Indices[Index]);
      }
    }

    /// Add an argument to the kernel.
    /// If the argument existed before, it is replaced.
    /// Otherwise, it is added.
    /// Gaps are filled with empty arguments.
    /// Implicit offset argument is kept at the back of the indices collection.
    void addArg(size_t Index, size_t Size, const void *Arg) {
      bool NeedAlloc = true;
      if (Index + 1 > Indices.size()) {
        Indices.resize(Index + 1);
        OwnsMem.resize(Index + 1);
        ParamSizes.resize(Index + 1);
      } else if (OwnsMem[Index]) {
        if (ParamSizes[Index] == Size) {
          NeedAlloc = false;
        } else {
          native_cpu::aligned_free(Indices[Index]);
        }
      }
      if (NeedAlloc) {
        Indices[Index] = native_cpu::aligned_malloc(Size);
        ParamSizes[Index] = Size;
        OwnsMem[Index] = true;
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
    native_cpu::aligned_free(_localMemPool);
    _localMemPool = static_cast<char *>(native_cpu::aligned_malloc(reqSize));
    _localMemPoolSize = reqSize;
  }

  bool hasLocalArgs() const { return !_localArgInfo.empty(); }

  const std::vector<void *> &getArgs() const {
    assert(!hasLocalArgs() && "For kernels with local arguments, thread "
                              "information must be supplied.");
    return Args.getIndices();
  }

  std::vector<void *> getArgs(size_t numThreads, size_t threadId) const {
    auto Result = Args.getIndices();

    // For each local argument we have size*numthreads
    size_t offset = 0;
    for (auto &entry : _localArgInfo) {
      Result[entry.argIndex] =
          _localMemPool + offset + (entry.argSize * threadId);
      // update offset in the memory pool
      offset += entry.argSize * numThreads;
    }

    return Result;
  }

  inline ur_result_t addArg(const void *Ptr, size_t Index, size_t Size) {
    UR_ASSERT(Size, UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
    Args.addArg(Index, Size, Ptr);
    return UR_RESULT_SUCCESS;
  }

  inline ur_result_t addPtrArg(void *Ptr, size_t Index) {
    UR_ASSERT(Ptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
    Args.addPtrArg(Index, Ptr);
    return UR_RESULT_SUCCESS;
  }

  void addArgReference(ur_mem_handle_t Arg) {
    Arg->incrementReferenceCount();
    ReferencedArgs.push_back(Arg);
  }

  inline ur_result_t addMemObjArg(ur_mem_handle_t ArgValue, size_t Index) {
    // Taken from ur/adapters/cuda/kernel.cpp
    // zero-sized buffers are expected to be null.
    if (ArgValue == nullptr) {
      addPtrArg(nullptr, Index);
      return UR_RESULT_SUCCESS;
    }

    addArgReference(ArgValue);
    addPtrArg(ArgValue->_mem, Index);
    return UR_RESULT_SUCCESS;
  }

  inline ur_result_t addLocalArg(size_t Index, size_t Size) {
    // emplace a placeholder kernel arg, gets replaced with a pointer to the
    // memory pool before enqueueing the kernel.
    Args.addPtrArg(Index, nullptr);
    _localArgInfo.emplace_back(Index, Size);
    return UR_RESULT_SUCCESS;
  }

private:
  void removeArgReferences() {
    for (auto arg : ReferencedArgs)
      decrementOrDelete(arg);
  }
  void takeArgReferences(const ur_kernel_handle_t_ &other) {
    for (auto arg : other.ReferencedArgs)
      addArgReference(arg);
  }

private:
  char *_localMemPool = nullptr;
  size_t _localMemPoolSize = 0;
  std::optional<native_cpu::WGSize_t> ReqdWGSize = std::nullopt;
  std::optional<native_cpu::WGSize_t> MaxWGSize = std::nullopt;
  std::optional<uint64_t> MaxLinearWGSize = std::nullopt;
  std::vector<ur_mem_handle_t> ReferencedArgs;
};
