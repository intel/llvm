//===----------- kernel.hpp - LLVM Offload Adapter  -----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <OffloadAPI.h>
#include <array>
#include <cstring>
#include <numeric>
#include <unified-runtime/ur_api.h>
#include <vector>

#include "common.hpp"
#include "memory.hpp"

struct ur_kernel_handle_t_ : RefCounted {
  explicit ur_kernel_handle_t_(ur_program_handle_t Program) : Program(Program) {
    urProgramRetain(Program);
  }

  ~ur_kernel_handle_t_() { urProgramRelease(Program); }

  // Simplified version of the CUDA adapter's argument implementation
  struct OffloadKernelArguments {
    static constexpr size_t MaxParamBytes = 4096u;
    using args_t = std::array<char, MaxParamBytes>;
    using args_size_t = std::vector<size_t>;
    using args_ptr_t = std::vector<void *>;
    args_t Storage;
    size_t StorageUsed = 0;
    // Argument sizes passed to olLaunchKernel.
    args_size_t ParamSizes;
    // Bytes each argument occupies in Storage. Only differs from ParamSizes
    // for local memory arguments.
    args_size_t StorageSizes;
    args_ptr_t Pointers;

    struct MemObjArg {
      ur_mem_handle_t_ *Mem;
      int Index;
      ur_mem_flags_t AccessFlags;
    };
    std::vector<MemObjArg> MemObjArgs;

    // Add an argument. If it already exists, it is replaced. Gaps are filled
    // with empty arguments.
    void addArg(size_t Index, size_t Size, const void *Arg) {
      setArg(Index, Size, Arg, Size);
    }

    // Add a local memory argument of Size bytes. Only the Level Zero plugin
    // takes local memory as kernel arguments, and it hands them to the driver
    // in one of two ways: zeCommandListAppendLaunchKernelWithArguments reads
    // the size from the size_t the argument pointer points to, and the
    // zeKernelSetArgumentValue fallback takes the argument size as the size
    // and ignores the value. Store the size in both places so that both work.
    void addLocalArg(size_t Index, size_t Size) {
      setArg(Index, sizeof(Size), &Size, Size);
    }

    void addMemObjArg(int Index, ur_mem_handle_t hMem, ur_mem_flags_t Flags) {
      // Handle zero-sized buffers
      if (hMem == nullptr) {
        addArg(Index, 0, nullptr);
        return;
      }

      // If a memobj is already set at this index, update the entry rather
      // than adding a duplicate one
      for (auto &Arg : MemObjArgs) {
        if (Arg.Index == Index) {
          Arg = MemObjArg{hMem, Index, Flags};
          return;
        }
      }
      MemObjArgs.push_back(MemObjArg{hMem, Index, Flags});

      auto Ptr = std::get<BufferMem>(hMem->Mem).Ptr;
      addArg(Index, sizeof(void *), &Ptr);
    }

    const args_ptr_t &getPointers() const noexcept { return Pointers; }

    const char *getStorage() const noexcept { return Storage.data(); }

    size_t getStorageSize() const noexcept {
      return std::accumulate(std::begin(StorageSizes), std::end(StorageSizes),
                             size_t{0});
    }

  private:
    // Store StorageSize bytes from Arg as argument Index, and report its size
    // to liboffload as ParamSize.
    void setArg(size_t Index, size_t StorageSize, const void *Arg,
                size_t ParamSize) {
      if (Index + 1 > Pointers.size()) {
        Pointers.resize(Index + 1);
        ParamSizes.resize(Index + 1);
        StorageSizes.resize(Index + 1);
      }
      // Calculate the insertion point in the array.
      size_t InsertPos =
          std::accumulate(std::begin(StorageSizes),
                          std::begin(StorageSizes) + Index, size_t{0});
      // Validate bounds before memcpy to prevent heap buffer overflow
      // Check for integer overflow: validate Size first to avoid underflow
      if (StorageSize > MaxParamBytes ||
          InsertPos > MaxParamBytes - StorageSize) {
        throw UR_RESULT_ERROR_OUT_OF_RESOURCES;
      }
      StorageSizes[Index] = StorageSize;
      ParamSizes[Index] = ParamSize;
      // Update the stored value for the argument.
      std::memcpy(&Storage[InsertPos], Arg, StorageSize);
      Pointers[Index] = &Storage[InsertPos];
    }
  };

  ol_symbol_handle_t OffloadKernel = nullptr;
  ur_program_handle_t Program;
  OffloadKernelArguments Args{};
  std::string Name;
};
