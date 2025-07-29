//===----------- kernel.hpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <OffloadAPI.h>
#include <array>
#include <cstring>
#include <numeric>
#include <ur_api.h>
#include <vector>

#include "common.hpp"
#include "memory.hpp"

struct ur_kernel_handle_t_ : RefCounted {

  // Simplified version of the CUDA adapter's argument implementation
  struct OffloadKernelArguments {
    static constexpr size_t MaxParamBytes = 4096u;
    using args_t = std::array<char, MaxParamBytes>;
    using args_size_t = std::vector<size_t>;
    using args_ptr_t = std::vector<void *>;
    args_t Storage;
    size_t StorageUsed = 0;
    args_size_t ParamSizes;
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
      if (Index + 1 > Pointers.size()) {
        Pointers.resize(Index + 1);
        ParamSizes.resize(Index + 1);
      }
      ParamSizes[Index] = Size;
      // Calculate the insertion point in the array.
      size_t InsertPos = std::accumulate(std::begin(ParamSizes),
                                         std::begin(ParamSizes) + Index, 0);
      // Update the stored value for the argument.
      std::memcpy(&Storage[InsertPos], Arg, Size);
      Pointers[Index] = &Storage[InsertPos];
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
      return std::accumulate(std::begin(ParamSizes), std::end(ParamSizes), 0);
    }
  };

  ol_symbol_handle_t OffloadKernel;
  OffloadKernelArguments Args{};
};
