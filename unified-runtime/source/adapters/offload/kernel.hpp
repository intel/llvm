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

namespace {
template <typename T> T nextPowerOf2(T Val) {
  // https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
  Val--;
  Val |= Val >> 1;
  Val |= Val >> 2;
  Val |= Val >> 4;
  Val |= Val >> 8;
  Val |= Val >> 16;
  if constexpr (sizeof(Val) > 4) {
    Val |= Val >> 32;
  }
  return ++Val;
}
} // namespace

struct ur_kernel_handle_t_ : RefCounted {

  // Simplified version of the CUDA adapter's argument implementation
  struct OffloadKernelArguments {
    static constexpr size_t MaxParamBytes = 4096u;
    using final_buffer_t = std::array<char, MaxParamBytes>;
    using args_t = std::vector<char>;
    using args_size_t = std::vector<size_t>;
    using args_offset_t = std::vector<size_t>;
    final_buffer_t RealisedBuffer;
    args_t ParamStorage;
    args_size_t ParamSizes;
    args_offset_t Pointers;
    bool Dirty = true;
    size_t RealisedSpace;

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

      auto Base = ParamStorage.size();
      ParamStorage.resize(Base + Size);
      std::memcpy(&ParamStorage[Base], Arg, Size);
      Pointers[Index] = Base;
      Dirty = true;
    }

    void addMemObjArg(int Index, ur_mem_handle_t hMem, ur_mem_flags_t Flags) {
      assert(hMem && "Invalid mem handle");
      // If a memobj is already set at this index, update the entry rather
      // than adding a duplicate one
      for (auto &Arg : MemObjArgs) {
        if (Arg.Index == Index) {
          Arg = MemObjArg{hMem, Index, Flags};
          return;
        }
      }
      MemObjArgs.push_back(MemObjArg{hMem, Index, Flags});
      Dirty = true;
    }

    void realise() noexcept {
      if (!Dirty) {
        return;
      }

      size_t Space = sizeof(RealisedBuffer);
      void *Offset = reinterpret_cast<void *>(0);
      char *Base = &RealisedBuffer[0];
      for (size_t I = 0; I < Pointers.size(); I++) {
        void *ValueBase = &ParamStorage[Pointers[I]];
        size_t Size = ParamSizes[I];
        size_t Align = nextPowerOf2(Size);

        // Align the value to a multiple of the size
        // TODO: This is probably not correct, but UR doesn't allow specifying
        // the alignment of arguments
        if (!std::align(Align, Size, Offset, Space) && Offset) {
          // Ran out of space. TODO: Handle properly
          // TODO: Since we start at address 0, there's no way to check whether
          // the first allocation is a success or not.
          abort();
        }
        Space -= Size;

        std::memcpy(&Base[reinterpret_cast<uintptr_t>(Offset)], ValueBase,
                    Size);
        Offset = &reinterpret_cast<char *>(Offset)[Size];
      }

      Dirty = false;
      RealisedSpace = reinterpret_cast<uintptr_t>(Offset);
    }

    const char *getStorage() noexcept {
      realise();
      return &RealisedBuffer[0];
    }

    size_t getStorageSize() noexcept {
      realise();
      return RealisedSpace;
    }
  };

  ol_symbol_handle_t OffloadKernel;
  ur_program_handle_t Program;
  OffloadKernelArguments Args{};
  std::string Name;
};
