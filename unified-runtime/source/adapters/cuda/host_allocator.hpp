//===--------- host_allocator.hpp - CUDA Adapter
//-----------------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <mutex>
#include <ur/ur.hpp>
#include <vector>

#include "common.hpp"
#include "queue.hpp"

/// Allocator to manage async memory allocated on the host
class host_allocator {
public:
  host_allocator(const host_allocator &obj) = delete;

  ~host_allocator() {
    for (auto HeadPtr : PoolsHeadPtr) {
      UR_CHECK_ERROR(urUSMFree(Context, HeadPtr));
    }
    Allocations.clear();
    FreeMemories.clear();
  }

  static host_allocator &getInstance(ur_context_handle_t hContext) {
    std::lock_guard<std::mutex> Lock(Mtx);
    static host_allocator Instance(hContext);

    return Instance;
  }

  ur_result_t allocate(size_t size, void **ppMem) {
    std::lock_guard<std::mutex> Lock(Mtx);
    for (auto FreeMemoriesIt = FreeMemories.begin();
         FreeMemoriesIt != FreeMemories.end(); FreeMemoriesIt++) {
      if (size >= FreeMemoriesIt->second) {
        *ppMem = FreeMemoriesIt->first;
        uint64_t CurFreeMemAddress =
            reinterpret_cast<uint64_t>(FreeMemoriesIt->first);
        size_t RemainingSize = FreeMemoriesIt->second - size;
        if (RemainingSize == 0) {
          Allocations.insert({FreeMemoriesIt->first, size});
          FreeMemories.erase(FreeMemoriesIt);
        } else {
          Allocations.insert({FreeMemoriesIt->first, size});
          FreeMemories.insert(
              {reinterpret_cast<void *>(CurFreeMemAddress + size),
               RemainingSize});
          FreeMemories.erase(FreeMemoriesIt);
        }
        return UR_RESULT_SUCCESS;
      }
    }
    void *NewAllocatedMem = nullptr;
    UR_CHECK_ERROR(
        urUSMHostAlloc(Context, nullptr, nullptr, size, &NewAllocatedMem));

    Allocations.insert({NewAllocatedMem, size});
    *ppMem = NewAllocatedMem;
    PoolsHeadPtr.push_back(NewAllocatedMem);

    return UR_RESULT_SUCCESS;
  }

  ur_result_t deallocate(void *pMem) {
    std::lock_guard<std::mutex> Lock(Mtx);
    for (auto AllocationsIt = Allocations.begin();
         AllocationsIt != Allocations.end(); AllocationsIt++) {
      if (AllocationsIt->first == pMem) {
        size_t FreeSize = AllocationsIt->second;
        uint64_t FreeMemStartAddress =
            reinterpret_cast<uint64_t>(AllocationsIt->first);

        // Merge before and after freeMemory if it exist.
        auto FindFreeMemoryBef =
            [FreeMemStartAddress](const std::pair<void *, size_t> &FreeMemory) {
              return reinterpret_cast<uint64_t>(FreeMemory.first) ==
                     FreeMemStartAddress - FreeMemory.second;
            };
        auto FindFreeMemoryAft =
            [FreeMemStartAddress,
             AllocationsIt](const std::pair<void *, size_t> &FreeMemory) {
              return reinterpret_cast<uint64_t>(FreeMemory.first) ==
                     FreeMemStartAddress + AllocationsIt->second;
            };

        auto FreeMemBeforeIt = std::find_if(
            FreeMemories.begin(), FreeMemories.end(), FindFreeMemoryBef);
        auto FreeMemAfterIt = std::find_if(
            FreeMemories.begin(), FreeMemories.end(), FindFreeMemoryAft);
        if (FreeMemBeforeIt != FreeMemories.end()) {
          FreeSize += FreeMemBeforeIt->second;
          FreeMemStartAddress =
              reinterpret_cast<uint64_t>(FreeMemBeforeIt->first);
          FreeMemories.erase(FreeMemBeforeIt);
        }
        if (FreeMemAfterIt != FreeMemories.end()) {
          FreeSize += FreeMemAfterIt->second;
          FreeMemories.erase(FreeMemAfterIt);
        }

        FreeMemories.insert(
            {reinterpret_cast<void *>(FreeMemStartAddress), FreeSize});
        Allocations.erase(AllocationsIt);
        return UR_RESULT_SUCCESS;
      }
    }
    return UR_RESULT_ERROR_INVALID_HOST_PTR;
  }

private:
  std::set<std::pair<void *, size_t>> Allocations;
  std::set<std::pair<void *, size_t>> FreeMemories;
  size_t TotalPoolSize = 0;
  std::vector<void *> PoolsHeadPtr;
  ur_context_handle_t Context;

  static std::mutex Mtx;

  host_allocator(ur_context_handle_t hContext) : Context(hContext) {};
};

std::mutex host_allocator::Mtx;
