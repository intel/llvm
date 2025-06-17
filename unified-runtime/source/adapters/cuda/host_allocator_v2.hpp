//===--------- host_allocator.hpp - CUDA Adapter -----------------------------------===//
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
#include <vector>
#include <mutex>
#include <ur/ur.hpp>

#include "common.hpp"
#include "queue.hpp"

/// Allocator to manage async memory allocated on the host
class host_allocator_v2 {
public:
  host_allocator_v2(const host_allocator_v2& obj) = delete;

  ~host_allocator_v2() {
    for (auto HeadPtr : PoolsHeadPtr) {
      UR_CHECK_ERROR(urUSMFree(Context, HeadPtr));
    }
    Allocations.clear();
    FreeMemories.clear();
  }

  static host_allocator_v2& getInstance(ur_context_handle_t hContext) {
    std::lock_guard<std::mutex> lock(mtx);
    static host_allocator_v2 instance(hContext);
      
    return instance;
  }

  ur_result_t allocate(size_t size, void** ppMem) {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto it = FreeMemories.begin(); it != FreeMemories.end(); it++) {
      if (size >= it->second) {
        *ppMem = it->first;
        size_t remainingSize = it->second - size;
        if (remainingSize == 0) {
          Allocations.insert({it->first, size});
          FreeMemories.erase(it);
        } else {
          Allocations.insert({it->first, size});
          FreeMemories.insert({it->first + size, remainingSize});
          FreeMemories.erase(it);
        }
        return UR_RESULT_SUCCESS;
      }
    }
    void *NewAllocatedMem = nullptr;
    UR_CHECK_ERROR(urUSMHostAlloc(Context, nullptr, nullptr, size, &NewAllocatedMem));
    
    Allocations.insert({NewAllocatedMem, size});
    *ppMem = NewAllocatedMem;
    PoolsHeadPtr.push_back(NewAllocatedMem);

    return UR_RESULT_SUCCESS;
  }

  ur_result_t deallocate(void* pMem) {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto it = Allocations.begin(); it != Allocations.end(); it++) {
      if (it->first == pMem) {
        size_t freeSize = it->second;
        void* freeMemStartPtr = it->first;

        // Merge before and after freeMemory if it exist.
        auto findFreeMemoryBef = [it](const std::pair<void*, size_t> &FreeMemory) {
                                                return FreeMemory.first == it->first - FreeMemory.second;
                                            };
        auto findFreeMemoryAft = [it](const std::pair<void*, size_t> &FreeMemory) {
                                                return FreeMemory.first == it->first + it->second;
                                            };
                  
        auto FreeMemBeforeIt = std::find_if(FreeMemories.begin(), FreeMemories.end(), findFreeMemoryBef);
        auto FreeMemAfterIt = std::find_if(FreeMemories.begin(), FreeMemories.end(), findFreeMemoryAft);
        if (FreeMemBeforeIt != FreeMemories.end()) {
          freeSize += FreeMemBeforeIt->second;
          freeMemStartPtr = FreeMemBeforeIt->first;
          FreeMemories.erase(FreeMemBeforeIt);
        }
        if (FreeMemAfterIt != FreeMemories.end()) {
          freeSize += FreeMemAfterIt->second;
          FreeMemories.erase(FreeMemAfterIt);
        }

        FreeMemories.insert({freeMemStartPtr, freeSize});
        Allocations.erase(it);
        return UR_RESULT_SUCCESS;
      }
    }
    return UR_RESULT_ERROR_INVALID_HOST_PTR;
  }

private:
    std::set<std::pair<void*, size_t>> Allocations;
    std::set<std::pair<void*, size_t>> FreeMemories;
    size_t TotalPoolSize = 0;
    std::vector<void*> PoolsHeadPtr;
    ur_context_handle_t Context;

    static std::mutex mtx;

    host_allocator_v2(ur_context_handle_t hContext) : Context(hContext) {};
};

std::mutex host_allocator_v2::mtx;
