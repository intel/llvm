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
class host_allocator {
public:
  host_allocator(const host_allocator& obj) = delete;

  ~host_allocator() {
    for (auto Allocation : Allocations) {
      UR_CHECK_ERROR(urUSMFree(Context, Allocation.first));
    }

    for (auto FreeMemory : FreeMemories) {
      UR_CHECK_ERROR(urUSMFree(Context, FreeMemory.second));
    }
    Allocations.clear();
    FreeMemories.clear();
  }

  static host_allocator& getInstance(ur_context_handle_t hContext) {
    std::lock_guard<std::mutex> lock(mtx);
    static host_allocator instance(hContext);
      
    return instance;
  }

  ur_result_t allocate(size_t size, void** ppMem) {
    auto FirstFitMem = FreeMemories.lower_bound({size, 0});

    if (FirstFitMem == FreeMemories.end()) {
      UR_CHECK_ERROR(urUSMHostAlloc(Context, nullptr, nullptr, size, ppMem));
      Allocations.insert({*ppMem, {size, size}});
      TotalAllocatedMem += size;
      return UR_RESULT_SUCCESS;
    } else if (WastedMem >= size) {
      std::vector<void*> needResizedMems;
      size_t curWastedMem = 0;
      for (auto AllocationsIt = Allocations.begin(); curWastedMem < size && AllocationsIt != Allocations.end(); AllocationsIt++) {
        if (AllocationsIt->second.second > AllocationsIt->second.first) {
          needResizedMems.push_back(AllocationsIt->first);
          curWastedMem += (AllocationsIt->second.second - AllocationsIt->second.first);
        }
      }
      for (size_t i = 0; i < needResizedMems.size(); i++) {
        void* newMem = nullptr;
        auto &allocation = Allocations[needResizedMems[i]];
        UR_CHECK_ERROR(urUSMHostAlloc(Context, nullptr, nullptr, allocation.first, &newMem));
        std::memcpy(newMem, needResizedMems[i], allocation.first);
        WastedMem -= (allocation.second - allocation.first);
        Allocations.insert({newMem, {allocation.first, allocation.first}});
        Allocations.erase(needResizedMems[i]);
      }
      UR_CHECK_ERROR(urUSMHostAlloc(Context, nullptr, nullptr, size, ppMem));
      Allocations.insert({*ppMem, {size, size}});
      TotalAllocatedMem += size;

      return UR_RESULT_SUCCESS;
    }
    *ppMem = FirstFitMem->second;
    Allocations.insert({FirstFitMem->second, {size, FirstFitMem->first}});
    TotalAllocatedMem += size;
    WastedMem += FirstFitMem->first - size;
    FreeMemories.erase(FirstFitMem);
    return UR_RESULT_SUCCESS;
  }

  ur_result_t deallocate(void* pMem) {

    auto AllocatedMemory = Allocations.find(pMem);
    if (AllocatedMemory == Allocations.end()) {
      return UR_RESULT_ERROR_INVALID_HOST_PTR;
    }
    FreeMemories.insert({AllocatedMemory->second.second, AllocatedMemory->first});
    TotalAllocatedMem -= AllocatedMemory->second.first;
    WastedMem -= (AllocatedMemory->second.second - AllocatedMemory->second.first);
    Allocations.erase(AllocatedMemory);
    return UR_RESULT_SUCCESS;
  }

private:
    // Map between host allocated ptr and {allocation_memory_size, total_memory_size}
    std::unordered_map<void*, std::pair<size_t, size_t>> Allocations;
    // Set with total_free_memory_size and host_allocated_ptr
    std::set<std::pair<size_t, void*>> FreeMemories;
    size_t TotalAllocatedMem = 0;
    size_t WastedMem = 0;
    ur_context_handle_t Context;

    static std::mutex mtx;

    host_allocator(ur_context_handle_t hContext) : Context(hContext) {};
};

std::mutex host_allocator::mtx;
