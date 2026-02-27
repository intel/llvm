//===--------- usm.cpp - Level Zero Adapter ------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ur_api.h"

#include "../enqueued_pool.hpp"
#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "event.hpp"
#include "ur_pool_manager.hpp"

struct UsmPool;

struct AllocationStats {
public:
  enum UpdateType {
    INCREASE,
    DECREASE,
  };

  void update(UpdateType Type, size_t Size) {
    if (Type == INCREASE) {
      AllocatedMemorySize += Size;
      size_t Current = AllocatedMemorySize.load(std::memory_order_relaxed);
      size_t Peak = PeakAllocatedMemorySize.load(std::memory_order_relaxed);
      if (Peak < Current) {
        PeakAllocatedMemorySize.store(Current, std::memory_order_relaxed);
      }
    } else if (Type == DECREASE) {
      AllocatedMemorySize -= Size;
    }
  }

  size_t getCurrent() {
    return AllocatedMemorySize.load(std::memory_order_relaxed);
  }
  size_t getPeak() {
    return PeakAllocatedMemorySize.load(std::memory_order_relaxed);
  }

private:
  std::atomic_size_t AllocatedMemorySize{0};
  std::atomic_size_t PeakAllocatedMemorySize{0};
};

struct ur_usm_pool_handle_t_ : ur_object {
  ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                        ur_usm_pool_desc_t *pPoolDes);
  ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                        ur_device_handle_t hDevice,
                        ur_usm_pool_desc_t *pPoolDes);

  ur_context_handle_t getContextHandle() const;

  ur_result_t allocate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                       const ur_usm_desc_t *pUSMDesc, ur_usm_type_t type,
                       size_t size, void **ppRetMem);
  ur_result_t free(void *ptr, umf_memory_pool_handle_t umfPool = nullptr);

  bool hasPool(const umf_memory_pool_handle_t hPool);

  std::optional<std::pair<void *, ur_event_handle_t>>
  allocateEnqueued(ur_context_handle_t hContext, void *hQueue,
                   bool isInOrderQueue, ur_device_handle_t hDevice,
                   ur_usm_type_t type, size_t size);

  void cleanupPools();
  void cleanupPoolsForQueue(void *hQueue);
  size_t getTotalReservedSize();
  size_t getPeakReservedSize();
  size_t getTotalUsedSize();
  size_t getPeakUsedSize();

  UsmPool *getPool(const usm::pool_descriptor &desc);

  ur::RefCount RefCount;

private:
  ur_context_handle_t hContext;
  usm::pool_manager<usm::pool_descriptor, UsmPool> poolManager;
  AllocationStats allocStats;
};

struct UsmPool {
  UsmPool(ur_usm_pool_handle_t urPool, umf::pool_unique_handle_t umfPool)
      : urPool(urPool), umfPool(std::move(umfPool)),
        asyncPool([](ur_event_handle_t hEvent) { return hEvent->release(); },
                  [context = urPool->getContextHandle()](void *ptr) {
                    return ur::level_zero::urUSMFree(context, ptr);
                  }) {}
  ur_usm_pool_handle_t urPool;
  umf::pool_unique_handle_t umfPool;
  // 'asyncPool' needs to be declared after 'umfPool' so its destructor is
  // invoked first.
  EnqueuedPool asyncPool;
};
