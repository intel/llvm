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
#include "event.hpp"
#include "ur_pool_manager.hpp"

struct UsmPool {
  UsmPool(umf::pool_unique_handle_t pPool)
      : umfPool(std::move(pPool)),
        asyncPool([](ur_event_handle_t hEvent) { return hEvent->release(); }) {}
  umf::pool_unique_handle_t umfPool;
  // 'asyncPool' needs to be declared after 'umfPool' so its destructor is
  // invoked first.
  EnqueuedPool asyncPool;
};

struct ur_usm_pool_handle_t_ : ur_object {
  ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                        ur_usm_pool_desc_t *pPoolDes);

  ur_context_handle_t getContextHandle() const;

  ur_result_t allocate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                       const ur_usm_desc_t *pUSMDesc, ur_usm_type_t type,
                       size_t size, void **ppRetMem);
  ur_result_t free(void *ptr);

  std::optional<std::pair<void *, ur_event_handle_t>>
  allocateEnqueued(ur_context_handle_t hContext, void *hQueue,
                   bool isInOrderQueue, ur_device_handle_t hDevice,
                   const ur_usm_desc_t *pUSMDesc, ur_usm_type_t type,
                   size_t size);

  void cleanupPools();
  void cleanupPoolsForQueue(void *hQueue);

private:
  ur_context_handle_t hContext;
  usm::pool_manager<usm::pool_descriptor, UsmPool> poolManager;

  UsmPool *getPool(const usm::pool_descriptor &desc);
};
