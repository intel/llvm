//===--------- usm.hpp - CUDA Adapter -------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include <umf_helpers.hpp>
#include <umf_pools/disjoint_pool_config_parser.hpp>

usm::DisjointPoolAllConfigs InitializeDisjointPoolConfig();

struct ur_usm_pool_handle_t_ {
  std::atomic_uint32_t RefCount = 1;

  ur_context_handle_t Context = nullptr;

  usm::DisjointPoolAllConfigs DisjointPoolConfigs =
      usm::DisjointPoolAllConfigs();

  umf::pool_unique_handle_t DeviceMemPool;
  umf::pool_unique_handle_t SharedMemPool;
  umf::pool_unique_handle_t HostMemPool;

  ur_usm_pool_handle_t_(ur_context_handle_t Context,
                        ur_usm_pool_desc_t *PoolDesc);

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  bool hasUMFPool(umf_memory_pool_t *umf_pool);
};

// Exception type to pass allocation errors
class UsmAllocationException {
  const ur_result_t Error;

public:
  UsmAllocationException(ur_result_t Err) : Error{Err} {}
  ur_result_t getError() const { return Error; }
};

ur_result_t USMDeviceAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               ur_usm_device_mem_flags_t Flags, size_t Size,
                               uint32_t Alignment);

ur_result_t USMSharedAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               ur_usm_host_mem_flags_t,
                               ur_usm_device_mem_flags_t, size_t Size,
                               uint32_t Alignment);

ur_result_t USMHostAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                             ur_usm_host_mem_flags_t Flags, size_t Size,
                             uint32_t Alignment);
