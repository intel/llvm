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

// A ur_usm_pool_handle_t can represent different types of memory pools. It may
// sit on top of a UMF pool or a CUmemoryPool, but not both.
struct ur_usm_pool_handle_t_ : ur::cuda::handle_base {
  std::atomic_uint32_t RefCount = 1;

  ur_context_handle_t Context = nullptr;
  ur_device_handle_t Device = nullptr;

  usm::DisjointPoolAllConfigs DisjointPoolConfigs =
      usm::DisjointPoolAllConfigs();

  umf::pool_unique_handle_t DeviceMemPool;
  umf::pool_unique_handle_t SharedMemPool;
  umf::pool_unique_handle_t HostMemPool;

  CUmemoryPool CUmemPool{0};
  size_t maxSize = 0;

  ur_usm_pool_handle_t_(ur_context_handle_t Context,
                        ur_usm_pool_desc_t *PoolDesc);

  // Explicit device pool.
  ur_usm_pool_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device,
                        ur_usm_pool_desc_t *PoolDesc);

  // Explicit device default pool.
  ur_usm_pool_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device,
                        CUmemoryPool CUmemPool);

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  bool hasUMFPool(umf_memory_pool_t *umf_pool);

  // To be used if ur_usm_pool_handle_t represents a CUmemoryPool.
  bool usesCudaPool() const { return CUmemPool != CUmemoryPool{0}; };
  CUmemoryPool getCudaPool() { return CUmemPool; };
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
