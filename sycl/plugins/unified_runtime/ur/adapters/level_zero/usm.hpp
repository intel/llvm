//===--------- usm.hpp - Level Zero Adapter --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "common.hpp"

#include <umf_helpers.hpp>

usm::DisjointPoolAllConfigs InitializeDisjointPoolConfig();

struct ur_usm_pool_handle_t_ : _ur_object {
  bool zeroInit;

  usm::DisjointPoolAllConfigs DisjointPoolConfigs =
      InitializeDisjointPoolConfig();

  std::unordered_map<ur_device_handle_t, umf::pool_unique_handle_t>
      DeviceMemPools;
  std::unordered_map<ur_device_handle_t, umf::pool_unique_handle_t>
      SharedMemPools;
  std::unordered_map<ur_device_handle_t, umf::pool_unique_handle_t>
      SharedReadOnlyMemPools;
  umf::pool_unique_handle_t HostMemPool;

  ur_usm_pool_handle_t_(ur_context_handle_t Context,
                        ur_usm_pool_desc_t *PoolDesc);
};

// Exception type to pass allocation errors
class UsmAllocationException {
  const ur_result_t Error;

public:
  UsmAllocationException(ur_result_t Err) : Error{Err} {}
  ur_result_t getError() const { return Error; }
};

// Implements memory allocation via L0 RT for USM allocator interface.
class USMMemoryProvider {
private:
  ur_result_t &getLastStatusRef() {
    static thread_local ur_result_t LastStatus = UR_RESULT_SUCCESS;
    return LastStatus;
  }

protected:
  ur_context_handle_t Context;
  ur_device_handle_t Device;
  size_t MinPageSize;

  // Internal allocation routine which must be implemented for each allocation
  // type
  virtual ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                                   uint32_t Alignment) = 0;

public:
  umf_result_t initialize(ur_context_handle_t Ctx, ur_device_handle_t Dev);
  umf_result_t alloc(size_t Size, size_t Align, void **Ptr);
  umf_result_t free(void *Ptr, size_t Size);
  void get_last_native_error(const char **ErrMsg, int32_t *ErrCode);
  umf_result_t get_min_page_size(void *, size_t *);
  umf_result_t get_recommended_page_size(size_t, size_t *) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  umf_result_t purge_lazy(void *, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  umf_result_t purge_force(void *, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  // TODO: Different name for each provider (Host/Shared/SharedRO/Device)
  const char *get_name() { return "USM"; };

  virtual ~USMMemoryProvider() = default;
};

// Allocation routines for shared memory type
class USMSharedMemoryProvider final : public USMMemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

// Allocation routines for shared memory type that is only modified from host.
class USMSharedReadOnlyMemoryProvider final : public USMMemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

// Allocation routines for device memory type
class USMDeviceMemoryProvider final : public USMMemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

// Allocation routines for host memory type
class USMHostMemoryProvider final : public USMMemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

ur_result_t USMDeviceAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               ur_usm_device_mem_flags_t *Flags, size_t Size,
                               uint32_t Alignment);

ur_result_t USMSharedAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               ur_usm_host_mem_flags_t *,
                               ur_usm_device_mem_flags_t *, size_t Size,
                               uint32_t Alignment);

ur_result_t USMHostAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                             ur_usm_host_mem_flags_t *Flags, size_t Size,
                             uint32_t Alignment);

// If indirect access tracking is not enabled then this functions just performs
// zeMemFree. If indirect access tracking is enabled then reference counting is
// performed.
ur_result_t ZeMemFreeHelper(ur_context_handle_t Context, void *Ptr);

ur_result_t USMFreeHelper(ur_context_handle_t Context, void *Ptr,
                          bool OwnZeMemHandle = true);

bool ShouldUseUSMAllocator();

extern const bool UseUSMAllocator;
