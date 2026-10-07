//===--------- usm.hpp - HIP Adapter --------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"
#include "common/ur_ref_count.hpp"

#include <umf_helpers.hpp>
#include <umf_pools/disjoint_pool_config_parser.hpp>

usm::DisjointPoolAllConfigs InitializeDisjointPoolConfig();

// A ur_usm_pool_handle_t can represent different types of memory pools. It may
// sit on top of a UMF pool or a hipMemPool_t, but not both.
struct ur_usm_pool_handle_t_ : ur::hip::handle_base {
  ur::RefCount RefCount;

  ur_context_handle_t Context = nullptr;
  ur_device_handle_t Device = nullptr;

  usm::DisjointPoolAllConfigs DisjointPoolConfigs =
      usm::DisjointPoolAllConfigs();

  umf::pool_unique_handle_t DeviceMemPool;
  umf::pool_unique_handle_t SharedMemPool;
  umf::pool_unique_handle_t HostMemPool;

  hipMemPool_t HIPMemPool{0};
  size_t maxSize = 0;

  ur_usm_pool_handle_t_(ur_context_handle_t Context,
                        ur_usm_pool_desc_t *PoolDesc);

  // Explicit device pool backed by a native hipMemPool_t.
  ur_usm_pool_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device,
                        ur_usm_pool_desc_t *PoolDesc);

  // Explicit device default pool.
  ur_usm_pool_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device,
                        hipMemPool_t HIPMemPool);

  bool hasUMFPool(umf_memory_pool_t *umf_pool);

  // To be used if ur_usm_pool_handle_t represents a hipMemPool_t.
  bool usesHipPool() const { return HIPMemPool != hipMemPool_t{0}; };
  hipMemPool_t getHipPool() { return HIPMemPool; };
};

// Implements memory allocation via driver API for USM allocator interface
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
  umf_result_t get_last_native_error(const char **ErrMsg, int32_t *ErrCode);
  umf_result_t get_min_page_size(const void *, size_t *);
  umf_result_t get_recommended_page_size(size_t, size_t *) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  umf_result_t get_name(const char **Name) {
    if (!Name) {
      return UMF_RESULT_ERROR_INVALID_ARGUMENT;
    }

    *Name = "HIP";
    return UMF_RESULT_SUCCESS;
  }

  virtual ~USMMemoryProvider() = default;
};

// Allocation routines for shared memory type
class USMSharedMemoryProvider final : public USMMemoryProvider {
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

bool checkUSMAlignment(uint32_t &alignment, const ur_usm_desc_t *pUSMDesc);
