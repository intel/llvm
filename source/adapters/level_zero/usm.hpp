//===--------- usm.hpp - Level Zero Adapter -------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

  ur_context_handle_t Context{};

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

// UMF memory provider interface for USM.
class USMMemoryProviderBase {
protected:
  ur_context_handle_t Context;
  ur_device_handle_t Device;

  ur_result_t &getLastStatusRef() {
    static thread_local ur_result_t LastStatus = UR_RESULT_SUCCESS;
    return LastStatus;
  }

  // Internal allocation routine which must be implemented for each allocation
  // type
  virtual ur_result_t allocateImpl(void **, size_t, uint32_t) = 0;

public:
  virtual void get_last_native_error(const char **ErrMsg, int32_t *ErrCode) {
    std::ignore = ErrMsg;
    *ErrCode = static_cast<int32_t>(getLastStatusRef());
  };
  virtual umf_result_t initialize(ur_context_handle_t, ur_device_handle_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  virtual umf_result_t alloc(size_t, size_t, void **) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  virtual umf_result_t free(void *, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  virtual umf_result_t get_min_page_size(void *, size_t *) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  virtual umf_result_t get_recommended_page_size(size_t, size_t *) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  virtual umf_result_t purge_lazy(void *, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  virtual umf_result_t purge_force(void *, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  };
  virtual const char *get_name() { return ""; };
  virtual ~USMMemoryProviderBase() = default;
};

// Implements USM memory provider interface for L0 RT USM memory allocations.
class L0MemoryProvider : public USMMemoryProviderBase {
private:
  // Min page size query function for L0MemoryProvider.
  umf_result_t GetL0MinPageSize(void *Mem, size_t *PageSize);
  size_t MinPageSize = 0;
  bool MinPageSizeCached = false;

public:
  umf_result_t initialize(ur_context_handle_t Ctx,
                          ur_device_handle_t Dev) override;
  umf_result_t alloc(size_t Size, size_t Align, void **Ptr) override;
  umf_result_t free(void *Ptr, size_t Size) override;
  umf_result_t get_min_page_size(void *, size_t *) override;
  // TODO: Different name for each provider (Host/Shared/SharedRO/Device)
  const char *get_name() override { return "L0"; };
};

// Allocation routines for shared memory type
class L0SharedMemoryProvider final : public L0MemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

// Allocation routines for shared memory type that is only modified from host.
class L0SharedReadOnlyMemoryProvider final : public L0MemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

// Allocation routines for device memory type
class L0DeviceMemoryProvider final : public L0MemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

// Allocation routines for host memory type
class L0HostMemoryProvider final : public L0MemoryProvider {
protected:
  ur_result_t allocateImpl(void **ResultPtr, size_t Size,
                           uint32_t Alignment) override;
};

// Simple proxy for memory allocations. It is used for the UMF tracking
// capabilities.
class USMProxyPool {
public:
  umf_result_t initialize(umf_memory_provider_handle_t Provider) noexcept {
    this->hProvider = Provider;
    return UMF_RESULT_SUCCESS;
  }
  void *malloc(size_t Size) noexcept { return aligned_malloc(Size, 0); }
  void *calloc(size_t Num, size_t Size) noexcept {
    std::ignore = Num;
    std::ignore = Size;

    // Currently not needed
    umf::getPoolLastStatusRef<USMProxyPool>() = UMF_RESULT_ERROR_NOT_SUPPORTED;
    return nullptr;
  }
  void *realloc(void *Ptr, size_t Size) noexcept {
    std::ignore = Ptr;
    std::ignore = Size;

    // Currently not needed
    umf::getPoolLastStatusRef<USMProxyPool>() = UMF_RESULT_ERROR_NOT_SUPPORTED;
    return nullptr;
  }
  void *aligned_malloc(size_t Size, size_t Alignment) noexcept {
    void *Ptr = nullptr;
    auto Ret = umfMemoryProviderAlloc(hProvider, Size, Alignment, &Ptr);
    if (Ret != UMF_RESULT_SUCCESS) {
      umf::getPoolLastStatusRef<USMProxyPool>() = Ret;
    }
    return Ptr;
  }
  size_t malloc_usable_size(void *Ptr) noexcept {
    std::ignore = Ptr;

    // Currently not needed
    return 0;
  }
  enum umf_result_t free(void *Ptr) noexcept {
    return umfMemoryProviderFree(hProvider, Ptr, 0);
  }
  enum umf_result_t get_last_allocation_error() {
    return umf::getPoolLastStatusRef<USMProxyPool>();
  }
  umf_memory_provider_handle_t hProvider;
};

// If indirect access tracking is not enabled then this functions just performs
// zeMemFree. If indirect access tracking is enabled then reference counting is
// performed.
ur_result_t ZeMemFreeHelper(ur_context_handle_t Context, void *Ptr);

ur_result_t USMFreeHelper(ur_context_handle_t Context, void *Ptr,
                          bool OwnZeMemHandle = true);

extern const bool UseUSMAllocator;
