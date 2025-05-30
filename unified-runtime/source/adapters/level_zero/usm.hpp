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

#include "enqueued_pool.hpp"
#include "event.hpp"
#include "ur_api.h"
#include "ur_pool_manager.hpp"
#include <set>
#include <umf_helpers.hpp>

usm::DisjointPoolAllConfigs InitializeDisjointPoolConfig();

struct UsmPool {
  UsmPool(umf::pool_unique_handle_t Pool)
      : UmfPool(std::move(Pool)), AsyncPool([](ur_event_handle_t Event) {
          return urEventReleaseInternal(Event);
        }) {}
  umf::pool_unique_handle_t UmfPool;
  // 'AsyncPool' needs to be declared after 'UmfPool' so its destructor is
  // invoked first.
  EnqueuedPool AsyncPool;
};

struct ur_usm_pool_handle_t_ : ur_object {
  ur_usm_pool_handle_t_(ur_context_handle_t Context,
                        ur_usm_pool_desc_t *PoolDesc, bool IsProxy = false);
  ur_usm_pool_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device,
                        ur_usm_pool_desc_t *PoolDesc);

  ur_result_t allocate(ur_context_handle_t Context, ur_device_handle_t Device,
                       const ur_usm_desc_t *USMDesc, ur_usm_type_t Type,
                       size_t Size, void **RetMem);

  std::optional<std::pair<void *, ur_event_handle_t>>
  allocateEnqueued(ur_queue_handle_t Queue, ur_device_handle_t Device,
                   const ur_usm_desc_t *USMDesc, ur_usm_type_t Type,
                   size_t Size);

  bool hasPool(const umf_memory_pool_handle_t Pool);
  UsmPool *getPoolByHandle(const umf_memory_pool_handle_t Pool);
  void cleanupPools();
  void cleanupPoolsForQueue(ur_queue_handle_t Queue);

  ur_context_handle_t Context;

private:
  UsmPool *getPool(const usm::pool_descriptor &Desc);
  usm::pool_manager<usm::pool_descriptor, UsmPool> PoolManager;
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
  virtual void get_last_native_error(const char ** /*ErrMsg*/,
                                     int32_t *ErrCode) {
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
  virtual umf_result_t get_min_page_size(const void *, size_t *) {
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
  virtual umf_result_t allocation_merge(void *, void *, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  }
  virtual umf_result_t allocation_split(void *, size_t, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  }
  virtual umf_result_t get_ipc_handle_size(size_t *) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  }
  virtual umf_result_t get_ipc_handle(const void *, size_t, void *) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  }
  virtual umf_result_t put_ipc_handle(void *) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  }
  virtual umf_result_t open_ipc_handle(void *, void **) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  }
  virtual umf_result_t close_ipc_handle(void *, size_t) {
    return UMF_RESULT_ERROR_NOT_SUPPORTED;
  }
  virtual const char *get_name() { return ""; };
  virtual ~USMMemoryProviderBase() = default;
};

// Implements USM memory provider interface for L0 RT USM memory allocations.
class L0MemoryProvider : public USMMemoryProviderBase {
private:
  // Min page size query function for L0MemoryProvider.
  umf_result_t GetL0MinPageSize(const void *Mem, size_t *PageSize);
  size_t MinPageSize = 0;
  bool MinPageSizeCached = false;

public:
  umf_result_t initialize(ur_context_handle_t Ctx,
                          ur_device_handle_t Dev) override;
  umf_result_t alloc(size_t Size, size_t Align, void **Ptr) override;
  umf_result_t free(void *Ptr, size_t Size) override;
  umf_result_t get_min_page_size(const void *, size_t *) override;
  // TODO: Different name for each provider (Host/Shared/SharedRO/Device)
  const char *get_name() override { return "Level Zero"; };
  umf_result_t get_ipc_handle_size(size_t *) override;
  umf_result_t get_ipc_handle(const void *, size_t, void *) override;
  umf_result_t put_ipc_handle(void *) override;
  umf_result_t open_ipc_handle(void *, void **) override;
  umf_result_t close_ipc_handle(void *, size_t) override;
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
  void *calloc(size_t /*Num*/, size_t /*Size*/) noexcept {

    // Currently not needed
    umf::getPoolLastStatusRef<USMProxyPool>() = UMF_RESULT_ERROR_NOT_SUPPORTED;
    return nullptr;
  }
  void *realloc(void * /*Ptr*/, size_t /*Size*/) noexcept {

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
  size_t malloc_usable_size(void * /*Ptr*/) noexcept {

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
