//===--------- usm.cpp - CUDA Adapter -------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>

#include "adapter.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "event.hpp"
#include "platform.hpp"
#include "queue.hpp"
#include "ur_util.hpp"
#include "usm.hpp"

#include <cuda.h>

namespace umf {
ur_result_t getProviderNativeError(const char *, int32_t) {
  // TODO: implement when UMF supports CUDA
  return UR_RESULT_ERROR_UNKNOWN;
}
} // namespace umf

/// USM: Implements USM Host allocations using CUDA Pinned Memory
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory
UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t hPool, size_t size, void **ppMem) {
  auto alignment = pUSMDesc ? pUSMDesc->align : 0u;
  UR_ASSERT(!pUSMDesc ||
                (alignment == 0 || ((alignment & (alignment - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  if (!hPool) {
    return USMHostAllocImpl(ppMem, hContext, /* flags */ 0, size, alignment);
  }

  auto UMFPool = hPool->HostMemPool.get();
  *ppMem = umfPoolAlignedMalloc(UMFPool, size, alignment);
  if (*ppMem == nullptr) {
    auto umfErr = umfPoolGetLastAllocationError(UMFPool);
    return umf::umf2urResult(umfErr);
  }
  return UR_RESULT_SUCCESS;
}

/// USM: Implements USM device allocations using a normal CUDA device pointer
///
UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t hPool,
                 size_t size, void **ppMem) {
  auto alignment = pUSMDesc ? pUSMDesc->align : 0u;
  UR_ASSERT(!pUSMDesc ||
                (alignment == 0 || ((alignment & (alignment - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  if (!hPool) {
    return USMDeviceAllocImpl(ppMem, hContext, hDevice, /* flags */ 0, size,
                              alignment);
  }

  auto UMFPool = hPool->DeviceMemPool.get();
  *ppMem = umfPoolAlignedMalloc(UMFPool, size, alignment);
  if (*ppMem == nullptr) {
    auto umfErr = umfPoolGetLastAllocationError(UMFPool);
    return umf::umf2urResult(umfErr);
  }
  return UR_RESULT_SUCCESS;
}

/// USM: Implements USM Shared allocations using CUDA Managed Memory
///
UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t hPool,
                 size_t size, void **ppMem) {
  auto alignment = pUSMDesc ? pUSMDesc->align : 0u;
  UR_ASSERT(!pUSMDesc ||
                (alignment == 0 || ((alignment & (alignment - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  if (!hPool) {
    return USMSharedAllocImpl(ppMem, hContext, hDevice, /*host flags*/ 0,
                              /*device flags*/ 0, size, alignment);
  }

  auto UMFPool = hPool->SharedMemPool.get();
  *ppMem = umfPoolAlignedMalloc(UMFPool, size, alignment);
  if (*ppMem == nullptr) {
    auto umfErr = umfPoolGetLastAllocationError(UMFPool);
    return umf::umf2urResult(umfErr);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t USMFreeImpl(ur_context_handle_t hContext, void *Pointer) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    unsigned int IsManaged;
    unsigned int Type;
    unsigned int DeviceOrdinal;
    const int NumAttributes = 3;
    void *AttributeValues[NumAttributes] = {&IsManaged, &Type, &DeviceOrdinal};

    CUpointer_attribute Attributes[NumAttributes] = {
        CU_POINTER_ATTRIBUTE_IS_MANAGED, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
    UR_CHECK_ERROR(cuPointerGetAttributes(
        NumAttributes, Attributes, AttributeValues, (CUdeviceptr)Pointer));
    UR_ASSERT(Type == CU_MEMORYTYPE_DEVICE || Type == CU_MEMORYTYPE_HOST,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);

    std::vector<ur_device_handle_t> ContextDevices = hContext->getDevices();
    ur_platform_handle_t Platform = ContextDevices[0]->getPlatform();
    unsigned int NumDevices = Platform->Devices.size();
    UR_ASSERT(DeviceOrdinal < NumDevices, UR_RESULT_ERROR_INVALID_DEVICE);

    ur_device_handle_t Device = Platform->Devices[DeviceOrdinal].get();
    umf_memory_provider_handle_t MemoryProvider;

    if (IsManaged) {
      MemoryProvider = Device->MemoryProviderShared;
    } else if (Type == CU_MEMORYTYPE_DEVICE) {
      MemoryProvider = Device->MemoryProviderDevice;
    } else {
      MemoryProvider = hContext->MemoryProviderHost;
    }

    UMF_CHECK_ERROR(umfMemoryProviderFree(MemoryProvider, Pointer,
                                          0 /* size is unknown */));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

/// USM: Frees the given USM pointer associated with the context.
///
UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  if (auto Pool = umfPoolByPtr(pMem))
    return umf::umf2urResult(umfPoolFree(Pool, pMem));
  return USMFreeImpl(hContext, pMem);
}

ur_result_t USMDeviceAllocImpl(void **ResultPtr, ur_context_handle_t,
                               ur_device_handle_t Device,
                               ur_usm_device_mem_flags_t, size_t Size,
                               uint32_t Alignment) {
  try {
    ScopedContext Active(Device);
    UMF_CHECK_ERROR(umfMemoryProviderAlloc(Device->MemoryProviderDevice, Size,
                                           Alignment, ResultPtr));
  } catch (ur_result_t Err) {
    return Err;
  }

#ifdef NDEBUG
  std::ignore = Alignment;
#else
  assert((Alignment == 0 ||
          reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0));
#endif
  return UR_RESULT_SUCCESS;
}

ur_result_t USMSharedAllocImpl(void **ResultPtr, ur_context_handle_t,
                               ur_device_handle_t Device,
                               ur_usm_host_mem_flags_t,
                               ur_usm_device_mem_flags_t, size_t Size,
                               uint32_t Alignment) {
  try {
    ScopedContext Active(Device);
    UMF_CHECK_ERROR(umfMemoryProviderAlloc(Device->MemoryProviderShared, Size,
                                           Alignment, ResultPtr));
  } catch (ur_result_t Err) {
    return Err;
  }

#ifdef NDEBUG
  std::ignore = Alignment;
#else
  assert((Alignment == 0 ||
          reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0));
#endif
  return UR_RESULT_SUCCESS;
}

ur_result_t USMHostAllocImpl(void **ResultPtr, ur_context_handle_t hContext,
                             ur_usm_host_mem_flags_t, size_t Size,
                             uint32_t Alignment) {
  try {
    UMF_CHECK_ERROR(umfMemoryProviderAlloc(hContext->MemoryProviderHost, Size,
                                           Alignment, ResultPtr));
  } catch (ur_result_t Err) {
    return Err;
  }

#ifdef NDEBUG
  std::ignore = Alignment;
#else
  assert((Alignment == 0 ||
          reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0));
#endif
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propValueSize,
                     void *pPropValue, size_t *pPropValueSizeRet) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  try {
    switch (propName) {
    case UR_USM_ALLOC_INFO_TYPE: {
      unsigned int Value;
      // do not throw if cuPointerGetAttribute returns CUDA_ERROR_INVALID_VALUE
      CUresult Ret = cuPointerGetAttribute(
          &Value, CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr)pMem);
      if (Ret == CUDA_ERROR_INVALID_VALUE) {
        // pointer not known to the CUDA subsystem
        return ReturnValue(UR_USM_TYPE_UNKNOWN);
      }
      checkErrorUR(Ret, __func__, __LINE__ - 5, __FILE__);
      if (Value) {
        // pointer to managed memory
        return ReturnValue(UR_USM_TYPE_SHARED);
      }
      UR_CHECK_ERROR(cuPointerGetAttribute(
          &Value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)pMem));
      UR_ASSERT(Value == CU_MEMORYTYPE_DEVICE || Value == CU_MEMORYTYPE_HOST,
                UR_RESULT_ERROR_INVALID_MEM_OBJECT);
      if (Value == CU_MEMORYTYPE_DEVICE) {
        // pointer to device memory
        return ReturnValue(UR_USM_TYPE_DEVICE);
      }
      if (Value == CU_MEMORYTYPE_HOST) {
        // pointer to host memory
        return ReturnValue(UR_USM_TYPE_HOST);
      }
      // should never get here
      ur::unreachable();
    }
    case UR_USM_ALLOC_INFO_BASE_PTR: {
#if CUDA_VERSION >= 10020
      // CU_POINTER_ATTRIBUTE_RANGE_START_ADDR was introduced in CUDA 10.2
      void *Base;
      UR_CHECK_ERROR(cuPointerGetAttribute(
          &Base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)pMem));
      return ReturnValue(Base);
#else
      return UR_RESULT_ERROR_INVALID_VALUE;
#endif
    }
    case UR_USM_ALLOC_INFO_SIZE: {
#if CUDA_VERSION >= 10020
      // CU_POINTER_ATTRIBUTE_RANGE_SIZE was introduced in CUDA 10.2
      size_t Value;
      UR_CHECK_ERROR(cuPointerGetAttribute(
          &Value, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)pMem));
      return ReturnValue(Value);
#else
      return UR_RESULT_ERROR_INVALID_VALUE;
#endif
    }
    case UR_USM_ALLOC_INFO_DEVICE: {
      // get device index associated with this pointer
      unsigned int DeviceIndex;
      UR_CHECK_ERROR(cuPointerGetAttribute(&DeviceIndex,
                                           CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                           (CUdeviceptr)pMem));

      // cuda backend has only one platform containing all devices
      ur_platform_handle_t platform;
      ur_adapter_handle_t AdapterHandle = &adapter;
      Result = urPlatformGet(&AdapterHandle, 1, 1, &platform, nullptr);

      // get the device from the platform
      ur_device_handle_t Device = platform->Devices[DeviceIndex].get();
      return ReturnValue(Device);
    }
    case UR_USM_ALLOC_INFO_POOL: {
      auto UMFPool = umfPoolByPtr(pMem);
      if (!UMFPool) {
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      ur_usm_pool_handle_t Pool = hContext->getOwningURPool(UMFPool);
      if (!Pool) {
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
      return ReturnValue(Pool);
    }
    default:
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(ur_context_handle_t Context,
                                                   void *HostPtr, size_t Size) {
  UR_ASSERT(Context, UR_RESULT_ERROR_INVALID_CONTEXT);
  UR_ASSERT(!HostPtr, UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(Size > 0, UR_RESULT_ERROR_INVALID_VALUE);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMReleaseExp(ur_context_handle_t Context,
                                                    void *HostPtr) {
  UR_ASSERT(Context, UR_RESULT_ERROR_INVALID_CONTEXT);
  UR_ASSERT(!HostPtr, UR_RESULT_ERROR_INVALID_VALUE);
  return UR_RESULT_SUCCESS;
}

umf_result_t USMMemoryProvider::initialize(ur_context_handle_t Ctx,
                                           ur_device_handle_t Dev) {
  Context = Ctx;
  Device = Dev;
  // There isn't a way to query this in cuda, and there isn't much info on
  // cuda's approach to alignment or transfer granularity between host and
  // device. Within UMF this is only used to influence alignment, and since we
  // discard that in our alloc implementations it seems we can safely ignore
  // this as well, for now.
  MinPageSize = 0;

  return UMF_RESULT_SUCCESS;
}

enum umf_result_t USMMemoryProvider::alloc(size_t Size, size_t Align,
                                           void **Ptr) {
  auto Res = allocateImpl(Ptr, Size, Align);
  if (Res != UR_RESULT_SUCCESS) {
    getLastStatusRef() = Res;
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  return UMF_RESULT_SUCCESS;
}

enum umf_result_t USMMemoryProvider::free(void *Ptr, size_t Size) {
  (void)Size;

  auto Res = USMFreeImpl(Context, Ptr);
  if (Res != UR_RESULT_SUCCESS) {
    getLastStatusRef() = Res;
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  return UMF_RESULT_SUCCESS;
}

void USMMemoryProvider::get_last_native_error(const char **ErrMsg,
                                              int32_t *ErrCode) {
  (void)ErrMsg;
  *ErrCode = static_cast<int32_t>(getLastStatusRef());
}

umf_result_t USMMemoryProvider::get_min_page_size(void *Ptr, size_t *PageSize) {
  (void)Ptr;
  *PageSize = MinPageSize;

  return UMF_RESULT_SUCCESS;
}

ur_result_t USMSharedMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                  uint32_t Alignment) {
  return USMSharedAllocImpl(ResultPtr, Context, Device, /*host flags*/ 0,
                            /*device flags*/ 0, Size, Alignment);
}

ur_result_t USMDeviceMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                  uint32_t Alignment) {
  return USMDeviceAllocImpl(ResultPtr, Context, Device, /* flags */ 0, Size,
                            Alignment);
}

ur_result_t USMHostMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                uint32_t Alignment) {
  return USMHostAllocImpl(ResultPtr, Context, /* flags */ 0, Size, Alignment);
}

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t Context,
                                             ur_usm_pool_desc_t *PoolDesc)
    : Context{Context} {
  const void *pNext = PoolDesc->pNext;
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc = static_cast<const ur_base_desc_t *>(pNext);
    switch (BaseDesc->stype) {
    case UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC: {
      const ur_usm_pool_limits_desc_t *Limits =
          reinterpret_cast<const ur_usm_pool_limits_desc_t *>(BaseDesc);
      for (auto &config : DisjointPoolConfigs.Configs) {
        config.MaxPoolableSize = Limits->maxPoolableSize;
        config.SlabMinSize = Limits->minDriverAllocSize;
      }
      break;
    }
    default: {
      throw UsmAllocationException(UR_RESULT_ERROR_INVALID_ARGUMENT);
    }
    }
    pNext = BaseDesc->pNext;
  }

  auto MemProvider =
      umf::memoryProviderMakeUnique<USMHostMemoryProvider>(Context, nullptr)
          .second;

  auto UmfHostParamsHandle = getUmfParamsHandle(
      DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Host]);
  HostMemPool =
      umf::poolMakeUniqueFromOps(umfDisjointPoolOps(), std::move(MemProvider),
                                 UmfHostParamsHandle.get())
          .second;

  for (const auto &Device : Context->getDevices()) {
    MemProvider =
        umf::memoryProviderMakeUnique<USMDeviceMemoryProvider>(Context, Device)
            .second;
    auto UmfDeviceParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Device]);
    DeviceMemPool =
        umf::poolMakeUniqueFromOps(umfDisjointPoolOps(), std::move(MemProvider),
                                   UmfDeviceParamsHandle.get())
            .second;
    MemProvider =
        umf::memoryProviderMakeUnique<USMSharedMemoryProvider>(Context, Device)
            .second;
    auto UmfSharedParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Shared]);
    SharedMemPool =
        umf::poolMakeUniqueFromOps(umfDisjointPoolOps(), std::move(MemProvider),
                                   UmfSharedParamsHandle.get())
            .second;
    Context->addPool(this);
  }
}

bool ur_usm_pool_handle_t_::hasUMFPool(umf_memory_pool_t *umf_pool) {
  return DeviceMemPool.get() == umf_pool || SharedMemPool.get() == umf_pool ||
         HostMemPool.get() == umf_pool;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolCreate(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in] pointer to USM pool descriptor. Can be chained with
    /// ::ur_usm_pool_limits_desc_t
    ur_usm_pool_desc_t *PoolDesc,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *Pool) {
  // Without pool tracking we can't free pool allocations.
#ifdef UMF_ENABLE_POOL_TRACKING
  if (PoolDesc->flags & UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
  try {
    *Pool = reinterpret_cast<ur_usm_pool_handle_t>(
        new ur_usm_pool_handle_t_(Context, PoolDesc));
  } catch (const UsmAllocationException &Ex) {
    return Ex.getError();
  } catch (umf_result_t e) {
    return umf::umf2urResult(e);
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
#else
  std::ignore = Context;
  std::ignore = PoolDesc;
  std::ignore = Pool;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolRetain(
    /// [in] pointer to USM memory pool
    ur_usm_pool_handle_t Pool) {
  Pool->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolRelease(
    /// [in] pointer to USM memory pool
    ur_usm_pool_handle_t Pool) {
  if (Pool->decrementReferenceCount() > 0) {
    return UR_RESULT_SUCCESS;
  }
  Pool->Context->removePool(Pool);
  delete Pool;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetInfo(
    /// [in] handle of the USM memory pool
    ur_usm_pool_handle_t hPool,
    /// [in] name of the pool property to query
    ur_usm_pool_info_t propName,
    /// [in] size in bytes of the pool property value provided
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] value of the pool property
    void *pPropValue,
    /// [out][optional] size in bytes returned in pool property value
    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_USM_POOL_INFO_REFERENCE_COUNT: {
    return ReturnValue(hPool->getReferenceCount());
  }
  case UR_USM_POOL_INFO_CONTEXT: {
    return ReturnValue(hPool->Context);
  }
  default: {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  }
}
