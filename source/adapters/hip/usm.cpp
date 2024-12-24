//===--------- usm.cpp - HIP Adapter --------------------------------------===//
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
#include "platform.hpp"
#include "ur_util.hpp"
#include "usm.hpp"

namespace umf {
ur_result_t getProviderNativeError(const char *, int32_t) {
  // TODO: implement when UMF supports HIP
  return UR_RESULT_ERROR_UNKNOWN;
}
} // namespace umf

/// USM: Implements USM Host allocations using HIP Pinned Memory
UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t hPool, size_t size, void **ppMem) {
  uint32_t alignment;
  UR_ASSERT(checkUSMAlignment(alignment, pUSMDesc),
            UR_RESULT_ERROR_INVALID_VALUE);

  if (!hPool) {
    return USMHostAllocImpl(ppMem, hContext, /* flags */ 0, size, alignment);
  }

  return umfPoolMallocHelper(hPool, ppMem, size, alignment);
}

/// USM: Implements USM device allocations using a normal HIP device pointer
UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t hPool,
                 size_t size, void **ppMem) {
  uint32_t alignment;
  UR_ASSERT(checkUSMAlignment(alignment, pUSMDesc),
            UR_RESULT_ERROR_INVALID_VALUE);

  if (!hPool) {
    return USMDeviceAllocImpl(ppMem, hContext, hDevice, /* flags */ 0, size,
                              alignment);
  }

  return umfPoolMallocHelper(hPool, ppMem, size, alignment);
}

/// USM: Implements USM Shared allocations using HIP Managed Memory
UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t hPool,
                 size_t size, void **ppMem) {
  uint32_t alignment;
  UR_ASSERT(checkUSMAlignment(alignment, pUSMDesc),
            UR_RESULT_ERROR_INVALID_VALUE);

  if (!hPool) {
    return USMSharedAllocImpl(ppMem, hContext, hDevice, /*host flags*/ 0,
                              /*device flags*/ 0, size, alignment);
  }

  return umfPoolMallocHelper(hPool, ppMem, size, alignment);
}

UR_APIEXPORT ur_result_t UR_APICALL
USMFreeImpl([[maybe_unused]] ur_context_handle_t hContext, void *pMem) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    hipPointerAttribute_t hipPointerAttributeType;
    UR_CHECK_ERROR(hipPointerGetAttributes(&hipPointerAttributeType, pMem));
#if HIP_VERSION >= 50600000
    const auto Type = hipPointerAttributeType.type;
#else
    const auto Type = hipPointerAttributeType.memoryType;
#endif
    UR_ASSERT(Type == hipMemoryTypeDevice || Type == hipMemoryTypeHost ||
                  Type == hipMemoryTypeManaged,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);
    if (Type == hipMemoryTypeDevice || Type == hipMemoryTypeManaged) {
      UR_CHECK_ERROR(hipFree(pMem));
    }
    if (Type == hipMemoryTypeHost) {
      UR_CHECK_ERROR(hipHostFree(pMem));
    }
  } catch (ur_result_t Error) {
    Result = Error;
  }
  return Result;
}

/// USM: Frees the given USM pointer associated with the context.
UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  if (auto Pool = umfPoolByPtr(pMem)) {
    return umf::umf2urResult(umfPoolFree(Pool, pMem));
  } else {
    return USMFreeImpl(hContext, pMem);
  }
}

ur_result_t USMDeviceAllocImpl(void **ResultPtr, ur_context_handle_t,
                               ur_device_handle_t Device,
                               ur_usm_device_mem_flags_t, size_t Size,
                               [[maybe_unused]] uint32_t Alignment) {
  try {
    ScopedDevice Active(Device);
    UR_CHECK_ERROR(hipMalloc(ResultPtr, Size));
  } catch (ur_result_t Err) {
    return Err;
  }

  assert(checkUSMImplAlignment(Alignment, ResultPtr));
  return UR_RESULT_SUCCESS;
}

ur_result_t USMSharedAllocImpl(void **ResultPtr, ur_context_handle_t,
                               ur_device_handle_t Device,
                               ur_usm_host_mem_flags_t,
                               ur_usm_device_mem_flags_t, size_t Size,
                               [[maybe_unused]] uint32_t Alignment) {
  try {
    ScopedDevice Active(Device);
    UR_CHECK_ERROR(hipMallocManaged(ResultPtr, Size, hipMemAttachGlobal));
  } catch (ur_result_t Err) {
    return Err;
  }

  assert(checkUSMImplAlignment(Alignment, ResultPtr));
  return UR_RESULT_SUCCESS;
}

ur_result_t USMHostAllocImpl(void **ResultPtr,
                             [[maybe_unused]] ur_context_handle_t Context,
                             ur_usm_host_mem_flags_t, size_t Size,
                             [[maybe_unused]] uint32_t Alignment) {
  try {
    UR_CHECK_ERROR(hipHostMalloc(ResultPtr, Size));
  } catch (ur_result_t Err) {
    return Err;
  }

  assert(checkUSMImplAlignment(Alignment, ResultPtr));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propValueSize,
                     void *pPropValue, size_t *pPropValueSizeRet) {
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  try {
    switch (propName) {
    case UR_USM_ALLOC_INFO_TYPE: {
      auto MaybePointerAttrs = getPointerAttributes(pMem);
      if (!MaybePointerAttrs.has_value()) {
        // pointer not known to the HIP subsystem
        return ReturnValue(UR_USM_TYPE_UNKNOWN);
      }
      auto Value = getMemoryType(*MaybePointerAttrs);
      UR_ASSERT(Value == hipMemoryTypeDevice || Value == hipMemoryTypeHost ||
                    Value == hipMemoryTypeManaged,
                UR_RESULT_ERROR_INVALID_MEM_OBJECT);
      if (MaybePointerAttrs->isManaged || Value == hipMemoryTypeManaged) {
        // pointer to managed memory
        return ReturnValue(UR_USM_TYPE_SHARED);
      }
      if (Value == hipMemoryTypeDevice) {
        // pointer to device memory
        return ReturnValue(UR_USM_TYPE_DEVICE);
      }
      if (Value == hipMemoryTypeHost) {
        // pointer to host memory
        return ReturnValue(UR_USM_TYPE_HOST);
      }
      // should never get here
      ur::unreachable();
    }
    case UR_USM_ALLOC_INFO_DEVICE: {
      auto MaybePointerAttrs = getPointerAttributes(pMem);
      if (!MaybePointerAttrs.has_value()) {
        // pointer not known to the HIP subsystem
        return ReturnValue(UR_USM_TYPE_UNKNOWN);
      }

      int DeviceIdx = MaybePointerAttrs->device;

      // hip backend has only one platform containing all devices
      ur_platform_handle_t platform;
      ur_adapter_handle_t AdapterHandle = &adapter;
      UR_CHECK_ERROR(urPlatformGet(&AdapterHandle, 1, 1, &platform, nullptr));

      // get the device from the platform
      ur_device_handle_t Device = platform->Devices[DeviceIdx].get();
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
    case UR_USM_ALLOC_INFO_BASE_PTR:
      // HIP gives us the ability to query the base pointer for a device
      // pointer, so check whether we've got one of those.
      if (auto MaybePointerAttrs = getPointerAttributes(pMem)) {
        if (getMemoryType(*MaybePointerAttrs) == hipMemoryTypeDevice) {
          void *Base = nullptr;
          UR_CHECK_ERROR(hipPointerGetAttribute(
              &Base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
              reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(pMem))));
          return ReturnValue(Base);
        }
      }
      // If not, we can't be sure.
      return UR_RESULT_ERROR_INVALID_VALUE;
    case UR_USM_ALLOC_INFO_SIZE: {
      size_t RangeSize = 0;
      UR_CHECK_ERROR(hipMemPtrGetInfo(const_cast<void *>(pMem), &RangeSize));
      return ReturnValue(RangeSize);
    }
    default:
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
  } catch (ur_result_t Error) {
    return Error;
  }
  return UR_RESULT_SUCCESS;
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
    : Context(Context) {
  if (PoolDesc) {
    if (auto *Limits = find_stype_node<ur_usm_pool_limits_desc_t>(PoolDesc)) {
      for (auto &config : DisjointPoolConfigs.Configs) {
        config.MaxPoolableSize = Limits->maxPoolableSize;
        config.SlabMinSize = Limits->minDriverAllocSize;
      }
    } else {
      throw UsmAllocationException(UR_RESULT_ERROR_INVALID_ARGUMENT);
    }
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
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_usm_pool_desc_t
        *PoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *Pool ///< [out] pointer to USM memory pool
) {
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
    ur_usm_pool_handle_t Pool ///< [in] pointer to USM memory pool
) {
  Pool->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolRelease(
    ur_usm_pool_handle_t Pool ///< [in] pointer to USM memory pool
) {
  if (Pool->decrementReferenceCount() > 0) {
    return UR_RESULT_SUCCESS;
  }
  Pool->Context->removePool(Pool);
  delete Pool;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetInfo(
    ur_usm_pool_handle_t hPool,  ///< [in] handle of the USM memory pool
    ur_usm_pool_info_t propName, ///< [in] name of the pool property to query
    size_t propSize, ///< [in] size in bytes of the pool property value provided
    void *pPropValue, ///< [out][optional][typename(propName, propSize)] value
                      ///< of the pool property
    size_t *pPropSizeRet ///< [out][optional] size in bytes returned in pool
                         ///< property value
) {
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

bool checkUSMAlignment(uint32_t &alignment, const ur_usm_desc_t *pUSMDesc) {
  alignment = pUSMDesc ? pUSMDesc->align : 0u;
  return (!pUSMDesc ||
          (alignment == 0 || ((alignment & (alignment - 1)) == 0)));
}

bool checkUSMImplAlignment(uint32_t Alignment, void **ResultPtr) {
  return Alignment == 0 ||
         reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0;
}

ur_result_t umfPoolMallocHelper(ur_usm_pool_handle_t hPool, void **ppMem,
                                size_t size, uint32_t alignment) {
  auto UMFPool = hPool->DeviceMemPool.get();
  *ppMem = umfPoolAlignedMalloc(UMFPool, size, alignment);
  if (*ppMem == nullptr) {
    auto umfErr = umfPoolGetLastAllocationError(UMFPool);
    return umf::umf2urResult(umfErr);
  }
  return UR_RESULT_SUCCESS;
}
