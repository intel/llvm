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
#include "logger/ur_logger.hpp"
#include "platform.hpp"
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
    ur_usm_host_mem_flags_t flags = 0;
    if (pUSMDesc) {
      if (const auto *p = find_stype_node<ur_usm_host_desc_t>(pUSMDesc)) {
        flags = p->flags;
      }
    }
    return USMHostAllocImpl(ppMem, hContext, flags, size, alignment);
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
    ur_usm_device_mem_flags_t flags = 0;
    if (pUSMDesc) {
      if (const auto *p = find_stype_node<ur_usm_device_desc_t>(pUSMDesc)) {
        flags = p->flags;
      }
    }
    return USMDeviceAllocImpl(ppMem, hContext, hDevice, flags, size, alignment);
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

/// USM: Frees the given USM pointer associated with the context.
///
UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  (void)hContext; // unused
  return umf::umf2urResult(umfFree(pMem));
}

ur_result_t USMDeviceAllocImpl(void **ResultPtr, ur_context_handle_t,
                               ur_device_handle_t Device,
                               ur_usm_device_mem_flags_t, size_t Size,
                               uint32_t Alignment) {
  try {
    ScopedContext Active(Device);
    *ResultPtr = umfPoolMalloc(Device->getMemoryPoolDevice(), Size);
    UMF_CHECK_PTR(*ResultPtr);
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
    *ResultPtr = umfPoolMalloc(Device->getMemoryPoolShared(), Size);
    UMF_CHECK_PTR(*ResultPtr);
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
                             ur_usm_host_mem_flags_t Flags, size_t Size,
                             uint32_t Alignment) {
  try {
    unsigned CF = 0;
    // HEREBEDRAGONS The various extant CUDA docs are a contradictory mess
    // regarding the how flags behave and on which kind of hardware the
    // documented functionality is available.
    //
    // First:
    //
    // `CU_MEMHOSTALLOC_PORTABLE`:
    //
    // https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/html/group__CUDA__UNIFIED.html
    // says the following:
    //
    // > All host memory allocated in all contexts using cuMemAllocHost() and
    // > cuMemHostAlloc() is always directly accessible from all contexts on all
    // > devices that support unified addressing. This is the case regardless of
    // > whether or not the flags CU_MEMHOSTALLOC_PORTABLE and
    // > CU_MEMHOSTALLOC_DEVICEMAP are specified.
    //
    // If it's always available in all contexts, then Why does this flag exist
    // as a valid option for `cuMemHostAlloc`, at all? Is this a legacy thing?
    //
    // In any case we don't define an equivalent at this time, so we don't
    // bother setting the `CU_MEMHOSTALLOC_PORTABLE` bit.

    // Next we've got the lovely
    //
    // `CU_MEMHOSTALLOC_DEVICEMAP`.
    //
    // > CU_MEMHOSTALLOC_DEVICEMAP: Maps the allocation into the CUDA address
    // > space. The device pointer to the memory may be obtained by calling
    // > ::cuMemHostGetDevicePointer().
    //
    //  Not sure what "maps the allocation into the CUDA address space" means
    //  since the documentation linked above reports that it's always directly
    //  accessible from all contexts on all devices that support unified
    //  addressing. If that's the case, then do we need it? The memory is
    //  "always accessible from all devices and contexts"! As far as I can
    //  understand: no; not needed. Perhaps this is a legacy thing from
    //  pre-pascal era cards?
    //
    //  However, after all that nonsense, some features actually seem to require
    //  it such as for example Bindless images features fail in certain
    //  circumstances without it being set...Thus to make our default
    //  implementation generally usabable we actually need to set this by
    //  default, seeing as it apparently corrects some edge cases which show
    //  that the "accessible from all contexts and devices" wording is untrue in
    //  practise.
    CF |= CU_MEMHOSTALLOC_DEVICEMAP;

    // https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/html/group__CUDA__MEM_g572ca4011bfcb25034888a14d4e035b9.html
    //
    // > If the flag CU_MEMHOSTALLOC_WRITECOMBINED is specified, then the
    // > function cuMemHostGetDevicePointer() must be used to query the device
    // > pointer, even if the context supports unified addressing. See Unified
    // > Addressing for additional details.
    //
    // and from the CUDA header:
    //
    // > CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
    // > Note all host memory allocated using ::cuMemHostAlloc() will
    // > automatically be immediately accessible to all contexts on all devices
    // > which support unified addressing (as may be queried using
    // > ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING). Unless the flag
    // > ::CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device pointer that
    // > may be used to access this host memory from those contexts is always
    // > equal to the returned host pointer \p *pp.  If the flag
    // > ::CU_MEMHOSTALLOC_WRITECOMBINED is specified, then the function
    // > ::cuMemHostGetDevicePointer() must be used to query the device pointer,
    // > even if the context supports unified addressing. See \ref CUDA_UNIFIED
    // > for additional details.
    //
    // Since our version of USM requires the same value for host and device
    // pointers we need to catch this case and rollback if it's not supported.
    // There appears to be no way to query this ahead of time since - as
    // mentioned above - checking for `Unified Addressing` via the
    // `CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING` attribute is not sufficient when
    // write-combining is enabled.
    CF |= CU_MEMHOSTALLOC_WRITECOMBINED *
          !!(Flags & UR_USM_HOST_MEM_FLAG_WRITE_COMBINE);
    umf_memory_pool_t *Pool = hContext->getMemoryPoolHost(CF);
    *ResultPtr = umfPoolAlignedMalloc(Pool, Size, Alignment);
    if (!*ResultPtr) {
      auto E = umfPoolGetLastAllocationError(Pool);
      return umf::umf2urResult(E);
    }
    // Then check the pointers' have the same value or roll back the flag and
    // return a normal allocation
    if (CF & UR_USM_HOST_MEM_FLAG_WRITE_COMBINE) {
      CUdeviceptr DevPtr;
      if (cuMemHostGetDevicePointer(&DevPtr, ResultPtr,
                                    /*Flags must be 0*/ 0) != CUDA_SUCCESS ||
          ((void *)DevPtr != *ResultPtr)) {

        // This is allowed because the flags are optional
        logger::warning(
            "got different host and device pointers for "
            "write-combining host allocation:(host:%p, device %p). Masking",
            *ResultPtr, DevPtr);
        CF &= ~CU_MEMHOSTALLOC_WRITECOMBINED;
        Pool = hContext->getMemoryPoolHost(CF);
        if (!*ResultPtr) {
          auto E = umfPoolGetLastAllocationError(Pool);
          return umf::umf2urResult(E);
        }
      }
    }
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


  umf_memory_provider_t *HostProvider = nullptr;
  umfPoolGetMemoryProvider(Context->getMemoryPoolHost(), &HostProvider);
  auto UmfHostParamsHandle = getUmfParamsHandle(
      DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Host]);
  HostMemPool = umf::poolMakeUniqueFromOpsProviderHandle(
                    umfDisjointPoolOps(), HostProvider,
                    UmfHostParamsHandle.get())
                    .second;

  for (const auto &Device : Context->getDevices()) {
    umf_memory_provider_t *DeviceProvider = nullptr;
    umfPoolGetMemoryProvider(Device->getMemoryPoolDevice(), &DeviceProvider);
    auto UmfDeviceParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Device]);
    DeviceMemPool = umf::poolMakeUniqueFromOpsProviderHandle(
                        umfDisjointPoolOps(), DeviceProvider,
                        UmfDeviceParamsHandle.get())
                        .second;

    umf_memory_provider_t *SharedProvider = nullptr;
    umfPoolGetMemoryProvider(Device->getMemoryPoolShared(), &SharedProvider);
    auto UmfSharedParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Shared]);
    SharedMemPool = umf::poolMakeUniqueFromOpsProviderHandle(
                        umfDisjointPoolOps(), SharedProvider,
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

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolCreateExp(ur_context_handle_t,
                                                       ur_device_handle_t,
                                                       ur_usm_pool_desc_t *,
                                                       ur_usm_pool_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolDestroyExp(ur_context_handle_t,
                                                        ur_device_handle_t,
                                                        ur_usm_pool_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolSetThresholdExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t, size_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetInfoExp(ur_usm_pool_handle_t,
                                                        ur_usm_pool_info_t,
                                                        void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolTrimToExp(ur_context_handle_t,
                                                       ur_device_handle_t,
                                                       ur_usm_pool_handle_t,
                                                       size_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
