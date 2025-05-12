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

/// USM: Implements USM Host allocations using CUDA Pinned Memory
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory
UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t hPool, size_t size, void **ppMem) {
  auto alignment = pUSMDesc ? pUSMDesc->align : 0u;

  auto pool = hPool ? hPool->HostMemPool.get() : hContext->MemoryPoolHost;
  if (alignment) {
    UR_ASSERT(isPowerOf2(alignment), UR_RESULT_ERROR_INVALID_VALUE);
    *ppMem = umfPoolAlignedMalloc(pool, size, alignment);
  } else {
    *ppMem = umfPoolMalloc(pool, size);
  }

  if (*ppMem == nullptr) {
    auto umfErr = umfPoolGetLastAllocationError(pool);
    return umf::umf2urResult(umfErr);
  }
  return UR_RESULT_SUCCESS;
}

/// USM: Implements USM device allocations using a normal CUDA device pointer
///
UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t hPool,
                 size_t size, void **ppMem) {
  auto alignment = pUSMDesc ? pUSMDesc->align : 0u;

  ScopedContext SC(hDevice);
  auto pool = hPool ? hPool->DeviceMemPool.get() : hDevice->MemoryPoolDevice;
  if (alignment) {
    UR_ASSERT(isPowerOf2(alignment), UR_RESULT_ERROR_INVALID_VALUE);
    *ppMem = umfPoolAlignedMalloc(pool, size, alignment);
  } else {
    *ppMem = umfPoolMalloc(pool, size);
  }

  if (*ppMem == nullptr) {
    auto umfErr = umfPoolGetLastAllocationError(pool);
    return umf::umf2urResult(umfErr);
  }
  return UR_RESULT_SUCCESS;
}

/// USM: Implements USM Shared allocations using CUDA Managed Memory
///
UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t hPool,
                 size_t size, void **ppMem) {
  auto alignment = pUSMDesc ? pUSMDesc->align : 0u;

  ScopedContext SC(hDevice);
  auto pool = hPool ? hPool->SharedMemPool.get() : hDevice->MemoryPoolShared;
  if (alignment) {
    UR_ASSERT(isPowerOf2(alignment), UR_RESULT_ERROR_INVALID_VALUE);
    *ppMem = umfPoolAlignedMalloc(pool, size, alignment);
  } else {
    *ppMem = umfPoolMalloc(pool, size);
  }

  if (*ppMem == nullptr) {
    auto umfErr = umfPoolGetLastAllocationError(pool);
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
      ur_adapter_handle_t AdapterHandle = ur::cuda::adapter;
      Result = urPlatformGet(AdapterHandle, 1, &platform, nullptr);

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

UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(ur_context_handle_t, void *,
                                                   size_t Size) {
  UR_ASSERT(Size > 0, UR_RESULT_ERROR_INVALID_VALUE);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMReleaseExp(ur_context_handle_t,
                                                    void *) {
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
      throw UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    }
    pNext = BaseDesc->pNext;
  }

  auto UmfHostParamsHandle = getUmfParamsHandle(
      DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Host]);
  HostMemPool = umf::poolMakeUniqueFromOpsProviderHandle(
                    umfDisjointPoolOps(), Context->MemoryProviderHost,
                    UmfHostParamsHandle.get())
                    .second;

  for (const auto &Device : Context->getDevices()) {
    auto UmfDeviceParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Device]);
    DeviceMemPool = umf::poolMakeUniqueFromOpsProviderHandle(
                        umfDisjointPoolOps(), Device->MemoryProviderDevice,
                        UmfDeviceParamsHandle.get())
                        .second;

    auto UmfSharedParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Shared]);
    SharedMemPool = umf::poolMakeUniqueFromOpsProviderHandle(
                        umfDisjointPoolOps(), Device->MemoryProviderShared,
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
  if (PoolDesc->flags & UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
  try {
    *Pool = reinterpret_cast<ur_usm_pool_handle_t>(
        new ur_usm_pool_handle_t_(Context, PoolDesc));
  } catch (ur_result_t e) {
    return e;
  } catch (umf_result_t e) {
    return umf::umf2urResult(e);
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
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

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t Context,
                                             ur_device_handle_t Device,
                                             ur_usm_pool_desc_t *PoolDesc)
    : Context{Context}, Device{Device} {
  if (!(PoolDesc->flags & UR_USM_POOL_FLAG_USE_NATIVE_MEMORY_POOL_EXP))
    throw UR_RESULT_ERROR_INVALID_ARGUMENT;

  CUmemPoolProps MemPoolProps{};
  size_t threshold = 0;

  const void *pNext = PoolDesc->pNext;
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc = static_cast<const ur_base_desc_t *>(pNext);
    switch (BaseDesc->stype) {
    case UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC: {
      const ur_usm_pool_limits_desc_t *Limits =
          reinterpret_cast<const ur_usm_pool_limits_desc_t *>(BaseDesc);
#if CUDA_VERSION >= 12020
      // maxSize as a member of CUmemPoolProps was introduced in CUDA 12.2.
      MemPoolProps.maxSize =
          Limits->maxPoolableSize; // CUDA lazily reserves memory for pools in
                                   // 32MB chunks. maxSize is elevated to the
                                   // next 32MB multiple. Each 32MB chunk is
                                   // only reserved when it's needed for the
                                   // first time (cuMemAllocFromPoolAsync).
#else
      // Only warn if the user set a value >0 for the maximum size.
      // Otherwise, do nothing.
      // Set maximum size is effectively ignored.
      if (Limits->maxPoolableSize > 0)
        UR_LOG(WARN, "The memory pool maximum size feature requires CUDA "
                     "12.2 or later.\n");
#endif
      maxSize = Limits->maxPoolableSize;
      size_t chunkSize = 33554432; // 32MB
      size_t remainder = Limits->maxPoolableSize % chunkSize;
      if (remainder != 0) {
        maxSize = maxSize + chunkSize - remainder;
      }

      threshold = Limits->minDriverAllocSize;
      break;
    }
    default: {
      throw UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    }
    pNext = BaseDesc->pNext;
  }

  MemPoolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  // Clarification of what id means here:
  // https://forums.developer.nvidia.com/t/incomplete-description-in-cumemlocation-v1-struct-reference/318701
  MemPoolProps.location.id = Device->getIndex();
  MemPoolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  UR_CHECK_ERROR(cuMemPoolCreate(&CUmemPool, &MemPoolProps));

  // Release threshold is not a property when creating a pool.
  // It must be set separately.
  UR_CHECK_ERROR(urUSMPoolSetInfoExp(this,
                                     UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP,
                                     &threshold, 8 /*uint64_t*/));
}

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t Context,
                                             ur_device_handle_t Device,
                                             CUmemoryPool CUmemPool)
    : Context{Context}, Device{Device}, CUmemPool(CUmemPool) {}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolCreateExp(ur_context_handle_t Context, ur_device_handle_t Device,
                   ur_usm_pool_desc_t *pPoolDesc, ur_usm_pool_handle_t *pPool) {
  // This entry point only supports native mem pools.
  if (!(pPoolDesc->flags & UR_USM_POOL_FLAG_USE_NATIVE_MEMORY_POOL_EXP))
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  // Zero-init is on by default in CUDA.
  // Read-only has no support in CUDA.
  try {
    *pPool = reinterpret_cast<ur_usm_pool_handle_t>(
        new ur_usm_pool_handle_t_(Context, Device, pPoolDesc));
  } catch (ur_result_t e) {
    return e;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolDestroyExp(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                    ur_usm_pool_handle_t hPool) {

  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  ScopedContext Active(hDevice);

  try {
    UR_CHECK_ERROR(cuMemPoolDestroy(hPool->getCudaPool()));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_usm_pool_handle_t *pPool) {

  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  ScopedContext Active(hDevice);

  try {
    CUmemoryPool cuPool;
    UR_CHECK_ERROR(cuDeviceGetDefaultMemPool(&cuPool, hDevice->get()));

    *pPool = reinterpret_cast<ur_usm_pool_handle_t>(
        new ur_usm_pool_handle_t_(hContext, hDevice, cuPool));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolGetInfoExp(ur_usm_pool_handle_t hPool, ur_usm_pool_info_t propName,
                    void *pPropValue, size_t *pPropSizeRet) {

  CUmemPool_attribute attr;

  switch (propName) {
  case UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP:
    attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
    break;
  case UR_USM_POOL_INFO_RESERVED_CURRENT_EXP:
    attr = CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT;
    break;
  case UR_USM_POOL_INFO_RESERVED_HIGH_EXP:
    attr = CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH;
    break;
  case UR_USM_POOL_INFO_USED_CURRENT_EXP:
    attr = CU_MEMPOOL_ATTR_USED_MEM_CURRENT;
    break;
  case UR_USM_POOL_INFO_USED_HIGH_EXP:
    attr = CU_MEMPOOL_ATTR_USED_MEM_HIGH;
    break;
  default:
    // Unknown enumerator
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  uint64_t value = 0;
  UR_CHECK_ERROR(
      cuMemPoolGetAttribute(hPool->getCudaPool(), attr, (void *)&value));

  if (pPropValue) {
    *(size_t *)pPropValue = value;
  }
  if (pPropSizeRet) {
    *(size_t *)pPropSizeRet = sizeof(size_t);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolSetInfoExp(ur_usm_pool_handle_t hPool, ur_usm_pool_info_t propName,
                    void *pPropValue, size_t) {

  CUmemPool_attribute attr;

  // All current values are expected to be of size uint64_t
  switch (propName) {
  case UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP:
    attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
    break;
  case UR_USM_POOL_INFO_RESERVED_HIGH_EXP:
    attr = CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH;
    break;
  case UR_USM_POOL_INFO_USED_HIGH_EXP:
    attr = CU_MEMPOOL_ATTR_USED_MEM_HIGH;
    break;
  default:
    // Unknown enumerator
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  try {
    UR_CHECK_ERROR(
        cuMemPoolSetAttribute(hPool->getCudaPool(), attr, pPropValue));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolTrimToExp(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                   ur_usm_pool_handle_t hPool, size_t minBytesToKeep) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  ScopedContext Active(hDevice);

  try {
    UR_CHECK_ERROR(cuMemPoolTrimTo(hPool->getCudaPool(), minBytesToKeep));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}
