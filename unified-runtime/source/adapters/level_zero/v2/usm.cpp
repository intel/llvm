//===--------- usm.cpp - Level Zero Adapter ------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

#include "../device.hpp"
#include "context.hpp"
#include "umf_pools/disjoint_pool_config_parser.hpp"
#include "usm.hpp"

#include <umf/providers/provider_level_zero.h>

static inline void UMF_CALL_THROWS(umf_result_t res) {
  if (res != UMF_RESULT_SUCCESS) {
    throw res;
  }
}

namespace umf {
ur_result_t getProviderNativeError(const char *providerName,
                                   int32_t nativeError) {
  if (strcmp(providerName, "LEVEL_ZERO") == 0) {
    auto zeResult = static_cast<ze_result_t>(nativeError);
    if (zeResult == ZE_RESULT_ERROR_UNSUPPORTED_SIZE) {
      return UR_RESULT_ERROR_INVALID_USM_SIZE;
    }
    return ze2urResult(zeResult);
  }

  return UR_RESULT_ERROR_UNKNOWN;
}
} // namespace umf

static std::optional<usm::DisjointPoolAllConfigs>
initializeDisjointPoolConfig() {
  const char *UrRetDisable = std::getenv("UR_L0_DISABLE_USM_ALLOCATOR");
  const char *PiRetDisable =
      std::getenv("SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR");
  const char *Disable =
      UrRetDisable ? UrRetDisable : (PiRetDisable ? PiRetDisable : nullptr);
  if (Disable != nullptr && Disable != std::string("")) {
    return std::nullopt;
  }

  const char *PoolUrTraceVal = std::getenv("UR_L0_USM_ALLOCATOR_TRACE");

  int PoolTrace = 0;
  if (PoolUrTraceVal != nullptr) {
    PoolTrace = std::atoi(PoolUrTraceVal);
  }

  const char *PoolUrConfigVal = std::getenv("UR_L0_USM_ALLOCATOR");
  if (PoolUrConfigVal == nullptr) {
    return usm::DisjointPoolAllConfigs(PoolTrace);
  }

  // TODO: rework parseDisjointPoolConfig to return optional,
  // once EnableBuffers is no longer used (by legacy L0)
  auto configs = usm::parseDisjointPoolConfig(PoolUrConfigVal, PoolTrace);
  if (configs.EnableBuffers) {
    return configs;
  }

  return std::nullopt;
}

inline umf_usm_memory_type_t urToUmfMemoryType(ur_usm_type_t type) {
  switch (type) {
  case UR_USM_TYPE_DEVICE:
    return UMF_MEMORY_TYPE_DEVICE;
  case UR_USM_TYPE_SHARED:
    return UMF_MEMORY_TYPE_SHARED;
  case UR_USM_TYPE_HOST:
    return UMF_MEMORY_TYPE_HOST;
  default:
    throw UR_RESULT_ERROR_INVALID_ARGUMENT;
  }
}

static usm::DisjointPoolMemType
descToDisjoinPoolMemType(const usm::pool_descriptor &desc) {
  switch (desc.type) {
  case UR_USM_TYPE_DEVICE:
    return usm::DisjointPoolMemType::Device;
  case UR_USM_TYPE_SHARED: {
    if (desc.deviceReadOnly)
      return usm::DisjointPoolMemType::SharedReadOnly;
    else
      return usm::DisjointPoolMemType::Shared;
  }
  case UR_USM_TYPE_HOST:
    return usm::DisjointPoolMemType::Host;
  default:
    throw UR_RESULT_ERROR_INVALID_ARGUMENT;
  }
}

static umf::provider_unique_handle_t
makeProvider(usm::pool_descriptor poolDescriptor) {
  umf_level_zero_memory_provider_params_handle_t hParams;
  UMF_CALL_THROWS(umfLevelZeroMemoryProviderParamsCreate(&hParams));
  std::unique_ptr<umf_level_zero_memory_provider_params_t,
                  decltype(&umfLevelZeroMemoryProviderParamsDestroy)>
      params(hParams, &umfLevelZeroMemoryProviderParamsDestroy);

  UMF_CALL_THROWS(umfLevelZeroMemoryProviderParamsSetContext(
      hParams, poolDescriptor.hContext->getZeHandle()));

  ze_device_handle_t level_zero_device_handle =
      poolDescriptor.hDevice ? poolDescriptor.hDevice->ZeDevice : nullptr;

  UMF_CALL_THROWS(umfLevelZeroMemoryProviderParamsSetDevice(
      hParams, level_zero_device_handle));
  UMF_CALL_THROWS(umfLevelZeroMemoryProviderParamsSetMemoryType(
      hParams, urToUmfMemoryType(poolDescriptor.type)));

  std::vector<ze_device_handle_t> residentZeHandles;

  if (poolDescriptor.type == UR_USM_TYPE_DEVICE) {
    assert(level_zero_device_handle);
    auto residentHandles =
        poolDescriptor.hContext->getP2PDevices(poolDescriptor.hDevice);
    residentZeHandles.push_back(level_zero_device_handle);
    for (auto &device : residentHandles) {
      residentZeHandles.push_back(device->ZeDevice);
    }

    UMF_CALL_THROWS(umfLevelZeroMemoryProviderParamsSetResidentDevices(
        hParams, residentZeHandles.data(), residentZeHandles.size()));
  }

  UMF_CALL_THROWS(umfLevelZeroMemoryProviderParamsSetFreePolicy(
      hParams, UMF_LEVEL_ZERO_MEMORY_PROVIDER_FREE_POLICY_BLOCKING_FREE));

  auto [ret, provider] =
      umf::providerMakeUniqueFromOps(umfLevelZeroMemoryProviderOps(), hParams);
  if (ret != UMF_RESULT_SUCCESS) {
    throw umf::umf2urResult(ret);
  }

  return std::move(provider);
}

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                                             ur_usm_pool_desc_t *pPoolDesc)
    : hContext(hContext) {
  // TODO: handle UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK from pPoolDesc
  auto disjointPoolConfigs = initializeDisjointPoolConfig();

  if (disjointPoolConfigs.has_value()) {
    if (auto limits = find_stype_node<ur_usm_pool_limits_desc_t>(pPoolDesc)) {
      for (auto &config : disjointPoolConfigs.value().Configs) {
        config.MaxPoolableSize = limits->maxPoolableSize;
        config.SlabMinSize = limits->minDriverAllocSize;
      }
    }
  } else {
    // If pooling is disabled, do nothing.
    UR_LOG(INFO, "USM pooling is disabled. Skiping pool limits adjustment.");
  }

  auto devicesAndSubDevices =
      CollectDevicesAndSubDevices(hContext->getDevices());
  auto descriptors = usm::pool_descriptor::createFromDevices(
      this, hContext, devicesAndSubDevices);
  for (auto &desc : descriptors) {
    if (disjointPoolConfigs.has_value()) {
      auto &poolConfig =
          disjointPoolConfigs.value().Configs[descToDisjoinPoolMemType(desc)];
      poolManager.addPool(
          desc, usm::makeDisjointPool(makeProvider(desc), poolConfig));
    } else {
      poolManager.addPool(desc, usm::makeProxyPool(makeProvider(desc)));
    }
  }
}

ur_context_handle_t ur_usm_pool_handle_t_::getContextHandle() const {
  return hContext;
}

umf_memory_pool_handle_t
ur_usm_pool_handle_t_::getPool(const usm::pool_descriptor &desc) {
  auto pool = poolManager.getPool(desc).value();
  assert(pool);
  return pool;
}

static ur_usm_device_mem_flags_t getDeviceFlags(const ur_usm_desc_t *pUSMDesc) {
  if (auto devDesc = find_stype_node<ur_usm_device_desc_t>(pUSMDesc)) {
    return devDesc->flags;
  }

  return 0;
}

ur_result_t ur_usm_pool_handle_t_::allocate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice, const ur_usm_desc_t *pUSMDesc,
    ur_usm_type_t type, size_t size, void **ppRetMem) {
  uint32_t alignment = pUSMDesc ? pUSMDesc->align : 0;

  if ((alignment & (alignment - 1)) != 0) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  auto deviceFlags = getDeviceFlags(pUSMDesc);

  auto umfPool = getPool(usm::pool_descriptor{
      this, hContext, hDevice, type,
      bool(deviceFlags & UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY)});
  if (!umfPool) {
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  *ppRetMem = umfPoolAlignedMalloc(umfPool, size, alignment);
  if (*ppRetMem == nullptr) {
    auto umfRet = umfPoolGetLastAllocationError(umfPool);
    return umf::umf2urResult(umfRet);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_usm_pool_handle_t_::free(void *ptr) {
  auto umfPool = umfPoolByPtr(ptr);
  if (umfPool) {
    return umf::umf2urResult(umfPoolFree(umfPool, ptr));
  } else {
    UR_LOG(ERR, "Failed to find pool for pointer: {}", ptr);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
}

namespace ur::level_zero {
ur_result_t urUSMPoolCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to USM pool descriptor. Can be chained with
    /// ::ur_usm_pool_limits_desc_t
    ur_usm_pool_desc_t *pPoolDesc,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *hPool) try {
  *hPool = new ur_usm_pool_handle_t_(hContext, pPoolDesc);
  return UR_RESULT_SUCCESS;
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
/// [in] pointer to USM memory pool
urUSMPoolRetain(ur_usm_pool_handle_t hPool) try {
  hPool->RefCount.increment();
  return UR_RESULT_SUCCESS;
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
/// [in] pointer to USM memory pool
urUSMPoolRelease(ur_usm_pool_handle_t hPool) try {
  if (hPool->RefCount.decrementAndTest()) {
    delete hPool;
  }
  return UR_RESULT_SUCCESS;
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urUSMPoolGetInfo(
    /// [in] handle of the USM memory pool
    ur_usm_pool_handle_t hPool,
    /// [in] name of the pool property to query
    ur_usm_pool_info_t propName,
    /// [in] size in bytes of the pool property value provided
    size_t propSize,
    /// [out][typename(propName, propSize)] value of the pool property
    void *pPropValue,
    /// [out] size in bytes returned in pool property value
    size_t *pPropSizeRet) try {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_USM_POOL_INFO_REFERENCE_COUNT: {
    return ReturnValue(hPool->RefCount.load());
  }
  case UR_USM_POOL_INFO_CONTEXT: {
    return ReturnValue(hPool->getContextHandle());
  }
  default: {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  }
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urUSMDeviceAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t hPool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM device memory object
    void **ppRetMem) try {
  if (!hPool) {
    hPool = hContext->getDefaultUSMPool();
  }

  return hPool->allocate(hContext, hDevice, pUSMDesc, UR_USM_TYPE_DEVICE, size,
                         ppRetMem);
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urUSMSharedAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t hPool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM shared memory object
    void **ppRetMem) try {
  if (!hPool) {
    hPool = hContext->getDefaultUSMPool();
  }

  return hPool->allocate(hContext, hDevice, pUSMDesc, UR_USM_TYPE_SHARED, size,
                         ppRetMem);
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urUSMHostAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t hPool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM host memory object
    void **ppRetMem) try {
  if (!hPool) {
    hPool = hContext->getDefaultUSMPool();
  }

  return hPool->allocate(hContext, nullptr, pUSMDesc, UR_USM_TYPE_HOST, size,
                         ppRetMem);
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urUSMFree(
    /// [in] handle of the context object
    ur_context_handle_t /*hContext*/,
    /// [in] pointer to USM memory object
    void *pMem) try {
  return umf::umf2urResult(umfFree(pMem));
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urUSMGetMemAllocInfo(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to USM memory object
    const void *ptr,
    /// [in] the name of the USM allocation property to query
    ur_usm_alloc_info_t propName,
    /// [in] size in bytes of the USM allocation property value
    size_t propValueSize,
    /// [out][optional] value of the USM allocation property
    void *pPropValue,
    /// [out][optional] bytes returned in USM allocation property
    size_t *pPropValueSizeRet) try {
  ze_device_handle_t zeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> zeMemoryAllocationProperties;

  // TODO: implement this using UMF once
  // https://github.com/oneapi-src/unified-memory-framework/issues/686
  // https://github.com/oneapi-src/unified-memory-framework/issues/687
  // are implemented
  ZE2UR_CALL(zeMemGetAllocProperties,
             (hContext->getZeHandle(), ptr, &zeMemoryAllocationProperties,
              &zeDeviceHandle));

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);
  switch (propName) {
  case UR_USM_ALLOC_INFO_TYPE: {
    ur_usm_type_t memAllocType;
    switch (zeMemoryAllocationProperties.type) {
    case ZE_MEMORY_TYPE_UNKNOWN:
      memAllocType = UR_USM_TYPE_UNKNOWN;
      break;
    case ZE_MEMORY_TYPE_HOST:
      memAllocType = UR_USM_TYPE_HOST;
      break;
    case ZE_MEMORY_TYPE_DEVICE:
      memAllocType = UR_USM_TYPE_DEVICE;
      break;
    case ZE_MEMORY_TYPE_SHARED:
      memAllocType = UR_USM_TYPE_SHARED;
      break;
    default:
      UR_LOG(ERR, "urUSMGetMemAllocInfo: unexpected usm memory type");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    return ReturnValue(memAllocType);
  }
  case UR_USM_ALLOC_INFO_DEVICE:
    if (zeDeviceHandle) {
      auto Platform = hContext->getPlatform();
      auto Device = Platform->getDeviceFromNativeHandle(zeDeviceHandle);
      return Device ? ReturnValue(Device) : UR_RESULT_ERROR_INVALID_VALUE;
    } else {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  case UR_USM_ALLOC_INFO_BASE_PTR: {
    void *base;
    ZE2UR_CALL(zeMemGetAddressRange,
               (hContext->getZeHandle(), ptr, &base, nullptr));
    return ReturnValue(base);
  }
  case UR_USM_ALLOC_INFO_SIZE: {
    size_t size;
    ZE2UR_CALL(zeMemGetAddressRange,
               (hContext->getZeHandle(), ptr, nullptr, &size));
    return ReturnValue(size);
  }
  case UR_USM_ALLOC_INFO_POOL: {
    // TODO
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  default:
    UR_LOG(ERR, "urUSMGetMemAllocInfo: unsupported ParamName");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  }
  return UR_RESULT_SUCCESS;
} catch (umf_result_t e) {
  return umf::umf2urResult(e);
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urUSMImportExp(ur_context_handle_t hContext, void *hostPtr,
                           size_t size) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_CONTEXT);

  // Promote the host ptr to USM host memory.
  if (ZeUSMImport.Supported && hostPtr != nullptr) {
    // Query memory type of the host pointer
    ze_device_handle_t hDevice;
    ZeStruct<ze_memory_allocation_properties_t> zeMemoryAllocationProperties;
    ZE2UR_CALL(zeMemGetAllocProperties,
               (hContext->getZeHandle(), hostPtr, &zeMemoryAllocationProperties,
                &hDevice));

    // If not shared of any type, we can import the ptr
    if (zeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN) {
      // Promote the host ptr to USM host memory
      ze_driver_handle_t driverHandle =
          hContext->getPlatform()->ZeDriverHandleExpTranslated;
      ZeUSMImport.doZeUSMImport(driverHandle, hostPtr, size);
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urUSMReleaseExp(ur_context_handle_t hContext, void *hostPtr) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_CONTEXT);

  // Release the imported memory.
  if (ZeUSMImport.Supported && hostPtr != nullptr)
    ZeUSMImport.doZeUSMRelease(
        hContext->getPlatform()->ZeDriverHandleExpTranslated, hostPtr);
  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
