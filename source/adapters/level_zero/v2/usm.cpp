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

#include <umf/pools/pool_disjoint.h>
#include <umf/pools/pool_proxy.h>
#include <umf/providers/provider_level_zero.h>

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

static usm::DisjointPoolAllConfigs initializeDisjointPoolConfig() {
  const char *PoolUrTraceVal = std::getenv("UR_L0_USM_ALLOCATOR_TRACE");

  int PoolTrace = 0;
  if (PoolUrTraceVal != nullptr) {
    PoolTrace = std::atoi(PoolUrTraceVal);
  }

  const char *PoolUrConfigVal = std::getenv("UR_L0_USM_ALLOCATOR");
  if (PoolUrConfigVal == nullptr) {
    return usm::DisjointPoolAllConfigs(PoolTrace);
  }

  return usm::parseDisjointPoolConfig(PoolUrConfigVal, PoolTrace);
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

static umf::pool_unique_handle_t
makePool(umf_disjoint_pool_params_t *poolParams,
         usm::pool_descriptor poolDescriptor) {
  level_zero_memory_provider_params_t params = {};
  params.level_zero_context_handle = poolDescriptor.hContext->getZeHandle();
  params.level_zero_device_handle =
      poolDescriptor.hDevice ? poolDescriptor.hDevice->ZeDevice : nullptr;
  params.memory_type = urToUmfMemoryType(poolDescriptor.type);

  std::vector<ze_device_handle_t> residentZeHandles;

  if (poolDescriptor.type == UR_USM_TYPE_DEVICE) {
    assert(params.level_zero_device_handle);
    auto residentHandles =
        poolDescriptor.hContext->getP2PDevices(poolDescriptor.hDevice);
    residentZeHandles.push_back(params.level_zero_device_handle);
    for (auto &device : residentHandles) {
      residentZeHandles.push_back(device->ZeDevice);
    }

    params.resident_device_handles = residentZeHandles.data();
    params.resident_device_count = residentZeHandles.size();
  }

  auto [ret, provider] =
      umf::providerMakeUniqueFromOps(umfLevelZeroMemoryProviderOps(), &params);
  if (ret != UMF_RESULT_SUCCESS) {
    throw umf::umf2urResult(ret);
  }

  if (!poolParams) {
    auto [ret, poolHandle] = umf::poolMakeUniqueFromOps(
        umfProxyPoolOps(), std::move(provider), nullptr);
    if (ret != UMF_RESULT_SUCCESS)
      throw umf::umf2urResult(ret);
    return std::move(poolHandle);
  } else {
    auto [ret, poolHandle] =
        umf::poolMakeUniqueFromOps(umfDisjointPoolOps(), std::move(provider),
                                   static_cast<void *>(poolParams));
    if (ret != UMF_RESULT_SUCCESS)
      throw umf::umf2urResult(ret);
    return std::move(poolHandle);
  }
}

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                                             ur_usm_pool_desc_t *pPoolDesc)
    : hContext(hContext) {
  // TODO: handle UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK from pPoolDesc
  auto disjointPoolConfigs = initializeDisjointPoolConfig();
  if (auto limits = find_stype_node<ur_usm_pool_limits_desc_t>(pPoolDesc)) {
    for (auto &config : disjointPoolConfigs.Configs) {
      config.MaxPoolableSize = limits->maxPoolableSize;
      config.SlabMinSize = limits->minDriverAllocSize;
    }
  }

  auto [result, descriptors] = usm::pool_descriptor::create(this, hContext);
  if (result != UR_RESULT_SUCCESS) {
    throw result;
  }

  for (auto &desc : descriptors) {
    if (disjointPoolConfigs.EnableBuffers) {
      auto &poolConfig =
          disjointPoolConfigs.Configs[descToDisjoinPoolMemType(desc)];
      poolManager.addPool(desc, makePool(&poolConfig, desc));
    } else {
      poolManager.addPool(desc, makePool(nullptr, desc));
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

ur_result_t ur_usm_pool_handle_t_::allocate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t *pUSMDesc, ur_usm_type_t type, size_t size,
    void **ppRetMem) {
  uint32_t alignment = pUSMDesc ? pUSMDesc->align : 0;

  auto umfPool =
      getPool(usm::pool_descriptor{this, hContext, hDevice, type, false});
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
  return umf::umf2urResult(umfFree(ptr));
}

namespace ur::level_zero {
ur_result_t urUSMPoolCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_pool_desc_t *
        pPoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *hPool ///< [out] pointer to USM memory pool
) {

  *hPool = new ur_usm_pool_handle_t_(hContext, pPoolDesc);
  return UR_RESULT_SUCCESS;
}

ur_result_t
urUSMPoolRetain(ur_usm_pool_handle_t hPool ///< [in] pointer to USM memory pool
) {
  hPool->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t
urUSMPoolRelease(ur_usm_pool_handle_t hPool ///< [in] pointer to USM memory pool
) {
  if (hPool->RefCount.decrementAndTest()) {
    delete hPool;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urUSMPoolGetInfo(
    ur_usm_pool_handle_t hPool,  ///< [in] handle of the USM memory pool
    ur_usm_pool_info_t propName, ///< [in] name of the pool property to query
    size_t propSize, ///< [in] size in bytes of the pool property value provided
    void *pPropValue, ///< [out][typename(propName, propSize)] value of the pool
                      ///< property
    size_t
        *pPropSizeRet ///< [out] size in bytes returned in pool property value
) {
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
}

ur_result_t urUSMDeviceAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t hPool, ///< [in][optional] Pointer to a pool created
                                ///< using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppRetMem ///< [out] pointer to USM device memory object
) {
  if (!hPool) {
    hPool = hContext->getDefaultUSMPool();
  }

  return hPool->allocate(hContext, hDevice, pUSMDesc, UR_USM_TYPE_DEVICE, size,
                         ppRetMem);
}

ur_result_t urUSMSharedAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t hPool, ///< [in][optional] Pointer to a pool created
                                ///< using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppRetMem ///< [out] pointer to USM shared memory object
) {
  if (!hPool) {
    hPool = hContext->getDefaultUSMPool();
  }

  return hPool->allocate(hContext, hDevice, pUSMDesc, UR_USM_TYPE_SHARED, size,
                         ppRetMem);
}

ur_result_t urUSMHostAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t hPool, ///< [in][optional] Pointer to a pool created
                                ///< using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppRetMem ///< [out] pointer to USM host memory object
) {
  if (!hPool) {
    hPool = hContext->getDefaultUSMPool();
  }

  return hPool->allocate(hContext, nullptr, pUSMDesc, UR_USM_TYPE_HOST, size,
                         ppRetMem);
}

ur_result_t
urUSMFree(ur_context_handle_t hContext, ///< [in] handle of the context object
          void *pMem                    ///< [in] pointer to USM memory object
) {
  std::ignore = hContext;
  return umf::umf2urResult(umfFree(pMem));
}

ur_result_t urUSMGetMemAllocInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const void *ptr,              ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t
        propName, ///< [in] the name of the USM allocation property to query
    size_t propValueSize, ///< [in] size in bytes of the USM allocation property
                          ///< value
    void *pPropValue, ///< [out][optional] value of the USM allocation property
    size_t *pPropValueSizeRet ///< [out][optional] bytes returned in USM
                              ///< allocation property
) {
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
      logger::error("urUSMGetMemAllocInfo: unexpected usm memory type");
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
    logger::error("urUSMGetMemAllocInfo: unsupported ParamName");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  }
  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
