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
#include "usm.hpp"

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t hContext,
                                             ur_usm_pool_desc_t *)
    : hContext(hContext) {}

ur_context_handle_t ur_usm_pool_handle_t_::getContextHandle() const {
  return hContext;
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
  std::ignore = pUSMDesc;
  std::ignore = hPool;

  ZeStruct<ze_device_mem_alloc_desc_t> devDesc;
  devDesc.ordinal = 0;
  ZE2UR_CALL(zeMemAllocDevice, (hContext->getZeHandle(), &devDesc, size, 0,
                                hDevice->ZeDevice, ppRetMem));

  return UR_RESULT_SUCCESS;
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
  std::ignore = pUSMDesc;
  std::ignore = hPool;

  ZeStruct<ze_host_mem_alloc_desc_t> hostDesc;

  ZeStruct<ze_device_mem_alloc_desc_t> devDesc;
  devDesc.ordinal = 0;

  ZE2UR_CALL(zeMemAllocShared, (hContext->getZeHandle(), &devDesc, &hostDesc,
                                size, 0, hDevice->ZeDevice, ppRetMem));

  return UR_RESULT_SUCCESS;
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
  std::ignore = pUSMDesc;
  std::ignore = hPool;

  ZeStruct<ze_host_mem_alloc_desc_t> hostDesc;

  ZE2UR_CALL(zeMemAllocHost,
             (hContext->getZeHandle(), &hostDesc, size, 0, ppRetMem));

  return UR_RESULT_SUCCESS;
}

ur_result_t
urUSMFree(ur_context_handle_t hContext, ///< [in] handle of the context object
          void *pMem                    ///< [in] pointer to USM memory object
) {
  std::ignore = hContext;

  ZE2UR_CALL(zeMemFree, (hContext->getZeHandle(), pMem));
  return UR_RESULT_SUCCESS;
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
