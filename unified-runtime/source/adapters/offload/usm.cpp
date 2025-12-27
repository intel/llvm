//===----------- usm.cpp - LLVM Offload Adapter  --------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "context.hpp"
#include "device.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urUSMHostAlloc(ur_context_handle_t hContext,
                                                   const ur_usm_desc_t *,
                                                   ur_usm_pool_handle_t,
                                                   size_t size, void **ppMem) {
  OL_RETURN_ON_ERR(olMemAlloc(hContext->Device->OffloadDevice,
                              OL_ALLOC_TYPE_HOST, size, ppMem));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t size, void **ppMem) {
  OL_RETURN_ON_ERR(olMemAlloc(hContext->Device->OffloadDevice,
                              OL_ALLOC_TYPE_DEVICE, size, ppMem));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t size, void **ppMem) {
  OL_RETURN_ON_ERR(olMemAlloc(hContext->Device->OffloadDevice,
                              OL_ALLOC_TYPE_MANAGED, size, ppMem));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t, void *pMem) {
  return offloadResultToUR(olMemFree(pMem));
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ol_mem_info_t olInfo;

  switch (propName) {
  case UR_USM_ALLOC_INFO_TYPE:
    olInfo = OL_MEM_INFO_TYPE;
    break;
  case UR_USM_ALLOC_INFO_BASE_PTR:
    olInfo = OL_MEM_INFO_BASE;
    break;
  case UR_USM_ALLOC_INFO_SIZE:
    olInfo = OL_MEM_INFO_SIZE;
    break;
  case UR_USM_ALLOC_INFO_DEVICE:
    // Contexts can only contain one device
    return ReturnValue(hContext->Device);
  case UR_USM_ALLOC_INFO_POOL:
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    break;
  }

  if (pPropSizeRet) {
    OL_RETURN_ON_ERR(olGetMemInfoSize(pMem, olInfo, pPropSizeRet));
  }

  if (pPropValue) {
    auto Err = olGetMemInfo(pMem, olInfo, propSize, pPropValue);
    if (Err && Err->Code == OL_ERRC_NOT_FOUND) {
      // If the device didn't allocate this object, return default values
      switch (propName) {
      case UR_USM_ALLOC_INFO_TYPE:
        return ReturnValue(UR_USM_TYPE_UNKNOWN);
      case UR_USM_ALLOC_INFO_BASE_PTR:
        return ReturnValue(nullptr);
      case UR_USM_ALLOC_INFO_SIZE:
        return ReturnValue(0);
      default:
        return UR_RESULT_ERROR_UNKNOWN;
      }
    }
    OL_RETURN_ON_ERR(Err);

    if (propName == UR_USM_ALLOC_INFO_TYPE) {
      auto *OlType = reinterpret_cast<ol_alloc_type_t *>(pPropValue);
      auto *UrType = reinterpret_cast<ur_usm_type_t *>(pPropValue);
      switch (*OlType) {
      case OL_ALLOC_TYPE_HOST:
        *UrType = UR_USM_TYPE_HOST;
        break;
      case OL_ALLOC_TYPE_DEVICE:
        *UrType = UR_USM_TYPE_DEVICE;
        break;
      case OL_ALLOC_TYPE_MANAGED:
        *UrType = UR_USM_TYPE_SHARED;
        break;
      default:
        *UrType = UR_USM_TYPE_UNKNOWN;
        break;
      }
    }
  }

  return UR_RESULT_SUCCESS;
}
