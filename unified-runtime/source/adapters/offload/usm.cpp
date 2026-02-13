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

  hContext->AllocTypeMap.insert_or_assign(
      *ppMem, alloc_info_t{OL_ALLOC_TYPE_HOST, size});
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t size, void **ppMem) {
  OL_RETURN_ON_ERR(olMemAlloc(hContext->Device->OffloadDevice,
                              OL_ALLOC_TYPE_DEVICE, size, ppMem));

  hContext->AllocTypeMap.insert_or_assign(
      *ppMem, alloc_info_t{OL_ALLOC_TYPE_DEVICE, size});
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t size, void **ppMem) {
  OL_RETURN_ON_ERR(olMemAlloc(hContext->Device->OffloadDevice,
                              OL_ALLOC_TYPE_MANAGED, size, ppMem));

  hContext->AllocTypeMap.insert_or_assign(
      *ppMem, alloc_info_t{OL_ALLOC_TYPE_MANAGED, size});
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  hContext->AllocTypeMap.erase(pMem);
  return offloadResultToUR(olMemFree(pMem));
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMGetMemAllocInfo(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const void *pMem,
    [[maybe_unused]] ur_usm_alloc_info_t propName,
    [[maybe_unused]] size_t propSize, [[maybe_unused]] void *pPropValue,
    [[maybe_unused]] size_t *pPropSizeRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
