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

  hContext->AllocTypeMap.insert_or_assign(*ppMem, OL_ALLOC_TYPE_HOST);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t size, void **ppMem) {
  OL_RETURN_ON_ERR(olMemAlloc(hContext->Device->OffloadDevice,
                              OL_ALLOC_TYPE_DEVICE, size, ppMem));

  hContext->AllocTypeMap.insert_or_assign(*ppMem, OL_ALLOC_TYPE_DEVICE);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t size, void **ppMem) {
  OL_RETURN_ON_ERR(olMemAlloc(hContext->Device->OffloadDevice,
                              OL_ALLOC_TYPE_MANAGED, size, ppMem));

  hContext->AllocTypeMap.insert_or_assign(*ppMem, OL_ALLOC_TYPE_MANAGED);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t, void *pMem) {
  return offloadResultToUR(olMemFree(pMem));
}
