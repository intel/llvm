//===------------- usm.cpp - NATIVE CPU Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t pool, size_t size, void **ppMem) {
  std::ignore = hContext;
  std::ignore = pUSMDesc;
  std::ignore = pool;

  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // TODO: Check Max size when UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE is implemented
  UR_ASSERT(size > 0, UR_RESULT_ERROR_INVALID_USM_SIZE);

  *ppMem = malloc(size);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
                 size_t size, void **ppMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pUSMDesc;
  std::ignore = pool;

  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // TODO: Check Max size when UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE is implemented
  UR_ASSERT(size > 0, UR_RESULT_ERROR_INVALID_USM_SIZE);

  *ppMem = malloc(size);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
                 size_t size, void **ppMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pUSMDesc;
  std::ignore = pool;

  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // TODO: Check Max size when UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE is implemented
  UR_ASSERT(size > 0, UR_RESULT_ERROR_INVALID_USM_SIZE);

  *ppMem = malloc(size);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  std::ignore = hContext;

  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  free(pMem);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hContext;
  std::ignore = pMem;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolCreate(ur_context_handle_t hContext, ur_usm_pool_desc_t *pPoolDesc,
                ur_usm_pool_handle_t *ppPool) {
  std::ignore = hContext;
  std::ignore = pPoolDesc;
  std::ignore = ppPool;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRetain(ur_usm_pool_handle_t pPool) {
  std::ignore = pPool;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRelease(ur_usm_pool_handle_t pPool) {
  std::ignore = pPool;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolGetInfo(ur_usm_pool_handle_t hPool, ur_usm_pool_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hPool;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(ur_context_handle_t Context,
                                                   void *HostPtr, size_t Size) {
  std::ignore = Context;
  std::ignore = HostPtr;
  std::ignore = Size;
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMReleaseExp(ur_context_handle_t Context,
                                                    void *HostPtr) {
  std::ignore = Context;
  std::ignore = HostPtr;
  DIE_NO_IMPLEMENTATION;
}