//===------------- usm.cpp - NATIVE CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur/ur.hpp"
#include "ur_api.h"

#include "common.hpp"
#include "context.hpp"
#include <cstdlib>

namespace umf {
ur_result_t getProviderNativeError(const char *, int32_t) {
  return UR_RESULT_ERROR_UNKNOWN;
}
} // namespace umf

static ur_result_t alloc_helper(ur_context_handle_t hContext,
                                const ur_usm_desc_t *pUSMDesc, size_t size,
                                void **ppMem, ur_usm_type_t type) {
  auto alignment = (pUSMDesc && pUSMDesc->align) ? pUSMDesc->align : 1u;
  UR_ASSERT(isPowerOf2(alignment), UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // TODO: Check Max size when UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE is implemented
  UR_ASSERT(size > 0, UR_RESULT_ERROR_INVALID_USM_SIZE);

  auto *ptr = hContext->add_alloc(alignment, type, size, nullptr);
  UR_ASSERT(ptr != nullptr, UR_RESULT_ERROR_OUT_OF_RESOURCES);
  *ppMem = ptr;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t /*pool*/, size_t size, void **ppMem) {

  return alloc_helper(hContext, pUSMDesc, size, ppMem, UR_USM_TYPE_HOST);
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t /*hDevice*/,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t /*pool*/,
                 size_t size, void **ppMem) {

  return alloc_helper(hContext, pUSMDesc, size, ppMem, UR_USM_TYPE_DEVICE);
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t /*hDevice*/,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t /*pool*/,
                 size_t size, void **ppMem) {

  return alloc_helper(hContext, pUSMDesc, size, ppMem, UR_USM_TYPE_SHARED);
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {

  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto res = hContext->remove_alloc(pMem);

  return res;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {

  UR_ASSERT(pMem != nullptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  if (propName == UR_USM_ALLOC_INFO_BASE_PTR) {
    // TODO: logic to compute base ptr given ptr
    DIE_NO_IMPLEMENTATION;
  }

  const native_cpu::usm_alloc_info &alloc_info =
      hContext->get_alloc_info_entry(pMem);
  switch (propName) {
  case UR_USM_ALLOC_INFO_TYPE:
    return ReturnValue(alloc_info.type);
  case UR_USM_ALLOC_INFO_SIZE:
    return ReturnValue(alloc_info.size);
  case UR_USM_ALLOC_INFO_DEVICE:
    return ReturnValue(alloc_info.device);
  case UR_USM_ALLOC_INFO_POOL:
    return ReturnValue(alloc_info.pool);
  default:
    DIE_NO_IMPLEMENTATION;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolCreate(
    ur_context_handle_t /*hContext*/, ur_usm_pool_desc_t * /*pPoolDesc*/,
    ur_usm_pool_handle_t * /*ppPool*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRetain(ur_usm_pool_handle_t /*pPool*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRelease(ur_usm_pool_handle_t /*pPool*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetInfo(
    ur_usm_pool_handle_t /*hPool*/, ur_usm_pool_info_t /*propName*/,
    size_t /*propSize*/, void * /*pPropValue*/, size_t * /*pPropSizeRet*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(
    ur_context_handle_t /*Context*/, void * /*HostPtr*/, size_t /*Size*/) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMReleaseExp(ur_context_handle_t /*Context*/, void * /*HostPtr*/) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolCreateExp(ur_context_handle_t,
                                                       ur_device_handle_t,
                                                       ur_usm_pool_desc_t *,
                                                       ur_usm_pool_handle_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolDestroyExp(ur_context_handle_t,
                                                        ur_device_handle_t,
                                                        ur_usm_pool_handle_t) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolSetInfoExp(ur_usm_pool_handle_t,
                                                        ur_usm_pool_info_t,
                                                        void *, size_t) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetInfoExp(ur_usm_pool_handle_t,
                                                        ur_usm_pool_info_t,
                                                        void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolTrimToExp(ur_context_handle_t,
                                                       ur_device_handle_t,
                                                       ur_usm_pool_handle_t,
                                                       size_t) {
  DIE_NO_IMPLEMENTATION;
}
