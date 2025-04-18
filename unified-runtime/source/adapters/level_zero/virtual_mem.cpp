//===---------------- virtual_mem.cpp - Level Zero Adapter ----------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "device.hpp"
#include "logger/ur_logger.hpp"
#include "physical_mem.hpp"

#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "v2/context.hpp"
#else
#include "context.hpp"
#endif

namespace ur::level_zero {

ur_result_t urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_virtual_mem_granularity_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM:
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED: {
    // For L0 the minimum and recommended granularity is the same. We use a
    // memory size of 1 byte to get the actual granularity instead of the
    // aligned size.
    size_t PageSize;
    ZE2UR_CALL(zeVirtualMemQueryPageSize,
               (hContext->getZeHandle(), hDevice->ZeDevice, 1, &PageSize));
    return ReturnValue(PageSize);
  }
  default:
    logger::error("Unsupported propName in urQueueGetInfo: propName={}({})",
                  propName, propName);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urVirtualMemReserve(ur_context_handle_t hContext,
                                const void *pStart, size_t size,
                                void **ppStart) {
  ZE2UR_CALL(zeVirtualMemReserve,
             (hContext->getZeHandle(), pStart, size, ppStart));

  return UR_RESULT_SUCCESS;
}

ur_result_t urVirtualMemFree(ur_context_handle_t hContext, const void *pStart,
                             size_t size) {
  ZE2UR_CALL(zeVirtualMemFree, (hContext->getZeHandle(), pStart, size));

  return UR_RESULT_SUCCESS;
}

ur_result_t urVirtualMemSetAccess(ur_context_handle_t hContext,
                                  const void *pStart, size_t size,
                                  ur_virtual_mem_access_flags_t flags) {
  ze_memory_access_attribute_t AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_NONE;
  if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
  else if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY;

  ZE2UR_CALL(zeVirtualMemSetAccessAttribute,
             (hContext->getZeHandle(), pStart, size, AccessAttr));

  return UR_RESULT_SUCCESS;
}

ur_result_t urVirtualMemMap(ur_context_handle_t hContext, const void *pStart,
                            size_t size, ur_physical_mem_handle_t hPhysicalMem,
                            size_t offset,
                            ur_virtual_mem_access_flags_t flags) {
  ze_memory_access_attribute_t AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_NONE;
  if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
  else if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY;

  ZE2UR_CALL(zeVirtualMemMap,
             (hContext->getZeHandle(), pStart, size,
              hPhysicalMem->ZePhysicalMem, offset, AccessAttr));

  return UR_RESULT_SUCCESS;
}

ur_result_t urVirtualMemUnmap(ur_context_handle_t hContext, const void *pStart,
                              size_t size) {
  ZE2UR_CALL(zeVirtualMemUnmap, (hContext->getZeHandle(), pStart, size));

  return UR_RESULT_SUCCESS;
}

ur_result_t urVirtualMemGetInfo(ur_context_handle_t hContext,
                                const void *pStart,
                                [[maybe_unused]] size_t size,
                                ur_virtual_mem_info_t propName, size_t propSize,
                                void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_VIRTUAL_MEM_INFO_ACCESS_MODE: {
    size_t QuerySize;
    ze_memory_access_attribute_t Access;
    ZE2UR_CALL(zeVirtualMemGetAccessAttribute,
               (hContext->getZeHandle(), pStart, size, &Access, &QuerySize));
    ur_virtual_mem_access_flags_t RetFlags = 0;
    if (Access & ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE)
      RetFlags |= UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE;
    if (Access & ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY)
      RetFlags |= UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY;
    return ReturnValue(RetFlags);
  }
  default:
    logger::error("Unsupported propName in urQueueGetInfo: propName={}({})",
                  propName, propName);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
