//===--------- ur_level_zero_virtual_mem.cpp - Level Zero Adapter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_level_zero.hpp"
#include "ur_level_zero_common.hpp"
#include "ur_level_zero_context.hpp"
#include "ur_level_zero_device.hpp"
#include "ur_level_zero_physical_mem.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_virtual_mem_granularity_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pPropValue || pPropSizeRet, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM:
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED: {
    // For L0 the minimum and recommended granularity is the same. We use an
    // memory size of 1 byte to get the actual granularity instead of the
    // aligned size.
    size_t PageSize;
    ZE2UR_CALL(zeVirtualMemQueryPageSize,
               (hContext->ZeContext, hDevice->ZeDevice, 1, &PageSize));
    return ReturnValue(PageSize);
  }
  default:
    urPrint("Unsupported propName in urQueueGetInfo: propName=%d(0x%x)\n",
            propName, propName);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemReserve(ur_context_handle_t hContext, const void *pStart,
                    size_t size, void **ppStart) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(ppStart, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ZE2UR_CALL(zeVirtualMemReserve, (hContext->ZeContext, pStart, size, ppStart));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemFree(
    ur_context_handle_t hContext, const void *pStart, size_t size) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pStart, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ZE2UR_CALL(zeVirtualMemFree, (hContext->ZeContext, pStart, size));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemSetAccess(ur_context_handle_t hContext, const void *pStart,
                      size_t size, ur_virtual_mem_access_flags_t flags) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pStart, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ze_memory_access_attribute_t AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_NONE;
  if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
  else if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY;

  ZE2UR_CALL(zeVirtualMemSetAccessAttribute,
             (hContext->ZeContext, pStart, size, AccessAttr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemMap(ur_context_handle_t hContext, const void *pStart, size_t size,
                ur_physical_mem_handle_t hPhysicalMem, size_t offset,
                ur_virtual_mem_access_flags_t flags) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hPhysicalMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pStart, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ze_memory_access_attribute_t AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_NONE;
  if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
  else if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY)
    AccessAttr = ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY;

  ZE2UR_CALL(zeVirtualMemMap,
             (hContext->ZeContext, pStart, size, hPhysicalMem->ZePhysicalMem,
              offset, AccessAttr));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemUnmap(
    ur_context_handle_t hContext, const void *pStart, size_t size) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pStart, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ZE2UR_CALL(zeVirtualMemUnmap, (hContext->ZeContext, pStart, size));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGetInfo(
    ur_context_handle_t hContext, const void *pStart,
    [[maybe_unused]] size_t size, ur_virtual_mem_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pPropValue || pPropSizeRet, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_VIRTUAL_MEM_INFO_ACCESS_MODE: {
    size_t QuerySize;
    ze_memory_access_attribute_t Access;
    ZE2UR_CALL(zeVirtualMemGetAccessAttribute,
               (hContext->ZeContext, pStart, size, &Access, &QuerySize));
    ur_virtual_mem_access_flags_t RetFlags = 0;
    if (Access & ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE)
      RetFlags |= UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE;
    if (Access & ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY)
      RetFlags |= UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY;
    return ReturnValue(Access);
  }
  default:
    urPrint("Unsupported propName in urQueueGetInfo: propName=%d(0x%x)\n",
            propName, propName);
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}
