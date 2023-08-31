//===--------- virtual_memory.cpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_virtual_mem_granularity_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemReserve(ur_context_handle_t hContext, const void *pStart,
                    size_t size, void **ppStart) {
  std::ignore = hContext;
  std::ignore = pStart;
  std::ignore = size;
  std::ignore = ppStart;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemFree(
    ur_context_handle_t hContext, const void *pStart, size_t size) {
  std::ignore = hContext;
  std::ignore = pStart;
  std::ignore = size;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemMap(ur_context_handle_t hContext, const void *pStart, size_t size,
                ur_physical_mem_handle_t hPhysicalMem, size_t offset,
                ur_virtual_mem_access_flags_t flags) {
  std::ignore = hContext;
  std::ignore = pStart;
  std::ignore = size;
  std::ignore = hPhysicalMem;
  std::ignore = offset;
  std::ignore = flags;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemUnmap(
    ur_context_handle_t hContext, const void *pStart, size_t size) {
  std::ignore = hContext;
  std::ignore = pStart;
  std::ignore = size;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemSetAccess(ur_context_handle_t hContext, const void *pStart,
                      size_t size, ur_virtual_mem_access_flags_t flags) {
  std::ignore = hContext;
  std::ignore = pStart;
  std::ignore = size;
  std::ignore = flags;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemGetInfo(ur_context_handle_t hContext, const void *pStart,
                    size_t size, ur_virtual_mem_info_t propName,
                    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hContext;
  std::ignore = pStart;
  std::ignore = size;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urPhysicalMemCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    const ur_physical_mem_properties_t *pProperties,
    ur_physical_mem_handle_t *phPhysicalMem) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = size;
  std::ignore = pProperties;
  std::ignore = phPhysicalMem;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem) {
  std::ignore = hPhysicalMem;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem) {
  std::ignore = hPhysicalMem;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
