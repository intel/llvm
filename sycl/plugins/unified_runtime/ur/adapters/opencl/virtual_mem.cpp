//===----------- virtual_mem.cpp - OpenCL Adapter -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//
#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] ur_virtual_mem_granularity_info_t propName,
    [[maybe_unused]] size_t propSize, [[maybe_unused]] void *pPropValue,
    [[maybe_unused]] size_t *pPropSizeRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemReserve(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const void *pStart, [[maybe_unused]] size_t size,
    [[maybe_unused]] void **ppStart) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemFree(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const void *pStart, [[maybe_unused]] size_t size) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemSetAccess(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const void *pStart, [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_virtual_mem_access_flags_t flags) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemMap(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const void *pStart, [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_physical_mem_handle_t hPhysicalMem,
    [[maybe_unused]] size_t offset,
    [[maybe_unused]] ur_virtual_mem_access_flags_t flags) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemUnmap(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const void *pStart, [[maybe_unused]] size_t size) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGetInfo(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const void *pStart, [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_virtual_mem_info_t propName,
    [[maybe_unused]] size_t propSize, [[maybe_unused]] void *pPropValue,
    [[maybe_unused]] size_t *pPropSizeRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
