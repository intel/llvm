//===----------- physical_mem.cpp - OpenCL Adapter ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//
#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urPhysicalMemCreate(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice, [[maybe_unused]] size_t size,
    [[maybe_unused]] const ur_physical_mem_properties_t *pProperties,
    [[maybe_unused]] ur_physical_mem_handle_t *phPhysicalMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRetain([[maybe_unused]] ur_physical_mem_handle_t hPhysicalMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRelease([[maybe_unused]] ur_physical_mem_handle_t hPhysicalMem) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
