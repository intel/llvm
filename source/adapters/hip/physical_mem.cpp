//===--------- physical_mem.cpp - HIP Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "physical_mem.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urPhysicalMemCreate(
    ur_context_handle_t, ur_device_handle_t, size_t,
    const ur_physical_mem_properties_t *, ur_physical_mem_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRetain(ur_physical_mem_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRelease(ur_physical_mem_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemGetInfo(ur_physical_mem_handle_t, ur_physical_mem_info_t, size_t,
                     void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
