//===----------- context.cpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include <ur_api.h>

UR_APIEXPORT ur_result_t UR_APICALL urContextCreate(
    uint32_t DeviceCount, const ur_device_handle_t *phDevices,
    const ur_context_properties_t *, ur_context_handle_t *phContext) {
  if (DeviceCount > 1) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  auto Ctx = new ur_context_handle_t_(*phDevices);
  *phContext = Ctx;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(ur_context_handle_t hContext) {
  hContext->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urContextRelease(ur_context_handle_t hContext) {
  if (--hContext->RefCount == 0) {
    delete hContext;
  }
  return UR_RESULT_SUCCESS;
}
