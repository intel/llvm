//===--------- context.cpp - Level Zero Adapter --------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include "../device.hpp"

ur_context_handle_t_::ur_context_handle_t_(ze_context_handle_t hContext,
                                           uint32_t numDevices,
                                           const ur_device_handle_t *phDevices,
                                           bool)
    : hContext(hContext), hDevices(phDevices, phDevices + numDevices),
      commandListCache(hContext) {}

ur_context_handle_t_::~ur_context_handle_t_() noexcept(false) {
  // ur_context_handle_t_ is only created/destroyed through urContextCreate
  // and urContextRelease so it's safe to throw here
  ZE2UR_CALL_THROWS(zeContextDestroy, (hContext));
}

ur_result_t ur_context_handle_t_::retain() {
  RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_context_handle_t_::release() {
  if (!RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  delete this;
  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL
urContextCreate(uint32_t deviceCount, const ur_device_handle_t *phDevices,
                const ur_context_properties_t *pProperties,
                ur_context_handle_t *phContext) {
  std::ignore = pProperties;

  ur_platform_handle_t hPlatform = phDevices[0]->Platform;
  ZeStruct<ze_context_desc_t> contextDesc{};

  ze_context_handle_t zeContext{};
  ZE2UR_CALL(zeContextCreate, (hPlatform->ZeDriver, &contextDesc, &zeContext));

  *phContext =
      new ur_context_handle_t_(zeContext, deviceCount, phDevices, true);
  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urContextRetain(ur_context_handle_t hContext) {
  return hContext->retain();
}

ur_result_t UR_APICALL urContextRelease(ur_context_handle_t hContext) {
  return hContext->release();
}
