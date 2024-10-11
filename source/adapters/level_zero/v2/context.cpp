//===--------- context.cpp - Level Zero Adapter --------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device.hpp"

#include "context.hpp"
#include "event_provider_normal.hpp"

ur_context_handle_t_::ur_context_handle_t_(ze_context_handle_t hContext,
                                           uint32_t numDevices,
                                           const ur_device_handle_t *phDevices,
                                           bool ownZeContext)
    : hContext(hContext, ownZeContext),
      hDevices(phDevices, phDevices + numDevices), commandListCache(hContext),
      eventPoolCache(phDevices[0]->Platform->getNumDevices(),
                     [context = this,
                      platform = phDevices[0]->Platform](DeviceId deviceId) {
                       auto device = platform->getDeviceById(deviceId);
                       // TODO: just use per-context id?
                       return std::make_unique<v2::provider_normal>(
                           context, device, v2::EVENT_COUNTER,
                           v2::QUEUE_IMMEDIATE);
                     }),
      defaultUSMPool(this, nullptr) {}

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

ur_platform_handle_t ur_context_handle_t_::getPlatform() const {
  return hDevices[0]->Platform;
}

const std::vector<ur_device_handle_t> &
ur_context_handle_t_::getDevices() const {
  return hDevices;
}

bool ur_context_handle_t_::isValidDevice(ur_device_handle_t hDevice) const {
  while (hDevice) {
    if (std::find(hDevices.begin(), hDevices.end(), hDevice) != hDevices.end())
      return true;
    hDevice = hDevice->RootDevice;
  }
  return false;
}

ur_usm_pool_handle_t ur_context_handle_t_::getDefaultUSMPool() {
  return &defaultUSMPool;
}

namespace ur::level_zero {
ur_result_t urContextCreate(uint32_t deviceCount,
                            const ur_device_handle_t *phDevices,
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

ur_result_t urContextRetain(ur_context_handle_t hContext) {
  return hContext->retain();
}

ur_result_t urContextRelease(ur_context_handle_t hContext) {
  return hContext->release();
}

ur_result_t urContextGetInfo(ur_context_handle_t hContext,
                             ur_context_info_t contextInfoType, size_t propSize,

                             void *pContextInfo,

                             size_t *pPropSizeRet) {
  std::shared_lock<ur_shared_mutex> Lock(hContext->Mutex);
  UrReturnHelper ReturnValue(propSize, pContextInfo, pPropSizeRet);
  switch (
      (uint32_t)contextInfoType) { // cast to avoid warnings on EXT enum values
  case UR_CONTEXT_INFO_DEVICES:
    return ReturnValue(hContext->getDevices().data(),
                       hContext->getDevices().size());
  case UR_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(uint32_t(hContext->getDevices().size()));
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{hContext->RefCount.load()});
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    // TODO: this is currently not implemented
    return ReturnValue(uint8_t{false});
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    // 2D USM fill is not supported.
    return ReturnValue(uint8_t{false});
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }
}
} // namespace ur::level_zero
