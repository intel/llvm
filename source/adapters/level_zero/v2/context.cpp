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
#include "event_provider_counter.hpp"
#include "event_provider_normal.hpp"

static std::vector<ur_device_handle_t>
filterP2PDevices(ur_device_handle_t hSourceDevice,
                 const std::vector<ur_device_handle_t> &devices) {
  std::vector<ur_device_handle_t> p2pDevices;
  for (auto &device : devices) {
    if (device == hSourceDevice) {
      continue;
    }

    ze_bool_t p2p;
    ZE2UR_CALL_THROWS(zeDeviceCanAccessPeer,
                      (hSourceDevice->ZeDevice, device->ZeDevice, &p2p));

    if (p2p) {
      p2pDevices.push_back(device);
    }
  }
  return p2pDevices;
}

static std::vector<std::vector<ur_device_handle_t>>
populateP2PDevices(size_t maxDevices,
                   const std::vector<ur_device_handle_t> &devices) {
  std::vector<std::vector<ur_device_handle_t>> p2pDevices(maxDevices);
  for (auto &device : devices) {
    p2pDevices[device->Id.value()] = filterP2PDevices(device, devices);
  }
  return p2pDevices;
}

ur_context_handle_t_::ur_context_handle_t_(ze_context_handle_t hContext,
                                           uint32_t numDevices,
                                           const ur_device_handle_t *phDevices,
                                           bool ownZeContext)
    : commandListCache(hContext),
      eventPoolCache(this, phDevices[0]->Platform->getNumDevices(),
                     [context = this, platform = phDevices[0]->Platform](
                         DeviceId deviceId, v2::event_flags_t flags)
                         -> std::unique_ptr<v2::event_provider> {
                       assert((flags & v2::EVENT_FLAGS_COUNTER) != 0);

                       std::ignore = deviceId;
                       std::ignore = platform;

                       // TODO: just use per-context id?
                       return std::make_unique<v2::provider_normal>(
                           context, v2::QUEUE_IMMEDIATE, flags);
                     }),
      nativeEventsPool(this, std::make_unique<v2::provider_normal>(
                                 this, v2::QUEUE_IMMEDIATE,
                                 v2::EVENT_FLAGS_PROFILING_ENABLED)),
      hContext(hContext, ownZeContext),
      hDevices(phDevices, phDevices + numDevices),
      p2pAccessDevices(populateP2PDevices(
          phDevices[0]->Platform->getNumDevices(), this->hDevices)),
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

const std::vector<ur_device_handle_t> &
ur_context_handle_t_::getP2PDevices(ur_device_handle_t hDevice) const {
  return p2pAccessDevices[hDevice->Id.value()];
}

namespace ur::level_zero {
ur_result_t urContextCreate(uint32_t deviceCount,
                            const ur_device_handle_t *phDevices,
                            const ur_context_properties_t *pProperties,
                            ur_context_handle_t *phContext) try {
  std::ignore = pProperties;

  ur_platform_handle_t hPlatform = phDevices[0]->Platform;
  ZeStruct<ze_context_desc_t> contextDesc{};

  ze_context_handle_t zeContext{};
  ZE2UR_CALL(zeContextCreate, (hPlatform->ZeDriver, &contextDesc, &zeContext));

  *phContext =
      new ur_context_handle_t_(zeContext, deviceCount, phDevices, true);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urContextGetNativeHandle(ur_context_handle_t hContext,
                                     ur_native_handle_t *phNativeContext) try {
  *phNativeContext =
      reinterpret_cast<ur_native_handle_t>(hContext->getZeHandle());
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext, ur_adapter_handle_t, uint32_t numDevices,
    const ur_device_handle_t *phDevices,
    const ur_context_native_properties_t *pProperties,
    ur_context_handle_t *phContext) try {
  auto zeContext = reinterpret_cast<ze_context_handle_t>(hNativeContext);

  auto ownZeHandle = pProperties ? pProperties->isNativeHandleOwned : false;

  *phContext =
      new ur_context_handle_t_(zeContext, numDevices, phDevices, ownZeHandle);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urContextRetain(ur_context_handle_t hContext) try {
  return hContext->retain();
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urContextRelease(ur_context_handle_t hContext) try {
  return hContext->release();
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urContextGetInfo(ur_context_handle_t hContext,
                             ur_context_info_t contextInfoType, size_t propSize,
                             void *pContextInfo, size_t *pPropSizeRet) try {
  // No locking needed here, we only read const members

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
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
} catch (...) {
  return exceptionToResult(std::current_exception());
}
} // namespace ur::level_zero
