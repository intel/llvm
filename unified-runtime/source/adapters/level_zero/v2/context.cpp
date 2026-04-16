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
                      (device->ZeDevice, hSourceDevice->ZeDevice, &p2p));

    if (p2p) {
      p2pDevices.push_back(device);
    }
  }
  return p2pDevices;
}

static std::vector<std::vector<ur_device_handle_t>>
populateP2PDevices(const std::vector<ur_device_handle_t> &devices) {
  std::vector<ur_device_handle_t> allDevices;
  std::function<void(ur_device_handle_t)> collectDeviceAndSubdevices =
      [&allDevices, &collectDeviceAndSubdevices](ur_device_handle_t device) {
        allDevices.push_back(device);
        for (auto &subDevice : device->SubDevices) {
          collectDeviceAndSubdevices(subDevice);
        }
      };

  for (auto &device : devices) {
    collectDeviceAndSubdevices(device);
  }

  uint64_t maxDeviceId = 0;
  for (auto &device : allDevices) {
    maxDeviceId = std::max(maxDeviceId, device->Id.value());
  }

  std::vector<std::vector<ur_device_handle_t>> p2pDevices(maxDeviceId + 1);
  for (auto &device : allDevices) {
    p2pDevices[device->Id.value()] = filterP2PDevices(device, allDevices);
  }
  return p2pDevices;
}

static std::vector<ur_device_handle_t>
uniqueDevices(uint32_t numDevices, const ur_device_handle_t *phDevices) {
  std::vector<ur_device_handle_t> devices(phDevices, phDevices + numDevices);
  std::sort(devices.begin(), devices.end());
  devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
  return devices;
}

static bool isDriverRootDevice(ze_device_handle_t zeDevice) {
  ze_device_handle_t zeRootDevice = nullptr;
  ze_result_t zeResult =
      ZE_CALL_NOCHECK(zeDeviceGetRootDevice, (zeDevice, &zeRootDevice));

  if (zeResult == ZE_RESULT_SUCCESS) {
    return zeRootDevice == nullptr || zeRootDevice == zeDevice;
  }

  // If unsupported, keep behavior compatible with older loaders/drivers.
  if (zeResult == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return true;
  }

  return false;
}

static bool isFullPlatformRootDeviceList(uint32_t deviceCount,
                                         const ur_device_handle_t *phDevices) {
  if (deviceCount == 0 || !phDevices) {
    return false;
  }

  ur_platform_handle_t hPlatform = phDevices[0]->Platform;

  std::vector<ze_device_handle_t> requestedDevices;
  requestedDevices.reserve(deviceCount);
  for (uint32_t i = 0; i < deviceCount; ++i) {
    if (!phDevices[i] || phDevices[i]->Platform != hPlatform ||
        phDevices[i]->RootDevice) {
      return false;
    }

    if (!isDriverRootDevice(phDevices[i]->ZeDevice)) {
      return false;
    }

    requestedDevices.push_back(phDevices[i]->ZeDevice);
  }

  std::sort(requestedDevices.begin(), requestedDevices.end());
  requestedDevices.erase(
      std::unique(requestedDevices.begin(), requestedDevices.end()),
      requestedDevices.end());

  uint32_t zeDeviceCount = 0;
  ze_result_t zeResult =
      ZE_CALL_NOCHECK(zeDeviceGet, (hPlatform->ZeDriver, &zeDeviceCount, nullptr));
  if (zeResult != ZE_RESULT_SUCCESS || zeDeviceCount == 0) {
    return false;
  }

  std::vector<ze_device_handle_t> platformDevices(zeDeviceCount);
  zeResult = ZE_CALL_NOCHECK(
      zeDeviceGet, (hPlatform->ZeDriver, &zeDeviceCount, platformDevices.data()));
  if (zeResult != ZE_RESULT_SUCCESS) {
    return false;
  }

  platformDevices.resize(zeDeviceCount);
  platformDevices.erase(
      std::remove_if(platformDevices.begin(), platformDevices.end(),
                     [](ze_device_handle_t zeDevice) {
                       return !isDriverRootDevice(zeDevice);
                     }),
      platformDevices.end());

  if (platformDevices.empty()) {
    return false;
  }

  std::sort(platformDevices.begin(), platformDevices.end());
  platformDevices.erase(
      std::unique(platformDevices.begin(), platformDevices.end()),
      platformDevices.end());

  return requestedDevices == platformDevices;
}

ur_context_handle_t_::ur_context_handle_t_(ze_context_handle_t hContext,
                                           uint32_t numDevices,
                                           const ur_device_handle_t *phDevices,
                                           bool ownZeContext)
    : hContext(hContext, ownZeContext),
      hDevices(uniqueDevices(numDevices, phDevices)),
      commandListCache(
          hContext, {phDevices[0]->Platform->ZeCopyOffloadExtensionSupported,
                     phDevices[0]->Platform->ZeMutableCmdListExt.Supported,
                     phDevices[0]->Platform->ZeCopyOffloadQueueFlagSupported,
                     phDevices[0]->Platform->ZeCopyOffloadListFlagSupported}),
      eventPoolCacheImmediate(
          this, phDevices[0]->Platform->getNumDevices(),
          [context = this, platform = phDevices[0]->Platform](
              DeviceId deviceId,
              v2::event_flags_t flags) -> std::unique_ptr<v2::event_provider> {
            auto device = platform->getDeviceById(deviceId);

            // TODO: just use per-context id?
            return v2::createProvider(platform, context, v2::QUEUE_IMMEDIATE,
                                      device, flags);
          }),
      eventPoolCacheRegular(
          this, phDevices[0]->Platform->getNumDevices(),
          [context = this, platform = phDevices[0]->Platform](
              DeviceId deviceId,
              v2::event_flags_t flags) -> std::unique_ptr<v2::event_provider> {
            auto device = platform->getDeviceById(deviceId);

            // TODO: just use per-context id?
            return v2::createProvider(platform, context, v2::QUEUE_REGULAR,
                                      device, flags);
          }),
      nativeEventsPool(this, std::make_unique<v2::provider_normal>(
                                 this, v2::QUEUE_IMMEDIATE,
                                 v2::EVENT_FLAGS_PROFILING_ENABLED)),
      p2pAccessDevices(populateP2PDevices(this->hDevices)),
      defaultUSMPool(this, nullptr), asyncPool(this, nullptr) {}

ur_result_t ur_context_handle_t_::retain() {
  RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_context_handle_t_::release() {
  if (!RefCount.release())
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

ur_usm_pool_handle_t ur_context_handle_t_::getAsyncPool() { return &asyncPool; }

void ur_context_handle_t_::addUsmPool(ur_usm_pool_handle_t hPool) {
  std::scoped_lock<ur_shared_mutex> lock(Mutex);
  usmPoolHandles.push_back(hPool);
}

void ur_context_handle_t_::removeUsmPool(ur_usm_pool_handle_t hPool) {
  std::scoped_lock<ur_shared_mutex> lock(Mutex);
  usmPoolHandles.remove(hPool);
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

  ur_platform_handle_t hPlatform = phDevices[0]->Platform;
  ZeStruct<ze_context_desc_t> contextDesc{};

  ze_context_handle_t zeContext{};
  bool ownZeContext = true;

  if (!pProperties && isFullPlatformRootDeviceList(deviceCount, phDevices)) {
    ze_context_handle_t zeDefaultContext =
        zeDriverGetDefaultContext(hPlatform->ZeDriver);
    if (zeDefaultContext) {
      zeContext = zeDefaultContext;
      ownZeContext = false;
    }
  }

  if (!zeContext) {
    ZE2UR_CALL(zeContextCreate, (hPlatform->ZeDriver, &contextDesc, &zeContext));
  }

  *phContext =
      new ur_context_handle_t_(zeContext, deviceCount, phDevices, ownZeContext);
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
    return ReturnValue(uint32_t{hContext->RefCount.getCount()});
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    // TODO: this is currently not implemented
    return ReturnValue(uint8_t{false});
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    // 2D USM fill is not supported.
    return ReturnValue(uint8_t{false});
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
} catch (...) {
  return exceptionToResult(std::current_exception());
}
} // namespace ur::level_zero
