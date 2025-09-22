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
uniqueDevices(uint32_t numDevices, const ur_device_handle_t *phDevices) {
  std::vector<ur_device_handle_t> devices(phDevices, phDevices + numDevices);
  std::sort(devices.begin(), devices.end());
  devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
  return devices;
}

ur_context_handle_t_::ur_context_handle_t_(ze_context_handle_t hContext,
                                           uint32_t numDevices,
                                           const ur_device_handle_t *phDevices,
                                           bool ownZeContext)
    : hContext(hContext, ownZeContext),
      hDevices(uniqueDevices(numDevices, phDevices)),
      commandListCache(hContext,
                       {phDevices[0]->Platform->ZeCopyOffloadExtensionSupported,
                        phDevices[0]->Platform->ZeMutableCmdListExt.Supported}),
      eventPoolCacheImmediate(
          this, phDevices[0]->Platform->getNumDevices(),
          [context = this](DeviceId /* deviceId*/, v2::event_flags_t flags)
              -> std::unique_ptr<v2::event_provider> {
            // TODO: just use per-context id?
            return std::make_unique<v2::provider_normal>(
                context, v2::QUEUE_IMMEDIATE, flags);
          }),
      eventPoolCacheRegular(this, phDevices[0]->Platform->getNumDevices(),
                            [context = this, platform = phDevices[0]->Platform](
                                DeviceId deviceId, v2::event_flags_t flags)
                                -> std::unique_ptr<v2::event_provider> {
                              std::ignore = deviceId;
                              std::ignore = platform;

                              // TODO: just use per-context id?
                              return std::make_unique<v2::provider_normal>(
                                  context, v2::QUEUE_REGULAR, flags);
                            }),
      nativeEventsPool(this, std::make_unique<v2::provider_normal>(
                                 this, v2::QUEUE_IMMEDIATE,
                                 v2::EVENT_FLAGS_PROFILING_ENABLED)),
      defaultUSMPool(this, nullptr), asyncPool(this, nullptr) {
  UR_LOG(INFO, "UR context created with {} devices", numDevices);
}

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

const std::vector<ur_device_handle_t> &ur_context_handle_t_::getDevices() const {
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
  UR_LOG(INFO, "Adding USM pool {} to context:{}", hPool, this);
  std::scoped_lock<ur_shared_mutex> lock(Mutex);
  usmPoolHandles.push_back(hPool);
}

void ur_context_handle_t_::removeUsmPool(ur_usm_pool_handle_t hPool) {
  UR_LOG(INFO, "Removing USM pool {} from context:{}", hPool, this)
  std::scoped_lock<ur_shared_mutex> lock(Mutex);
  usmPoolHandles.remove(hPool);
}

void ur_context_handle_t_::changeResidentDevice(ur_device_handle_t hDevice,
                                                ur_device_handle_t peerDevice,
                                                bool isAdding) {
  if (!isValidDevice(hDevice)) {
    UR_LOG(INFO,
           "skipped changing peer device in context:%p because "
           "commandDevice:%d is invalid in this context",
           (void *)this, hDevice->Id.value());
    return;
  }

  if (!isValidDevice(peerDevice)) {
    UR_LOG(INFO,
           "skipped changing peer device in context:%p because peerDevice:%d "
           "is invalid in this context",
           (void *)this, peerDevice->Id.value());
    return;
  }

  UR_LOG(INFO, "{} peerDevice:{} in the default pool and {} usmPools",
         isAdding ? "adding" : "removing", peerDevice->Id.value(),
         usmPoolHandles.size())
  defaultUSMPool.changeResidentDevice(hDevice, peerDevice, isAdding);
  for (const auto &hPool : usmPoolHandles) {
    hPool->changeResidentDevice(hDevice, peerDevice, isAdding);
  }
}

std::vector<ur_device_handle_t>
ur_context_handle_t_::getDevicesWhoseAllocationsCanBeAccessedFrom(
    ur_device_handle_t hDevice) {
  UR_FASSERT(hDevice != nullptr && hDevice->Id.has_value(),
             "invalid device handle");

  std::vector<ur_device_handle_t_::PeerStatus> peers;
  {
    std::scoped_lock<ur_shared_mutex> lock(hDevice->Mutex);
    peers = hDevice->peers;
  }

  std::vector<ur_device_handle_t> retVal;
  std::copy_if(
      std::begin(hDevices), std::end(hDevices), std::back_inserter(retVal),
      [&](ur_device_handle_t peerCandidateDevice) {
        const auto candidateId = peerCandidateDevice->Id.value();
        UR_FASSERT(candidateId < peers.size(),
                   "there is no device:"
                       << candidateId << " in peers table, number of devices:"
                       << peers.size());
        return peers[candidateId] == ur_device_handle_t_::PeerStatus::ENABLED;
      });

  return retVal;
}

std::vector<ur_device_handle_t>
ur_context_handle_t_::getDevicesWhichCanAccessAllocationsPresentOn(
    ur_device_handle_t hDevice) {
  UR_FASSERT(hDevice != nullptr && hDevice->Id.has_value(),
             "invalid device handle");

  const auto hDeviceId = hDevice->Id.value();
  std::vector<ur_device_handle_t> retVal;
  std::copy_if(
      std::begin(hDevices), std::end(hDevices), std::back_inserter(retVal),
      [&](ur_device_handle_t peerCandidateDevice) {
        const auto candidateId = peerCandidateDevice->Id.value();
        UR_FASSERT(
            hDeviceId < peerCandidateDevice->peers.size(),
            "there is no device:"
                << hDeviceId << " in peers table of device:" << candidateId
                << ", number of devices:" << peerCandidateDevice->peers.size());
        std::scoped_lock<ur_shared_mutex> lock(peerCandidateDevice->Mutex);
        return peerCandidateDevice->peers[hDeviceId] ==
               ur_device_handle_t_::PeerStatus::ENABLED;
      });

  return retVal;
}

namespace ur::level_zero {
ur_result_t urContextCreate(uint32_t deviceCount,
                            const ur_device_handle_t *phDevices,
                            const ur_context_properties_t * /*pProperties*/,
                            ur_context_handle_t *phContext) {
  try {

    ur_platform_handle_t hPlatform = phDevices[0]->Platform;
    ZeStruct<ze_context_desc_t> contextDesc{};

    ze_context_handle_t zeContext{};
    ZE2UR_CALL(zeContextCreate,
               (hPlatform->ZeDriver, &contextDesc, &zeContext));
    UR_LOG(INFO, "ZE context created with {} devices", deviceCount);

    *phContext =
        new ur_context_handle_t_(zeContext, deviceCount, phDevices, true);
    {
      std::scoped_lock<ur_shared_mutex> Lock(hPlatform->ContextsMutex);
      hPlatform->Contexts.push_back(*phContext);
    }
    return UR_RESULT_SUCCESS;
  } catch (...) {
    UR_DFAILURE("creating context failed");
    return exceptionToResult(std::current_exception());
  }
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
  auto Platform = hContext->getPlatform();
  auto &Contexts = Platform->Contexts;
  {
    std::scoped_lock<ur_shared_mutex> Lock(Platform->ContextsMutex);
    auto It = std::find(Contexts.begin(), Contexts.end(), hContext);
    UR_ASSERT(It != Contexts.end(), UR_RESULT_ERROR_INVALID_CONTEXT);
    Contexts.erase(It);
  }
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
