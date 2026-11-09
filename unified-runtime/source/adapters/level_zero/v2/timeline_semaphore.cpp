//===--------- timeline_semaphore.cpp - Level Zero Adapter ----------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "timeline_semaphore.hpp"
#include "../common/platform.hpp"
#include "ur_interface_loader.hpp"

#include <algorithm>
#include <mutex>
#include <new>
#include <shared_mutex>

namespace ur::level_zero::v2 {

static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t) &&
                  std::atomic<uint64_t>::is_always_lock_free,
              "the semaphore value must be a plain 64-bit word in memory");

ur_exp_timeline_semaphore_handle_t_::ur_exp_timeline_semaphore_handle_t_(
    ur_context_handle_t hContext, uint64_t initialValue, uint64_t maxValue)
    : hContext(hContext), maxValue(maxValue) {
  ZeStruct<ze_host_mem_alloc_desc_t> HostDesc;
  void *Ptr = nullptr;
  ZE2UR_CALL_THROWS(zeMemAllocHost, (hContext->getZeHandle(), &HostDesc,
                                     sizeof(uint64_t), sizeof(uint64_t), &Ptr));
  value = new (Ptr) std::atomic<uint64_t>(initialValue);
}

ur_exp_timeline_semaphore_handle_t_::~ur_exp_timeline_semaphore_handle_t_() {
  // The events refer to the allocation, so destroy them first.
  waitEvents.clear();
  if (value && checkL0LoaderTeardown()) {
    ZE_CALL_NOCHECK(zeMemFree, (hContext->getZeHandle(), value));
  }
}

ze_event_handle_t
ur_exp_timeline_semaphore_handle_t_::getWaitEvent(ur_device_handle_t hDevice,
                                                  uint64_t target) {
  auto Key = std::make_pair(hDevice->ZeDevice, target);
  {
    std::shared_lock<ur_shared_mutex> Lock(waitEventsMutex);
    if (auto It = waitEvents.find(Key); It != waitEvents.end())
      return It->second.get();
  }

  std::scoped_lock<ur_shared_mutex> Lock(waitEventsMutex);
  if (auto It = waitEvents.find(Key); It != waitEvents.end())
    return It->second.get();

  ze_event_counter_based_external_sync_allocation_desc_t SyncDesc = {};
  SyncDesc.stype =
      ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_EXTERNAL_SYNC_ALLOCATION_DESC;
  SyncDesc.deviceAddress = getStorage();
  SyncDesc.hostAddress = getStorage();
  SyncDesc.completionValue = target;

  ze_event_counter_based_desc_t Desc = {};
  Desc.stype = ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC;
  Desc.pNext = &SyncDesc;
  Desc.flags = ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE |
               ZE_EVENT_COUNTER_BASED_FLAG_NON_IMMEDIATE |
               ZE_EVENT_COUNTER_BASED_FLAG_HOST_VISIBLE;
  Desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  Desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

  ze_event_handle_t ZeEvent = nullptr;
  ZE2UR_CALL_THROWS(
      zeEventCounterBasedCreate,
      (hContext->getZeHandle(), hDevice->ZeDevice, &Desc, &ZeEvent));
  auto [It, _] = waitEvents.emplace(Key, raii::ze_event_handle_t(ZeEvent));
  return It->second.get();
}

ur_result_t ur_exp_timeline_semaphore_handle_t_::release() {
  if (!RefCount.release())
    return UR_RESULT_SUCCESS;

  delete this;
  return UR_RESULT_SUCCESS;
}

ur_result_t urTimelineSemaphoreCreateExp(
    ::ur_context_handle_t hContextOpque, uint64_t initialValue,
    ::ur_exp_timeline_semaphore_handle_t *phSemaphore) try {
  auto hContext = v2_cast(hContextOpque);
  uint64_t MaxValue = UINT64_MAX;
  for (auto hDevice : hContext->getDevices()) {
    const auto &Caps = hDevice->ZeTimelineSemaphoreCaps.get();
    if (!Caps.Supported)
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    MaxValue = std::min(MaxValue, Caps.MaxValue);
  }
  if (initialValue > MaxValue)
    return UR_RESULT_ERROR_INVALID_VALUE;

  *phSemaphore = v2_cast(new ur_exp_timeline_semaphore_handle_t_(
      hContext, initialValue, MaxValue));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urTimelineSemaphoreRetainExp(::ur_exp_timeline_semaphore_handle_t hSemaphore) {
  v2_cast(hSemaphore)->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t urTimelineSemaphoreReleaseExp(
    ::ur_exp_timeline_semaphore_handle_t hSemaphore) try {
  return v2_cast(hSemaphore)->release();
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t
urTimelineSemaphoreHostWaitExp(::ur_exp_timeline_semaphore_handle_t hSemaphore,
                               uint64_t value) try {
  auto Semaphore = v2_cast(hSemaphore);
  if (value > Semaphore->getMaxValue())
    return UR_RESULT_ERROR_INVALID_VALUE;
  if (Semaphore->isReached(value))
    return UR_RESULT_SUCCESS;

  auto hDevice = Semaphore->getContext()->getDevices().front();
  ze_event_handle_t ZeEvent = Semaphore->getWaitEvent(hDevice, value);
  ZE2UR_CALL(zeEventHostSynchronize, (ZeEvent, UINT64_MAX));
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urTimelineSemaphoreHostSignalExp(
    ::ur_exp_timeline_semaphore_handle_t hSemaphore, uint64_t value) {
  auto Semaphore = v2_cast(hSemaphore);
  if (value > Semaphore->getMaxValue())
    return UR_RESULT_ERROR_INVALID_VALUE;
  Semaphore->hostSignal(value);
  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero::v2
