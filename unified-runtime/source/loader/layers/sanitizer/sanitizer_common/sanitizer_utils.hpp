/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_utils.hpp
 *
 */

#pragma once

#include "sanitizer_libdevice.hpp"
#include "unified-runtime/ur_api.h"
#include "ur/ur.hpp"
#include "ur_sanitizer_layer.hpp"

#include <string>
#include <vector>

namespace ur_sanitizer_layer {

// Accumulates events whose release must be deferred until a safe point
// (e.g., context release). L0 may not retain input events passed to
// pfnEventsWait long enough for the caller to release them immediately.
struct DeferredEventList {
  void add(const std::vector<ur_event_handle_t> &Events) {
    std::scoped_lock<ur_shared_mutex> Lock(Mutex);
    List.insert(List.end(), Events.begin(), Events.end());
  }

  void releaseAll() {
    std::scoped_lock<ur_shared_mutex> Lock(Mutex);
    for (auto &E : List) {
      [[maybe_unused]] auto Result =
          getContext()->urDdiTable.Event.pfnRelease(E);
      assert(Result == UR_RESULT_SUCCESS);
    }
    List.clear();
  }

private:
  ur_shared_mutex Mutex;
  std::vector<ur_event_handle_t> List;
};

struct ManagedQueue {
  ManagedQueue(ur_context_handle_t Context, ur_device_handle_t Device,
               bool IsOutOfOrder = false);
  ~ManagedQueue();

  // Disable copy semantics
  ManagedQueue(const ManagedQueue &) = delete;
  ManagedQueue &operator=(const ManagedQueue &) = delete;

  operator ur_queue_handle_t() { return Handle; }

private:
  ur_queue_handle_t Handle = nullptr;
};

ur_context_handle_t GetContext(ur_queue_handle_t Queue);
ur_context_handle_t GetContext(ur_program_handle_t Program);
ur_context_handle_t GetContext(ur_kernel_handle_t Kernel);
ur_device_handle_t GetDevice(ur_queue_handle_t Queue);
std::vector<ur_device_handle_t> GetDevices(ur_context_handle_t Context);
std::vector<ur_device_handle_t> GetDevices(ur_program_handle_t Program);
DeviceType GetDeviceType(ur_context_handle_t Context,
                         ur_device_handle_t Device);
ur_device_handle_t GetParentDevice(ur_device_handle_t Device);
bool GetDeviceUSMCapability(ur_device_handle_t Device,
                            ur_device_info_t Feature);
std::string GetKernelName(ur_kernel_handle_t Kernel);
size_t GetSubGroupSize(ur_kernel_handle_t Kernel, ur_device_handle_t Device);
size_t GetDeviceLocalMemorySize(ur_device_handle_t Device);
ur_program_handle_t GetProgram(ur_kernel_handle_t Kernel);
bool IsUSM(ur_context_handle_t Context, const void *MemPtr);
bool IsHostUSM(ur_context_handle_t Context, const void *MemPtr);
bool IsDeviceUSM(ur_context_handle_t Context, const void *MemPtr);
ur_device_handle_t GetUSMAllocDevice(ur_context_handle_t Context,
                                     const void *MemPtr);
// Get the device of MemPtr. If MemPtr is host USM, then return the device
// of Queue
ur_device_handle_t GetUSMAllocDevice(ur_queue_handle_t Queue,
                                     const void *MemPtr);
uint32_t GetKernelNumArgs(ur_kernel_handle_t Kernel);
size_t GetKernelLocalMemorySize(ur_kernel_handle_t Kernel,
                                ur_device_handle_t Device);
size_t GetKernelPrivateMemorySize(ur_kernel_handle_t Kernel,
                                  ur_device_handle_t Device);
size_t GetVirtualMemGranularity(ur_context_handle_t Context,
                                ur_device_handle_t Device);

template <class T>
ur_result_t EnqueueUSMSet(ur_queue_handle_t Queue, void *Ptr, T *Value,
                          size_t Size, uint32_t NumEvents = 0,
                          const ur_event_handle_t *EventWaitList = nullptr,
                          ur_event_handle_t *OutEvent = nullptr) {
  assert(Size % sizeof(T) == 0);
  return getContext()->urDdiTable.Enqueue.pfnUSMFill(
      Queue, Ptr, sizeof(T), Value, Size, NumEvents, EventWaitList, OutEvent);
}

ur_result_t EnqueueUSMSetZero(ur_queue_handle_t Queue, void *Ptr, size_t Size,
                              uint32_t NumEvents = 0,
                              const ur_event_handle_t *EventWaitList = nullptr,
                              ur_event_handle_t *OutEvent = nullptr);

void PrintUrBuildLogIfError(ur_result_t Result, ur_program_handle_t Program,
                            ur_device_handle_t *Devices, size_t NumDevices);

} // namespace ur_sanitizer_layer
