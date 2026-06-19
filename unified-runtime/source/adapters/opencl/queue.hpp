//===--------- queue.hpp - OpenCL Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "context.hpp"
#include "device.hpp"

#include <vector>

namespace ur::opencl {

struct ur_queue_handle_t_ : handle_base {
  using native_type = cl_command_queue;
  native_type CLQueue;
  ur_context_handle_t_ *Context;
  ur_device_handle_t_ *Device;
  // Used to keep a handle to the default queue alive if it is different
  std::optional<ur_queue_handle_t_ *> DeviceDefault = std::nullopt;
  bool IsNativeHandleOwned = true;
  // Used to implement UR_QUEUE_INFO_EMPTY query
  bool IsInOrder;
  ur_event_handle_t_ *LastEvent = nullptr;
  ur::RefCount RefCount;

  ur_queue_handle_t_(const ur_queue_handle_t_ &) = delete;
  ur_queue_handle_t_ &operator=(const ur_queue_handle_t_ &) = delete;

  ur_queue_handle_t_(native_type Queue, ur_context_handle_t_ *Ctx,
                     ur_device_handle_t_ *Dev, bool InOrder)
      : handle_base(), CLQueue(Queue), Context(Ctx), Device(Dev),
        IsInOrder(InOrder) {
    ur::opencl::urDeviceRetain(ur_cast<ur_device_handle_t>(Device));
    ur::opencl::urContextRetain(ur_cast<ur_context_handle_t>(Context));
  }

  static ur_result_t makeWithNative(native_type NativeQueue,
                                    ur_context_handle_t Context,
                                    ur_device_handle_t Device,
                                    ur_queue_handle_t &Queue);

  ~ur_queue_handle_t_() {
    ur::opencl::urDeviceRelease(ur_cast<ur_device_handle_t>(Device));
    ur::opencl::urContextRelease(ur_cast<ur_context_handle_t>(Context));
    if (IsNativeHandleOwned) {
      clReleaseCommandQueue(CLQueue);
    }
    if (DeviceDefault.has_value()) {
      ur::opencl::urQueueRelease(ur_cast<ur_queue_handle_t>(*DeviceDefault));
    }
  }

  // Stores last event for in-order queues. Has no effect if queue is Out Of
  // Order. The last event is used to implement UR_QUEUE_INFO_EMPTY query.
  ur_result_t storeLastEvent(ur_event_handle_t Event) {
    if (!IsInOrder) {
      return UR_RESULT_SUCCESS;
    }
    if (LastEvent) {
      UR_RETURN_ON_FAILURE(
          ur::opencl::urEventRelease(ur_cast<ur_event_handle_t>(LastEvent)));
    }
    LastEvent = ur_cast<ur_event_handle_t_ *>(Event);
    if (LastEvent) {
      UR_RETURN_ON_FAILURE(
          ur::opencl::urEventRetain(ur_cast<ur_event_handle_t>(LastEvent)));
    }
    return UR_RESULT_SUCCESS;
  }
};

} // namespace ur::opencl
