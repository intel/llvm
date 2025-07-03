//===--------- queue.hpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "context.hpp"
#include "device.hpp"

#include <vector>

struct ur_queue_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_command_queue;
  native_type CLQueue;
  ur_context_handle_t Context;
  ur_device_handle_t Device;
  // Used to keep a handle to the default queue alive if it is different
  std::optional<ur_queue_handle_t> DeviceDefault = std::nullopt;
  bool IsNativeHandleOwned = true;
  // Used to implement UR_QUEUE_INFO_EMPTY query
  bool IsInOrder;
  ur_event_handle_t LastEvent = nullptr;
  ur::RefCount RefCount;

  ur_queue_handle_t_(native_type Queue, ur_context_handle_t Ctx,
                     ur_device_handle_t Dev, bool InOrder)
      : handle_base(), CLQueue(Queue), Context(Ctx), Device(Dev),
        IsInOrder(InOrder) {
    urDeviceRetain(Device);
    urContextRetain(Context);
  }

  static ur_result_t makeWithNative(native_type NativeQueue,
                                    ur_context_handle_t Context,
                                    ur_device_handle_t Device,
                                    ur_queue_handle_t &Queue);

  ~ur_queue_handle_t_() {
    urDeviceRelease(Device);
    urContextRelease(Context);
    if (IsNativeHandleOwned) {
      clReleaseCommandQueue(CLQueue);
    }
    if (DeviceDefault.has_value()) {
      urQueueRelease(*DeviceDefault);
    }
  }

  // Stores last event for in-order queues. Has no effect if queue is Out Of
  // Order. The last event is used to implement UR_QUEUE_INFO_EMPTY query.
  ur_result_t storeLastEvent(ur_event_handle_t Event) {
    if (!IsInOrder) {
      return UR_RESULT_SUCCESS;
    }
    if (LastEvent) {
      UR_RETURN_ON_FAILURE(urEventRelease(LastEvent));
    }
    LastEvent = Event;
    if (LastEvent) {
      UR_RETURN_ON_FAILURE(urEventRetain(LastEvent));
    }
    return UR_RESULT_SUCCESS;
  }
};
