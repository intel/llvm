/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_utils.hpp
 *
 */

#pragma once

#include "sanitizer_libdevice.hpp"
#include "ur_api.h"

#include <string>
#include <vector>

namespace ur_sanitizer_layer {

struct ManagedQueue {
    ManagedQueue(ur_context_handle_t Context, ur_device_handle_t Device);
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
size_t GetDeviceLocalMemorySize(ur_device_handle_t Device);
ur_program_handle_t GetProgram(ur_kernel_handle_t Kernel);
ur_device_handle_t GetUSMAllocDevice(ur_context_handle_t Context,
                                     const void *MemPtr);
uint32_t GetKernelNumArgs(ur_kernel_handle_t Kernel);
size_t GetKernelLocalMemorySize(ur_kernel_handle_t Kernel,
                                ur_device_handle_t Device);
size_t GetKernelPrivateMemorySize(ur_kernel_handle_t Kernel,
                                  ur_device_handle_t Device);
size_t GetVirtualMemGranularity(ur_context_handle_t Context,
                                ur_device_handle_t Device);

ur_result_t
EnqueueUSMBlockingSet(ur_queue_handle_t Queue, void *Ptr, char Value,
                      size_t Size, uint32_t NumEvents = 0,
                      const ur_event_handle_t *EventWaitList = nullptr,
                      ur_event_handle_t *OutEvent = nullptr);

} // namespace ur_sanitizer_layer
