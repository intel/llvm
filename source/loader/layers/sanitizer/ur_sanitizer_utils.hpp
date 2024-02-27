/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_utils.hpp
 *
 */

#pragma once

#include "common.hpp"

namespace ur_sanitizer_layer {

ur_context_handle_t getContext(ur_queue_handle_t Queue);
ur_device_handle_t getDevice(ur_queue_handle_t Queue);
ur_program_handle_t getProgram(ur_kernel_handle_t Kernel);
size_t getLocalMemorySize(ur_device_handle_t Device);
std::string getKernelName(ur_kernel_handle_t Kernel);
ur_device_handle_t getUSMAllocDevice(ur_context_handle_t Context,
                                     const void *MemPtr);
DeviceType getDeviceType(ur_device_handle_t Device);

} // namespace ur_sanitizer_layer
