//===--------- mutable_helpers.hpp - Level Zero Adapter -------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>
#include <ze_api.h>

#include <utility>

#include "../command_buffer_command.hpp"
#include "../common.hpp"
#include "logger/ur_logger.hpp"
#include <ur/ur.hpp>

using device_ptr_storage_t = std::vector<std::unique_ptr<char *>>;

ur_result_t updateCommandBufferUnlocked(
    ur_result_t (*getZeKernel)(ur_kernel_handle_t, ze_kernel_handle_t &,
                               ur_device_handle_t),
    ur_result_t (*GetMemPtr)(ur_mem_handle_t,
                             const ur_kernel_arg_mem_obj_properties_t *,
                             char **&, ur_device_handle_t,
                             device_ptr_storage_t *),
    ze_command_list_handle_t ZeCommandList, ur_platform_handle_t Platform,
    ur_device_handle_t Device, device_ptr_storage_t *PtrStorage,
    uint32_t NumKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *CommandDescs);

ur_result_t validateCommandDescUnlocked(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_device_handle_t device,
    bool ZeDriverGlobalOffsetExtensionFound, size_t commandDescSize,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *CommandDescs);

ur_result_t createCommandHandleUnlocked(
    ur_exp_command_buffer_handle_t CommandBuffer,
    ze_command_list_handle_t ZeCommandList, ur_kernel_handle_t Kernel,
    uint32_t WorkDim, const size_t *GlobalWorkSize,
    uint32_t NumKernelAlternatives, ur_kernel_handle_t *KernelAlternatives,
    ur_platform_handle_t Platform,
    ur_result_t (*getZeKernel)(ur_kernel_handle_t, ze_kernel_handle_t &,
                               ur_device_handle_t),
    ur_device_handle_t Device, std::unique_ptr<kernel_command_handle> &Command);