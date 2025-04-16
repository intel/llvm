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

ur_result_t updateCommandBufferUnlocked(
    std::function<ur_result_t(ur_kernel_handle_t, ze_kernel_handle_t &)>
        GetZeKernel,
    std::function<
        ur_result_t(ur_mem_handle_t MemObj,
                    const ur_kernel_arg_mem_obj_properties_t *Properties,
                    char **&ZeHandlePtr)>
        GetMemPtr,
    ze_command_list_handle_t ZeCommandList, ur_platform_handle_t Platform,
    ur_device_handle_t Device, uint32_t NumKernelUpdates,
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
    std::function<ur_result_t(ur_kernel_handle_t, ze_kernel_handle_t &)>
        getZeKernel,
    std::unique_ptr<kernel_command_handle> &Command);