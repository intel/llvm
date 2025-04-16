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

using desc_storage_t = std::vector<std::variant<
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>>,
    std::unique_ptr<ZeStruct<ze_mutable_global_offset_exp_desc_t>>,
    std::unique_ptr<ZeStruct<ze_mutable_group_size_exp_desc_t>>,
    std::unique_ptr<ZeStruct<ze_mutable_group_count_exp_desc_t>>>>;

ur_result_t updateKernelHandle(
    ur_kernel_handle_t NewKernel,
    std::function<ur_result_t(ur_kernel_handle_t, ze_kernel_handle_t &)>
        getZeKernel,
    ur_platform_handle_t Platform, ze_command_list_handle_t ZeCommandList,
    kernel_command_handle *Command);

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

ur_result_t updateKernelSizes(
    const ur_exp_command_buffer_update_kernel_launch_desc_t commandDesc,
    kernel_command_handle *command, void **nextDesc,
    ze_group_count_t &zeThreadGroupDimensionsList,
    std::function<ur_result_t(ur_kernel_handle_t, ze_kernel_handle_t &)>
        getZeKernel,
    ur_device_handle_t device, desc_storage_t &descs);

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
ur_result_t setMutableOffsetDesc(
    std::unique_ptr<ZeStruct<ze_mutable_global_offset_exp_desc_t>> &Desc,
    uint32_t Dim, size_t *NewGlobalWorkOffset, const void *NextDesc,
    uint64_t CommandID);

ur_result_t setMutableGroupSizeDesc(
    std::unique_ptr<ZeStruct<ze_mutable_group_size_exp_desc_t>> &Desc,
    uint32_t Dim, uint32_t *NewLocalWorkSize, const void *NextDesc,
    uint64_t CommandID);

ur_result_t updateKernelArguments(
    const ur_exp_command_buffer_update_kernel_launch_desc_t commandDesc,
    kernel_command_handle *command, void **nextDesc,
    std::function<ur_result_t(
        ur_mem_handle_t, const ur_kernel_arg_mem_obj_properties_t *, char **&)>
        getMemPtr,
    desc_storage_t &descs);
ur_result_t setMutableGroupCountDesc(
    std::unique_ptr<ZeStruct<ze_mutable_group_count_exp_desc_t>> &Desc,
    ze_group_count_t *ZeThreadGroupDimensions, const void *NextDesc,
    uint64_t CommandID);

ur_result_t setMutableMemObjArgDesc(
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>> &Desc,
    uint32_t argIndex, const void *pArgValue, const void *NextDesc,
    uint64_t CommandID);

ur_result_t setMutablePointerArgDesc(
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>> &Desc,
    const ur_exp_command_buffer_update_pointer_arg_desc_t &NewPointerArgDesc,
    const void *NextDesc, uint64_t CommandID);

ur_result_t setMutableValueArgDesc(
    std::unique_ptr<ZeStruct<ze_mutable_kernel_argument_exp_desc_t>> &Desc,
    const ur_exp_command_buffer_update_value_arg_desc_t &NewValueArgDesc,
    const void *NextDesc, uint64_t CommandID);
