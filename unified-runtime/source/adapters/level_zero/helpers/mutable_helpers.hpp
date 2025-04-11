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

#include "../common.hpp"
#include "logger/ur_logger.hpp"
#include <ur/ur.hpp>

ur_result_t setMutableOffsetDesc(
    std::unique_ptr<ZeStruct<ze_mutable_global_offset_exp_desc_t>> &Desc,
    uint32_t Dim, size_t *NewGlobalWorkOffset, const void *NextDesc,
    uint64_t CommandID);

ur_result_t setMutableGroupSizeDesc(
    std::unique_ptr<ZeStruct<ze_mutable_group_size_exp_desc_t>> &Desc,
    uint32_t Dim, uint32_t *NewLocalWorkSize, const void *NextDesc,
    uint64_t CommandID);

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
