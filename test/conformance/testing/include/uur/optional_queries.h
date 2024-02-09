/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file optional_queries.h
 *
 */

// Auto-generated file, do not edit.

#pragma once

#include <algorithm>
#include <array>
#include <ur_api.h>

namespace uur {

template <class T> bool isQueryOptional(T) { return false; }

constexpr std::array optional_ur_device_info_t = {
    UR_DEVICE_INFO_DEVICE_ID,
    UR_DEVICE_INFO_MEMORY_CLOCK_RATE,
    UR_DEVICE_INFO_GLOBAL_MEM_FREE,
    UR_DEVICE_INFO_UUID,
    UR_DEVICE_INFO_PCI_ADDRESS,
    UR_DEVICE_INFO_GPU_EU_COUNT,
    UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH,
    UR_DEVICE_INFO_GPU_EU_SLICES,
    UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE,
    UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE,
    UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU,
    UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH,
    UR_DEVICE_INFO_MEMORY_BUS_WIDTH,
    UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP,
    UR_DEVICE_INFO_IP_VERSION,
    UR_DEVICE_INFO_COMPONENT_DEVICES,
    UR_DEVICE_INFO_COMPOSITE_DEVICE,
};

template <> inline bool isQueryOptional(ur_device_info_t query) {
    return std::find(optional_ur_device_info_t.begin(),
                     optional_ur_device_info_t.end(),
                     query) != optional_ur_device_info_t.end();
}

constexpr std::array optional_ur_context_info_t = {
    UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
    UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
    UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES,
    UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES,
};

template <> inline bool isQueryOptional(ur_context_info_t query) {
    return std::find(optional_ur_context_info_t.begin(),
                     optional_ur_context_info_t.end(),
                     query) != optional_ur_context_info_t.end();
}

constexpr std::array optional_ur_usm_alloc_info_t = {
    UR_USM_ALLOC_INFO_POOL,
};

template <> inline bool isQueryOptional(ur_usm_alloc_info_t query) {
    return std::find(optional_ur_usm_alloc_info_t.begin(),
                     optional_ur_usm_alloc_info_t.end(),
                     query) != optional_ur_usm_alloc_info_t.end();
}

constexpr std::array optional_ur_program_info_t = {
    UR_PROGRAM_INFO_NUM_KERNELS,
    UR_PROGRAM_INFO_KERNEL_NAMES,
};

template <> inline bool isQueryOptional(ur_program_info_t query) {
    return std::find(optional_ur_program_info_t.begin(),
                     optional_ur_program_info_t.end(),
                     query) != optional_ur_program_info_t.end();
}

constexpr std::array optional_ur_kernel_info_t = {
    UR_KERNEL_INFO_NUM_REGS,
};

template <> inline bool isQueryOptional(ur_kernel_info_t query) {
    return std::find(optional_ur_kernel_info_t.begin(),
                     optional_ur_kernel_info_t.end(),
                     query) != optional_ur_kernel_info_t.end();
}

constexpr std::array optional_ur_kernel_group_info_t = {
    UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE,
    UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE,
    UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE,
};

template <> inline bool isQueryOptional(ur_kernel_group_info_t query) {
    return std::find(optional_ur_kernel_group_info_t.begin(),
                     optional_ur_kernel_group_info_t.end(),
                     query) != optional_ur_kernel_group_info_t.end();
}

constexpr std::array optional_ur_queue_info_t = {
    UR_QUEUE_INFO_EMPTY,
};

template <> inline bool isQueryOptional(ur_queue_info_t query) {
    return std::find(optional_ur_queue_info_t.begin(),
                     optional_ur_queue_info_t.end(),
                     query) != optional_ur_queue_info_t.end();
}

} // namespace uur
