/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file urinfo.cpp
 *
 */

#pragma once

#include "utils.hpp"
#include <cstdlib>
#include <string_view>
#include <ur_api.h>

namespace urinfo {
inline void printLoaderConfigInfos(ur_loader_config_handle_t hLoaderConfig,
                                   std::string_view prefix = "  ") {
    std::cout << prefix;
    printLoaderConfigInfo<char[]>(hLoaderConfig,
                                  UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS);
}
inline void printAdapterInfos(ur_adapter_handle_t hAdapter,
                              std::string_view prefix = "  ") {
    std::cout << prefix;
    printAdapterInfo<ur_adapter_backend_t>(hAdapter, UR_ADAPTER_INFO_BACKEND);
}

inline void printPlatformInfos(ur_platform_handle_t hPlatform,
                               std::string_view prefix = "    ") {
    std::cout << prefix;
    printPlatformInfo<char[]>(hPlatform, UR_PLATFORM_INFO_NAME);
    std::cout << prefix;
    printPlatformInfo<char[]>(hPlatform, UR_PLATFORM_INFO_VENDOR_NAME);
    std::cout << prefix;
    printPlatformInfo<char[]>(hPlatform, UR_PLATFORM_INFO_VERSION);
    std::cout << prefix;
    printPlatformInfo<char[]>(hPlatform, UR_PLATFORM_INFO_EXTENSIONS);
    std::cout << prefix;
    printPlatformInfo<char[]>(hPlatform, UR_PLATFORM_INFO_PROFILE);
    std::cout << prefix;
    printPlatformInfo<ur_platform_backend_t>(hPlatform,
                                             UR_PLATFORM_INFO_BACKEND);
}

inline void printDeviceInfos(ur_device_handle_t hDevice,
                             std::string_view prefix = "      ") {
    std::cout << prefix;
    printDeviceInfo<ur_device_type_t>(hDevice, UR_DEVICE_INFO_TYPE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_VENDOR_ID);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_DEVICE_ID);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_COMPUTE_UNITS);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS);
    std::cout << prefix;
    printDeviceInfo<size_t[]>(hDevice, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE);
    std::cout << prefix;
    printDeviceInfo<ur_device_fp_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_SINGLE_FP_CONFIG);
    std::cout << prefix;
    printDeviceInfo<ur_device_fp_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_HALF_FP_CONFIG);
    std::cout << prefix;
    printDeviceInfo<ur_device_fp_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_DOUBLE_FP_CONFIG);
    std::cout << prefix;
    printDeviceInfo<ur_queue_flags_t>(hDevice, UR_DEVICE_INFO_QUEUE_PROPERTIES);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MEMORY_CLOCK_RATE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_ADDRESS_BITS);
    std::cout << prefix;
    printDeviceInfo<uint64_t>(hDevice, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_IMAGE_SUPPORTED);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_SAMPLERS);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_MAX_PARAMETER_SIZE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN);
    std::cout << prefix;
    printDeviceInfo<ur_device_mem_cache_type_t>(
        hDevice, UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE);
    std::cout << prefix;
    printDeviceInfo<uint64_t>(hDevice, UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE);
    std::cout << prefix;
    printDeviceInfo<uint64_t>(hDevice, UR_DEVICE_INFO_GLOBAL_MEM_SIZE);
    std::cout << prefix;
    printDeviceInfo<uint64_t>(hDevice, UR_DEVICE_INFO_GLOBAL_MEM_FREE);
    std::cout << prefix;
    printDeviceInfo<uint64_t>(hDevice, UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_CONSTANT_ARGS);
    std::cout << prefix;
    printDeviceInfo<ur_device_local_mem_type_t>(hDevice,
                                                UR_DEVICE_INFO_LOCAL_MEM_TYPE);
    std::cout << prefix;
    printDeviceInfo<uint64_t>(hDevice, UR_DEVICE_INFO_LOCAL_MEM_SIZE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice,
                               UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_HOST_UNIFIED_MEMORY);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_ENDIAN_LITTLE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_AVAILABLE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_COMPILER_AVAILABLE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_LINKER_AVAILABLE);
    std::cout << prefix;
    printDeviceInfo<ur_device_exec_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_EXECUTION_CAPABILITIES);
    std::cout << prefix;
    printDeviceInfo<ur_queue_flags_t>(
        hDevice, UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES);
    std::cout << prefix;
    printDeviceInfo<ur_queue_flags_t>(hDevice,
                                      UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_BUILT_IN_KERNELS);
    std::cout << prefix;
    printDeviceInfo<ur_platform_handle_t>(hDevice, UR_DEVICE_INFO_PLATFORM);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_REFERENCE_COUNT);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_IL_VERSION);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_NAME);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_VENDOR);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_DRIVER_VERSION);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_PROFILE);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_VERSION);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_EXTENSIONS);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_PRINTF_BUFFER_SIZE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice,
                               UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC);
    std::cout << prefix;
    printDeviceInfo<ur_device_handle_t>(hDevice, UR_DEVICE_INFO_PARENT_DEVICE);
    std::cout << prefix;
    printDeviceInfo<ur_device_partition_t[]>(
        hDevice, UR_DEVICE_INFO_SUPPORTED_PARTITIONS);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES);
    std::cout << prefix;
    printDeviceInfo<ur_device_affinity_domain_flags_t>(
        hDevice, UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN);
    std::cout << prefix;
    printDeviceInfo<ur_device_partition_property_t[]>(
        hDevice, UR_DEVICE_INFO_PARTITION_TYPE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS);
    std::cout << prefix;
    printDeviceInfo<uint32_t[]>(hDevice, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL);
    std::cout << prefix;
    printDeviceInfo<ur_device_usm_access_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_USM_HOST_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_device_usm_access_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_USM_DEVICE_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_device_usm_access_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_device_usm_access_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_device_usm_access_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_UUID);
    std::cout << prefix;
    printDeviceInfo<char[]>(hDevice, UR_DEVICE_INFO_PCI_ADDRESS);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_GPU_EU_COUNT);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_GPU_EU_SLICES);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_IMAGE_SRGB);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_BUILD_ON_SUBDEVICE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_ATOMIC_64);
    std::cout << prefix;
    printDeviceInfo<ur_memory_order_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES);
    std::cout << prefix;
    printDeviceInfo<ur_memory_scope_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES);
    std::cout << prefix;
    printDeviceInfo<ur_memory_order_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES);
    std::cout << prefix;
    printDeviceInfo<ur_memory_scope_capability_flags_t>(
        hDevice, UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_BFLOAT16);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_MEMORY_BUS_WIDTH);
    std::cout << prefix;
    printDeviceInfo<size_t[3]>(hDevice, UR_DEVICE_INFO_MAX_WORK_GROUPS_3D);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_ASYNC_BARRIER);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice,
                               UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_IP_VERSION);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_ESIMD_SUPPORT);
    std::cout << prefix;
    printDeviceInfo<ur_device_handle_t[]>(hDevice,
                                          UR_DEVICE_INFO_COMPONENT_DEVICES);
    std::cout << prefix;
    printDeviceInfo<ur_device_handle_t>(hDevice,
                                        UR_DEVICE_INFO_COMPOSITE_DEVICE);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice,
                               UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice,
                               UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice, UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice,
                            UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP);
    std::cout << prefix;
    printDeviceInfo<size_t>(hDevice, UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice, UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(hDevice,
                               UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<uint32_t>(hDevice,
                              UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_INTEROP_MEMORY_IMPORT_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_INTEROP_MEMORY_EXPORT_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_INTEROP_SEMAPHORE_IMPORT_SUPPORT_EXP);
    std::cout << prefix;
    printDeviceInfo<ur_bool_t>(
        hDevice, UR_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP);
}
} // namespace urinfo
