// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED

#include "ur_api.h"
#include <optional>
#include <string>
#include <uur/environment.h>
#include <vector>

namespace uur {

inline size_t RoundUpToNearestFactor(size_t num, size_t factor) {
    return ((num + factor - 1) / factor) * factor;
}

/// @brief Make a string a valid identifier for gtest.
/// @param str The string to sanitize.
inline std::string GTestSanitizeString(const std::string &str) {
    auto str_cpy = str;
    std::replace_if(
        str_cpy.begin(), str_cpy.end(), [](char c) { return !std::isalnum(c); },
        '_');
    return str_cpy;
}

inline ur_platform_handle_t GetPlatform() {
    return PlatformEnvironment::instance->platform;
}

template <class T, class ObjectTy, class InfoTy, class Callable>
ur_result_t GetInfo(ObjectTy object, InfoTy info, Callable cb, T &out_value) {
    // first get the size of the info
    size_t size = 0;
    ur_result_t result = cb(object, info, 0, nullptr, &size);
    if (result != UR_RESULT_SUCCESS || size == 0) {
        return result;
    }

    // special case for strings
    if constexpr (std::is_same_v<std::string, T>) {
        std::vector<char> data(size);
        result = cb(object, info, size, data.data(), nullptr);
        if (result != UR_RESULT_SUCCESS) {
            return result;
        }
        out_value = std::string(data.data(), data.size());
        return UR_RESULT_SUCCESS;
    } else {
        if (size != sizeof(T)) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }
        T value{};
        result = cb(object, info, sizeof(T), &value, nullptr);
        if (result != UR_RESULT_SUCCESS) {
            return result;
        }
        out_value = value;
        return UR_RESULT_SUCCESS;
    }
}

template <class T>
auto GetPlatformInfo =
    [](ur_platform_handle_t platform, ur_platform_info_t info, T &out_value) {
        return GetInfo(platform, info, urPlatformGetInfo, out_value);
    };

template <class T>
auto GetContextInfo =
    [](ur_context_handle_t context, ur_context_info_t info, T &out_value) {
        return GetInfo(context, info, urContextGetInfo, out_value);
    };

template <class T>
auto GetDeviceInfo =
    [](ur_device_handle_t device, ur_device_info_t info, T &out_value) {
        return GetInfo(device, info, urDeviceGetInfo, out_value);
    };

template <class T>
auto GetEventInfo =
    [](ur_event_handle_t event, ur_event_info_t info, T &out_value) {
        return GetInfo(event, info, urEventGetInfo, out_value);
    };

template <class T>
auto GetQueueInfo =
    [](ur_queue_handle_t queue, ur_queue_info_t info, T &out_value) {
        return GetInfo(queue, info, urQueueGetInfo, out_value);
    };

template <class T>
auto GetSamplerInfo =
    [](ur_sampler_handle_t sampler, ur_sampler_info_t info, T &out_value) {
        return GetInfo(sampler, info, urSamplerGetInfo, out_value);
    };

template <class T>
auto GetKernelInfo =
    [](ur_kernel_handle_t kernel, ur_kernel_info_t info, T &out_value) {
        return GetInfo(kernel, info, urKernelGetInfo, out_value);
    };

template <class T>
auto GetProgramInfo =
    [](ur_program_handle_t program, ur_program_info_t info, T &out_value) {
        return GetInfo(program, info, urProgramGetInfo, out_value);
    };

template <class T>
auto GetPoolInfo =
    [](ur_usm_pool_handle_t pool, ur_usm_pool_info_t info, T &out_value) {
        return GetInfo(pool, info, urUSMPoolGetInfo, out_value);
    };

template <class T>
auto GetCommandBufferInfo = [](ur_exp_command_buffer_handle_t cmd_buf,
                               ur_exp_command_buffer_info_t info,
                               T &out_value) {
    return GetInfo(cmd_buf, info, urCommandBufferGetInfoExp, out_value);
};

template <class T>
auto GetCommandBufferCommandInfo =
    [](ur_exp_command_buffer_command_handle_t command,
       ur_exp_command_buffer_command_info_t info, T &out_value) {
        return GetInfo(command, info, urCommandBufferCommandGetInfoExp,
                       out_value);
    };

template <class T>
ur_result_t GetObjectReferenceCount(T object, uint32_t &out_ref_count) {
    if constexpr (std::is_same_v<T, ur_context_handle_t>) {
        return GetContextInfo<uint32_t>(object, UR_CONTEXT_INFO_REFERENCE_COUNT,
                                        out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_device_handle_t>) {
        return GetDeviceInfo<uint32_t>(object, UR_DEVICE_INFO_REFERENCE_COUNT,
                                       out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_event_handle_t>) {
        return GetEventInfo<uint32_t>(object, UR_EVENT_INFO_REFERENCE_COUNT,
                                      out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_queue_handle_t>) {
        return GetQueueInfo<uint32_t>(object, UR_QUEUE_INFO_REFERENCE_COUNT,
                                      out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_sampler_handle_t>) {
        return GetSamplerInfo<uint32_t>(object, UR_SAMPLER_INFO_REFERENCE_COUNT,
                                        out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_kernel_handle_t>) {
        return GetKernelInfo<uint32_t>(object, UR_KERNEL_INFO_REFERENCE_COUNT,
                                       out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_program_handle_t>) {
        return GetProgramInfo<uint32_t>(object, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                        out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_usm_pool_handle_t>) {
        return GetPoolInfo<uint32_t>(object, UR_USM_POOL_INFO_REFERENCE_COUNT,
                                     out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_exp_command_buffer_handle_t>) {
        return GetCommandBufferInfo<uint32_t>(
            object, UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT, out_ref_count);
    }
    if constexpr (std::is_same_v<T, ur_exp_command_buffer_command_handle_t>) {
        return GetCommandBufferCommandInfo<uint32_t>(
            object, UR_EXP_COMMAND_BUFFER_COMMAND_INFO_REFERENCE_COUNT,
            out_ref_count);
    }

    return UR_RESULT_ERROR_INVALID_VALUE;
}

inline std::string GetPlatformName(ur_platform_handle_t hPlatform) {
    std::string platform_name;
    GetPlatformInfo<std::string>(hPlatform, UR_PLATFORM_INFO_NAME,
                                 platform_name);
    return GTestSanitizeString(
        std::string(platform_name.data(), platform_name.size()));
}

inline std::string GetDeviceName(ur_device_handle_t device) {
    std::string device_name;
    GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_NAME, device_name);
    return GTestSanitizeString(device_name);
}

inline std::string GetPlatformAndDeviceName(ur_device_handle_t device) {
    return GetPlatformName(GetPlatform()) + "__" + GetDeviceName(device);
}

ur_result_t GetDeviceType(ur_device_handle_t device,
                          ur_device_type_t &device_type);
ur_result_t GetDeviceVendorId(ur_device_handle_t device, uint32_t &vendor_id);
ur_result_t GetDeviceId(ur_device_handle_t device, uint32_t &device_id);
ur_result_t GetDeviceMaxComputeUnits(ur_device_handle_t device,
                                     uint32_t &max_compute_units);
ur_result_t GetDeviceMaxWorkItemDimensions(ur_device_handle_t device,
                                           uint32_t &max_work_item_dimensions);
ur_result_t GetDeviceMaxWorkItemSizes(ur_device_handle_t device,
                                      std::vector<size_t> &max_work_item_sizes);
ur_result_t GetDeviceMaxWorkGroupSize(ur_device_handle_t device,
                                      size_t &max_work_group_size);
ur_result_t
GetDeviceSingleFPCapabilities(ur_device_handle_t device,
                              ur_device_fp_capability_flags_t &fp_capabilities);
ur_result_t
GetDeviceHalfFPCapabilities(ur_device_handle_t device,
                            ur_device_fp_capability_flags_t &fp_capabilities);
ur_result_t
GetDeviceDoubleFPCapabilities(ur_device_handle_t device,
                              ur_device_fp_capability_flags_t &fp_capabilities);
ur_result_t GetDeviceQueueProperties(ur_device_handle_t device,
                                     ur_queue_flags_t &flags);
ur_result_t GetDevicePreferredVectorWidthChar(ur_device_handle_t device,
                                              uint32_t &pref_width);
ur_result_t GetDevicePreferredVectorWidthInt(ur_device_handle_t device,
                                             uint32_t &pref_width);
ur_result_t GetDevicePreferredVectorWidthLong(ur_device_handle_t device,
                                              uint32_t &pref_width);
ur_result_t GetDevicePreferredVectorWidthFloat(ur_device_handle_t device,
                                               uint32_t &pref_width);
ur_result_t GetDevicePreferredVectorWidthDouble(ur_device_handle_t device,
                                                uint32_t &pref_width);
ur_result_t GetDevicePreferredVectorWidthHalf(ur_device_handle_t device,
                                              uint32_t &pref_width);
ur_result_t GetDeviceNativeVectorWithChar(ur_device_handle_t device,
                                          uint32_t &vec_width);
ur_result_t GetDeviceNativeVectorWithShort(ur_device_handle_t device,
                                           uint32_t &vec_width);
ur_result_t GetDeviceNativeVectorWithInt(ur_device_handle_t device,
                                         uint32_t &vec_width);
ur_result_t GetDeviceNativeVectorWithLong(ur_device_handle_t device,
                                          uint32_t &vec_width);
ur_result_t GetDeviceNativeVectorWithFloat(ur_device_handle_t device,
                                           uint32_t &vec_width);
ur_result_t GetDeviceNativeVectorWithDouble(ur_device_handle_t device,
                                            uint32_t &vec_width);
ur_result_t GetDeviceNativeVectorWithHalf(ur_device_handle_t device,
                                          uint32_t &vec_width);
ur_result_t GetDeviceMaxClockFrequency(ur_device_handle_t device,
                                       uint32_t &max_freq);
ur_result_t GetDeviceMemoryClockRate(ur_device_handle_t device,
                                     uint32_t &mem_clock);
ur_result_t GetDeviceAddressBits(ur_device_handle_t device,
                                 uint32_t &addr_bits);
ur_result_t GetDeviceMaxMemAllocSize(ur_device_handle_t device,
                                     uint64_t &alloc_size);
ur_result_t GetDeviceImageSupport(ur_device_handle_t device,
                                  bool &image_support);
ur_result_t GetDeviceMaxReadImageArgs(ur_device_handle_t device,
                                      uint32_t &read_arg);
ur_result_t GetDeviceMaxWriteImageArgs(ur_device_handle_t device,
                                       uint32_t &write_args);
ur_result_t GetDeviceMaxReadWriteImageArgs(ur_device_handle_t device,
                                           uint32_t &read_write_args);
ur_result_t GetDeviceImage2DMaxWidth(ur_device_handle_t device,
                                     size_t &max_width);
ur_result_t GetDeviceImage2DMaxHeight(ur_device_handle_t device,
                                      size_t &max_height);
ur_result_t GetDeviceImage3DMaxWidth(ur_device_handle_t device,
                                     size_t &max_width);
ur_result_t GetDeviceImage3DMaxHeight(ur_device_handle_t device,
                                      size_t &max_height);
ur_result_t GetDeviceImage3DMaxDepth(ur_device_handle_t device,
                                     size_t &max_depth);
ur_result_t GetDeviceImageMaxBufferSize(ur_device_handle_t device,
                                        size_t &max_buf_size);
ur_result_t GetDeviceImageMaxArraySize(ur_device_handle_t device,
                                       size_t &max_arr_size);
ur_result_t GetDeviceMaxSamplers(ur_device_handle_t device,
                                 uint32_t &max_samplers);
ur_result_t GetDeviceMaxParameterSize(ur_device_handle_t device,
                                      size_t &max_param_size);
ur_result_t GetDeviceMemBaseAddressAlign(ur_device_handle_t device,
                                         uint32_t &align);
ur_result_t GetDeviceMemCacheType(ur_device_handle_t device,
                                  ur_device_mem_cache_type_t &cache_type);
ur_result_t GetDeviceMemCachelineSize(ur_device_handle_t device,
                                      uint32_t &cache_line_size);
ur_result_t GetDeviceMemCacheSize(ur_device_handle_t device,
                                  uint64_t &cache_size);
ur_result_t GetDeviceGlobalMemSize(ur_device_handle_t device,
                                   uint64_t &mem_size);
ur_result_t GetDeviceGlobalMemFree(ur_device_handle_t device,
                                   uint64_t &mem_free);
ur_result_t GetDeviceMaxConstantBufferSize(ur_device_handle_t device,
                                           uint64_t &buf_size);
ur_result_t GetDeviceMaxConstantArgs(ur_device_handle_t device, uint32_t &args);
ur_result_t GetDeviceLocalMemType(ur_device_handle_t device,
                                  ur_device_local_mem_type_t &type);
ur_result_t GetDeviceLocalMemSize(ur_device_handle_t device, uint64_t &size);
ur_result_t GetDeviceErrorCorrectionSupport(ur_device_handle_t device,
                                            bool &ecc_support);
ur_result_t GetDeviceProfilingTimerResolution(ur_device_handle_t device,
                                              size_t &resolution);
ur_result_t GetDeviceLittleEndian(ur_device_handle_t device,
                                  bool &little_endian);
ur_result_t GetDeviceAvailable(ur_device_handle_t device, bool &available);
ur_result_t GetDeviceCompilerAvailable(ur_device_handle_t device,
                                       bool &available);
ur_result_t GetDeviceLinkerAvailable(ur_device_handle_t device,
                                     bool &available);
ur_result_t
GetDeviceExecutionCapabilities(ur_device_handle_t device,
                               ur_device_exec_capability_flags_t &capabilities);
ur_result_t GetDeviceQueueOnDeviceProperties(ur_device_handle_t device,
                                             ur_queue_flags_t &properties);
ur_result_t GetDeviceQueueOnHostProperties(ur_device_handle_t device,
                                           ur_queue_flags_t &properties);
ur_result_t GetDeviceBuiltInKernels(ur_device_handle_t device,
                                    std::vector<std::string> &names);
ur_result_t GetDevicePlatform(ur_device_handle_t device,
                              ur_platform_handle_t &platform);
ur_result_t GetDeviceReferenceCount(ur_device_handle_t device,
                                    uint32_t &ref_count);
ur_result_t GetDeviceILVersion(ur_device_handle_t device,
                               std::string &il_version);
ur_result_t GetDeviceVendor(ur_device_handle_t device, std::string &vendor);
ur_result_t GetDeviceDriverVersion(ur_device_handle_t device,
                                   std::string &driver_version);
ur_result_t GetDeviceProfile(ur_device_handle_t device, std::string &profile);
ur_result_t GetDeviceVersion(ur_device_handle_t device, std::string &version);
ur_result_t GetDeviceBackendRuntimeVersion(ur_device_handle_t device,
                                           std::string &runtime_version);
ur_result_t GetDeviceExtensions(ur_device_handle_t device,
                                std::vector<std::string> &extensions);
ur_result_t GetDevicePrintfBufferSize(ur_device_handle_t device, size_t &size);
ur_result_t GetDevicePreferredInteropUserSync(ur_device_handle_t device,
                                              bool &sync);
ur_result_t GetDeviceParentDevice(ur_device_handle_t device,
                                  ur_device_handle_t &parent);
ur_result_t
GetDevicePartitionProperties(ur_device_handle_t device,
                             std::vector<ur_device_partition_t> &properties);
ur_result_t GetDevicePartitionMaxSubDevices(ur_device_handle_t device,
                                            uint32_t &max_sub_devices);
ur_result_t
GetDevicePartitionAffinityDomainFlags(ur_device_handle_t device,
                                      ur_device_affinity_domain_flags_t &flags);
ur_result_t
GetDevicePartitionType(ur_device_handle_t device,
                       std::vector<ur_device_partition_property_t> &type);
ur_result_t GetDeviceMaxNumberSubGroups(ur_device_handle_t device,
                                        uint32_t &max_sub_groups);
ur_result_t
GetDeviceSubGroupIndependentForwardProgress(ur_device_handle_t device,
                                            bool &progress);
ur_result_t GetDeviceSubGroupSizesIntel(ur_device_handle_t device,
                                        std::vector<uint32_t> &sizes);
ur_result_t
GetDeviceUSMHostSupport(ur_device_handle_t device,
                        ur_device_usm_access_capability_flags_t &support);
ur_result_t
GetDeviceUSMDeviceSupport(ur_device_handle_t device,
                          ur_device_usm_access_capability_flags_t &support);
ur_result_t GetDeviceUSMSingleSharedSupport(
    ur_device_handle_t device,
    ur_device_usm_access_capability_flags_t &support);
ur_result_t GetDeviceUSMCrossSharedSupport(
    ur_device_handle_t device,
    ur_device_usm_access_capability_flags_t &support);
ur_result_t GetDeviceUSMSystemSharedSupport(
    ur_device_handle_t device,
    ur_device_usm_access_capability_flags_t &support);
ur_result_t GetDeviceUUID(ur_device_handle_t device, std::string &uuid);
ur_result_t GetDevicePCIAddress(ur_device_handle_t device,
                                std::string &address);
ur_result_t GetDeviceGPUEUCount(ur_device_handle_t device, uint32_t &count);
ur_result_t GetDeviceGPUEUSIMDWidth(ur_device_handle_t device, uint32_t &width);
ur_result_t GetDeviceGPUEUSlices(ur_device_handle_t device, uint32_t &slices);
ur_result_t GetDeviceGPUSubslicesPerSlice(ur_device_handle_t device,
                                          uint32_t &subslices);
ur_result_t GetDeviceMaxMemoryBandwidth(ur_device_handle_t device,
                                        uint32_t &bandwidth);
ur_result_t GetDeviceImageSRGB(ur_device_handle_t device, bool &support);
ur_result_t GetDeviceAtomic64Support(ur_device_handle_t device, bool &support);
ur_result_t
GetDeviceMemoryOrderCapabilities(ur_device_handle_t device,
                                 ur_memory_order_capability_flags_t &flags);
ur_result_t
GetDeviceMemoryScopeCapabilities(ur_device_handle_t device,
                                 ur_memory_scope_capability_flags_t &flags);
ur_result_t GetDeviceBFloat16Support(ur_device_handle_t device, bool &support);
ur_result_t GetDeviceMaxComputeQueueIndices(ur_device_handle_t device,
                                            uint32_t &max_indices);
ur_result_t GetDeviceHostPipeRWSupported(ur_device_handle_t device,
                                         bool &support);

ur_device_partition_property_t makePartitionByCountsDesc(uint32_t count);

ur_device_partition_property_t makePartitionEquallyDesc(uint32_t cu_per_device);

ur_device_partition_property_t
makePartitionByAffinityDomain(ur_device_affinity_domain_flags_t aff_domain);

enum class USMKind {
    Device,
    Host,
    Shared,
};

ur_result_t MakeUSMAllocationByType(USMKind kind, ur_context_handle_t hContext,
                                    ur_device_handle_t hDevice,
                                    const ur_usm_desc_t *pUSMDesc,
                                    ur_usm_pool_handle_t hPool, size_t size,
                                    void **ppMem);

} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED
