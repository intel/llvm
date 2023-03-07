// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED

#include <optional>
#include <string>
#include <uur/assert.h>
#include <uur/environment.h>
#include <vector>

namespace uur {

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

inline std::string GetPlatformName(ur_platform_handle_t hPlatform) {
    size_t name_len = 0;
    urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_NAME, 0, nullptr, &name_len);
    std::string platform_name(name_len, '\0');
    urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_NAME, name_len,
                      &platform_name[0], nullptr);
    platform_name.resize(platform_name.find_first_of('\0'));
    return GTestSanitizeString(
        std::string(platform_name.data(), platform_name.size()));
}

inline std::string GetDeviceName(ur_device_handle_t device) {
    size_t name_len = 0;
    urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, 0, nullptr, &name_len);
    std::string name(name_len, '\0');
    urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, name_len, &name[0], nullptr);
    name.resize(name.find_first_of('\0'));
    return GTestSanitizeString(std::string(name.data(), name.size()));
}

inline std::string GetPlatformAndDeviceName(ur_device_handle_t device) {
    return GetPlatformName(GetPlatform()) + "__" + GetDeviceName(device);
}

template <class T, class ObjectTy, class InfoTy, class Callable>
std::optional<T> GetInfo(ObjectTy object, InfoTy info, Callable cb) {
    size_t infoSize = 0;
    if (cb(object, info, 0, nullptr, &infoSize)) {
        return std::nullopt;
    }
    if (infoSize == 0 || infoSize != sizeof(T)) {
        return std::nullopt;
    }
    T queryValue{};
    if (cb(object, info, sizeof(T), &queryValue, nullptr)) {
        return std::nullopt;
    }
    return queryValue;
}

template <class T>
auto GetContextInfo = [](ur_context_handle_t context, ur_context_info_t info) {
    return GetInfo<T>(context, info, urContextGetInfo);
};

template <class T>
auto GetDeviceInfo =
    [](ur_device_handle_t device, ur_device_info_t info) -> std::optional<T> {
    if constexpr (std::is_same_v<T, std::string>) {
        size_t size = 0;
        ur_result_t result =
            urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, 0, nullptr, &size);
        if (result != UR_RESULT_SUCCESS || size == 0) {
            return std::nullopt;
        }
        std::vector<char> data(size);
        result = urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, size, data.data(),
                                 nullptr);
        if (result != UR_RESULT_SUCCESS) {
            return std::nullopt;
        }
        return std::string(data.data(), data.size());
    } else {
        return GetInfo<T>(device, info, urDeviceGetInfo);
    }
};

template <class T>
auto GetEventInfo = [](ur_event_handle_t event, ur_event_info_t info) {
    return GetInfo<T>(event, info, urEventGetInfo);
};

template <class T>
auto GetQueueInfo = [](ur_queue_handle_t queue, ur_queue_info_t info) {
    return GetInfo<T>(queue, info, urQueueGetInfo);
};

template <class T>
auto GetSamplerInfo = [](ur_sampler_handle_t sampler, ur_sampler_info_t info) {
    return GetInfo<T>(sampler, info, urSamplerGetInfo);
};

template <class T>
auto GetKernelInfo = [](ur_kernel_handle_t kernel, ur_kernel_info_t info) {
    return GetInfo<T>(kernel, info, urKernelGetInfo);
};

template <class T>
auto GetProgramInfo = [](ur_program_handle_t program, ur_program_info_t info) {
    return GetInfo<T>(program, info, urProgramGetInfo);
};

template <class T>
std::optional<uint32_t> GetObjectReferenceCount(T object) {
    if constexpr (std::is_same_v<T, ur_context_handle_t>) {
        return GetContextInfo<uint32_t>(object,
                                        UR_CONTEXT_INFO_REFERENCE_COUNT);
    }
    if constexpr (std::is_same_v<T, ur_device_handle_t>) {
        return GetDeviceInfo<uint32_t>(object, UR_DEVICE_INFO_REFERENCE_COUNT);
    }
    if constexpr (std::is_same_v<T, ur_event_handle_t>) {
        return GetEventInfo<uint32_t>(object, UR_EVENT_INFO_REFERENCE_COUNT);
    }
    if constexpr (std::is_same_v<T, ur_queue_handle_t>) {
        return GetQueueInfo<uint32_t>(object, UR_QUEUE_INFO_REFERENCE_COUNT);
    }
    if constexpr (std::is_same_v<T, ur_sampler_handle_t>) {
        return GetSamplerInfo<uint32_t>(object,
                                        UR_SAMPLER_INFO_REFERENCE_COUNT);
    }
    if constexpr (std::is_same_v<T, ur_kernel_handle_t>) {
        return GetKernelInfo<uint32_t>(object, UR_KERNEL_INFO_REFERENCE_COUNT);
    }
    if constexpr (std::is_same_v<T, ur_program_handle_t>) {
        return GetProgramInfo<uint32_t>(object,
                                        UR_PROGRAM_INFO_REFERENCE_COUNT);
    }
    return std::nullopt;
}

ur_device_type_t GetDeviceType(ur_device_handle_t device);
uint32_t GetDeviceVendorId(ur_device_handle_t device);
uint32_t GetDeviceId(ur_device_handle_t device);
uint32_t GetDeviceMaxComputeUnits(ur_device_handle_t device);
uint32_t GetDeviceMaxWorkItemDimensions(ur_device_handle_t device);
std::vector<size_t> GetDeviceMaxWorkItemSizes(ur_device_handle_t device);
size_t GetDeviceMaxWorkGroupSize(ur_device_handle_t device);
ur_fp_capability_flags_t
GetDeviceSingleFPCapabilities(ur_device_handle_t device);
ur_fp_capability_flags_t GetDeviceHalfFPCapabilities(ur_device_handle_t device);
ur_fp_capability_flags_t
GetDeviceDoubleFPCapabilities(ur_device_handle_t device);
ur_queue_flags_t GetDeviceQueueProperties(ur_device_handle_t device);
uint32_t GetDevicePreferredVectorWidthChar(ur_device_handle_t device);
uint32_t GetDevicePreferredVectorWidthShort(ur_device_handle_t device);
uint32_t GetDevicePreferredVectorWidthInt(ur_device_handle_t device);
uint32_t GetDevicePreferredVectorWidthLong(ur_device_handle_t device);
uint32_t GetDevicePreferredVectorWidthFloat(ur_device_handle_t device);
uint32_t GetDevicePreferredVectorWidthDouble(ur_device_handle_t device);
uint32_t GetDevicePreferredVectorWidthHalf(ur_device_handle_t device);
uint32_t GetDeviceNativeVectorWithChar(ur_device_handle_t device);
uint32_t GetDeviceNativeVectorWithShort(ur_device_handle_t device);
uint32_t GetDeviceNativeVectorWithInt(ur_device_handle_t device);
uint32_t GetDeviceNativeVectorWithLong(ur_device_handle_t device);
uint32_t GetDeviceNativeVectorWithFloat(ur_device_handle_t device);
uint32_t GetDeviceNativeVectorWithDouble(ur_device_handle_t device);
uint32_t GetDeviceNativeVectorWithHalf(ur_device_handle_t device);
uint32_t GetDeviceMaxClockFrequency(ur_device_handle_t device);
uint32_t GetDeviceMemoryClockRate(ur_device_handle_t device);
uint32_t GetDeviceAddressBits(ur_device_handle_t device);
uint64_t GetDeviceMaxMemAllocSize(ur_device_handle_t device);
bool GetDeviceImageSupport(ur_device_handle_t device);
uint32_t GetDeviceMaxReadImageArgs(ur_device_handle_t device);
uint32_t GetDeviceMaxWriteImageArgs(ur_device_handle_t device);
uint32_t GetDeviceMaxReadWriteImageArgs(ur_device_handle_t device);
size_t GetDeviceImage2DMaxWidth(ur_device_handle_t device);
size_t GetDeviceImage2DMaxHeight(ur_device_handle_t device);
size_t GetDeviceImage3DMaxWidth(ur_device_handle_t device);
size_t GetDeviceImage3DMaxHeight(ur_device_handle_t device);
size_t GetDeviceImage3DMaxDepth(ur_device_handle_t device);
size_t GetDeviceImageMaxBufferSize(ur_device_handle_t device);
size_t GetDeviceImageMaxArraySize(ur_device_handle_t device);
uint32_t GetDeviceMaxSamplers(ur_device_handle_t device);
size_t GetDeviceMaxParameterSize(ur_device_handle_t device);
uint32_t GetDeviceMemBaseAddressAlign(ur_device_handle_t device);
ur_device_mem_cache_type_t GetDeviceMemCacheType(ur_device_handle_t device);
uint32_t GetDeviceMemCachelineSize(ur_device_handle_t device);
uint64_t GetDeviceMemCacheSize(ur_device_handle_t device);
uint64_t GetDeviceGlobalMemSize(ur_device_handle_t device);
uint64_t GetDeviceGlobalMemFree(ur_device_handle_t device);
uint64_t GetDeviceMaxConstantBufferSize(ur_device_handle_t device);
uint32_t GetDeviceMaxConstantArgs(ur_device_handle_t device);
ur_device_local_mem_type_t GetDeviceLocalMemType(ur_device_handle_t device);
uint64_t GetDeviceLocalMemSize(ur_device_handle_t device);
bool GetDeviceErrorCorrectionSupport(ur_device_handle_t device);
size_t GetDeviceProfilingTimerResolution(ur_device_handle_t device);
bool GetDeviceLittleEndian(ur_device_handle_t device);
bool GetDeviceAvailable(ur_device_handle_t device);
bool GetDeviceCompilerAvailable(ur_device_handle_t device);
bool GetDeviceLinkerAvailable(ur_device_handle_t device);
ur_device_exec_capability_flags_t
GetDeviceExecutionCapabilities(ur_device_handle_t device);
ur_queue_flags_t GetDeviceQueueOnDeviceProperties(ur_device_handle_t device);
ur_queue_flags_t GetDeviceQueueOnHostProperties(ur_device_handle_t device);
std::vector<std::string> GetDeviceBuiltInKernels(ur_device_handle_t device);
ur_platform_handle_t GetDevicePlatform(ur_device_handle_t device);
uint32_t GetDeviceReferenceCount(ur_device_handle_t device);
std::string GetDeviceILVersion(ur_device_handle_t device);
std::string GetDeviceVendor(ur_device_handle_t device);
std::string GetDeviceDriverVersion(ur_device_handle_t device);
std::string GetDeviceProfile(ur_device_handle_t device);
std::string GetDeviceVersion(ur_device_handle_t device);
std::string GetDeviceBackendRuntimeVersion(ur_device_handle_t device);
std::vector<std::string> GetDeviceExtensions(ur_device_handle_t device);
size_t GetDevicePrintfBufferSize(ur_device_handle_t device);
bool GetDevicePreferredInteropUserSync(ur_device_handle_t device);
ur_device_handle_t GetDeviceParentDevice(ur_device_handle_t device);
std::vector<ur_device_partition_property_t>
GetDevicePartitionProperties(ur_device_handle_t device);
uint32_t GetDevicePartitionMaxSubDevices(ur_device_handle_t device);
ur_device_affinity_domain_flags_t
GetDevicePartitionAffinityDomainFlags(ur_device_handle_t device);
std::vector<ur_device_partition_property_t>
GetDevicePartitionType(ur_device_handle_t device);
uint32_t GetDeviceMaxNumberSubGroups(ur_device_handle_t device);
bool GetDeviceSubGroupIndependentForwardProgress(ur_device_handle_t device);
std::vector<uint32_t> GetDeviceSubGroupSizesIntel(ur_device_handle_t device);
bool GetDeviceUSMHostSupport(ur_device_handle_t device);
bool GetDeviceUSMDeviceSupport(ur_device_handle_t device);
bool GetDeviceUSMSingleSharedSupport(ur_device_handle_t device);
bool GetDeviceUSMCrossSharedSupport(ur_device_handle_t device);
bool GetDeviceUSMSystemSharedSupport(ur_device_handle_t device);
std::string GetDeviceUUID(ur_device_handle_t device);
std::string GetDevicePCIAddress(ur_device_handle_t device);
uint32_t GetDeviceGPUEUCount(ur_device_handle_t device);
uint32_t GetDeviceGPUEUSIMDWidth(ur_device_handle_t device);
uint32_t GetDeviceGPUEUSlices(ur_device_handle_t device);
uint32_t GetDeviceGPUSubslicesPerSlice(ur_device_handle_t device);
uint32_t GetDeviceMaxMemoryBandwidth(ur_device_handle_t device);
bool GetDeviceImageSRGB(ur_device_handle_t device);
bool GetDeviceAtomic64Support(ur_device_handle_t device);
ur_memory_order_capability_flags_t
GetDeviceMemoryOrderCapabilities(ur_device_handle_t device);
ur_memory_scope_capability_flags_t
GetDeviceMemoryScopeCapabilities(ur_device_handle_t device);
bool GetDeviceBFloat16Support(ur_device_handle_t device);
uint32_t GetDeviceMaxComputeQueueIndices(ur_device_handle_t device);

} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED
