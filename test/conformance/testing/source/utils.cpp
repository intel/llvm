// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string_view>
#include <uur/utils.h>

namespace uur {
namespace {
std::vector<std::string> split(const std::string &str, char delim) {
    const auto n_words = std::count(str.begin(), str.end(), delim) + 1;
    std::vector<std::string> items;
    items.reserve(n_words);

    size_t start;
    size_t end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        items.push_back(str.substr(start, end - start));
    }
    return items;
}

template <class T>
ur_result_t GetDeviceVectorInfo(ur_device_handle_t device,
                                ur_device_info_t info, std::vector<T> &out) {
    size_t size = 0;
    ur_result_t result = urDeviceGetInfo(device, info, 0, nullptr, &size);
    if (result != UR_RESULT_SUCCESS || size == 0) {
        return result;
    }
    std::vector<T> data(size / sizeof(T));
    result = urDeviceGetInfo(device, info, size, data.data(), nullptr);
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    out = std::move(data);
    return UR_RESULT_SUCCESS;
}
} // namespace

ur_result_t GetDeviceType(ur_device_handle_t device,
                          ur_device_type_t &device_type) {
    return GetDeviceInfo<ur_device_type_t>(device, UR_DEVICE_INFO_TYPE,
                                           device_type);
}

ur_result_t GetDeviceVendorId(ur_device_handle_t device, uint32_t &vendor_id) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_VENDOR_ID, vendor_id);
}

ur_result_t GetDeviceId(ur_device_handle_t device, uint32_t &device_id) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_DEVICE_ID, device_id);
}

ur_result_t GetDeviceMaxComputeUnits(ur_device_handle_t device,
                                     uint32_t &max_compute_units) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_COMPUTE_UNITS,
                                   max_compute_units);
}

ur_result_t GetDeviceMaxWorkItemDimensions(ur_device_handle_t device,
                                           uint32_t &max_work_item_dimensions) {
    return GetDeviceInfo<uint32_t>(device,
                                   UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
                                   max_work_item_dimensions);
}

ur_result_t
GetDeviceMaxWorkItemSizes(ur_device_handle_t device,
                          std::vector<size_t> &max_work_item_sizes) {
    return GetDeviceVectorInfo<size_t>(
        device, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES, max_work_item_sizes);
}

ur_result_t GetDeviceMaxWorkGroupSize(ur_device_handle_t device,
                                      size_t &max_work_group_size) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                                 max_work_group_size);
}

ur_result_t GetDeviceSingleFPCapabilities(
    ur_device_handle_t device,
    ur_device_fp_capability_flags_t &fp_capabilities) {
    return GetDeviceInfo<ur_device_fp_capability_flags_t>(
        device, UR_DEVICE_INFO_SINGLE_FP_CONFIG, fp_capabilities);
}

ur_result_t
GetDeviceHalfFPCapabilities(ur_device_handle_t device,
                            ur_device_fp_capability_flags_t &fp_capabilities) {
    return GetDeviceInfo<ur_device_fp_capability_flags_t>(
        device, UR_DEVICE_INFO_HALF_FP_CONFIG, fp_capabilities);
}

ur_result_t GetDeviceDoubleFPCapabilities(
    ur_device_handle_t device,
    ur_device_fp_capability_flags_t &fp_capabilities) {
    return GetDeviceInfo<ur_device_fp_capability_flags_t>(
        device, UR_DEVICE_INFO_DOUBLE_FP_CONFIG, fp_capabilities);
}

ur_result_t GetDeviceQueueProperties(ur_device_handle_t device,
                                     ur_queue_flags_t &flags) {
    return GetDeviceInfo<ur_device_fp_capability_flags_t>(
        device, UR_DEVICE_INFO_QUEUE_PROPERTIES, flags);
}

ur_result_t GetDevicePreferredVectorWidthChar(ur_device_handle_t device,
                                              uint32_t &pref_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR, pref_width);
}

ur_result_t GetDevicePreferredVectorWidthInt(ur_device_handle_t device,
                                             uint32_t &pref_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT, pref_width);
}

ur_result_t GetDevicePreferredVectorWidthLong(ur_device_handle_t device,
                                              uint32_t &pref_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG, pref_width);
}

ur_result_t GetDevicePreferredVectorWidthFloat(ur_device_handle_t device,
                                               uint32_t &pref_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT, pref_width);
}

ur_result_t GetDevicePreferredVectorWidthDouble(ur_device_handle_t device,
                                                uint32_t &pref_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE, pref_width);
}

ur_result_t GetDevicePreferredVectorWidthHalf(ur_device_handle_t device,
                                              uint32_t &pref_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF, pref_width);
}

ur_result_t GetDeviceNativeVectorWithChar(ur_device_handle_t device,
                                          uint32_t &vec_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR, vec_width);
}

ur_result_t GetDeviceNativeVectorWithShort(ur_device_handle_t device,
                                           uint32_t &vec_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT, vec_width);
}

ur_result_t GetDeviceNativeVectorWithInt(ur_device_handle_t device,
                                         uint32_t &vec_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT, vec_width);
}

ur_result_t GetDeviceNativeVectorWithLong(ur_device_handle_t device,
                                          uint32_t &vec_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG, vec_width);
}

ur_result_t GetDeviceNativeVectorWithFloat(ur_device_handle_t device,
                                           uint32_t &vec_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT, vec_width);
}

ur_result_t GetDeviceNativeVectorWithDouble(ur_device_handle_t device,
                                            uint32_t &vec_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE, vec_width);
}

ur_result_t GetDeviceNativeVectorWithHalf(ur_device_handle_t device,
                                          uint32_t &vec_width) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF, vec_width);
}

ur_result_t GetDeviceMaxClockFrequency(ur_device_handle_t device,
                                       uint32_t &max_freq) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY,
                                   max_freq);
}

ur_result_t GetDeviceMemoryClockRate(ur_device_handle_t device,
                                     uint32_t &mem_clock) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MEMORY_CLOCK_RATE,
                                   mem_clock);
}

ur_result_t GetDeviceAddressBits(ur_device_handle_t device,
                                 uint32_t &addr_bits) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_ADDRESS_BITS,
                                   addr_bits);
}

ur_result_t GetDeviceMaxMemAllocSize(ur_device_handle_t device,
                                     uint64_t &alloc_size) {
    return GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                                   alloc_size);
}

ur_result_t GetDeviceImageSupport(ur_device_handle_t device,
                                  bool &image_support) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_IMAGE_SUPPORTED,
                               image_support);
}

ur_result_t GetDeviceMaxReadImageArgs(ur_device_handle_t device,
                                      uint32_t &read_arg) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS,
                                   read_arg);
}

ur_result_t GetDeviceMaxWriteImageArgs(ur_device_handle_t device,
                                       uint32_t &write_args) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS,
                                   write_args);
}

ur_result_t GetDeviceMaxReadWriteImageArgs(ur_device_handle_t device,
                                           uint32_t &read_write_args) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS, read_write_args);
}

ur_result_t GetDeviceImage2DMaxWidth(ur_device_handle_t device,
                                     size_t &max_width) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH,
                                 max_width);
}

ur_result_t GetDeviceImage2DMaxHeight(ur_device_handle_t device,
                                      size_t &max_height) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT,
                                 max_height);
}

ur_result_t GetDeviceImage3DMaxWidth(ur_device_handle_t device,
                                     size_t &max_width) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH,
                                 max_width);
}

ur_result_t GetDeviceImage3DMaxHeight(ur_device_handle_t device,
                                      size_t &max_height) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT,
                                 max_height);
}

ur_result_t GetDeviceImage3DMaxDepth(ur_device_handle_t device,
                                     size_t &max_depth) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH,
                                 max_depth);
}

ur_result_t GetDeviceImageMaxBufferSize(ur_device_handle_t device,
                                        size_t &max_buf_size) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE,
                                 max_buf_size);
}

ur_result_t GetDeviceImageMaxArraySize(ur_device_handle_t device,
                                       size_t &max_arr_size) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE,
                                 max_arr_size);
}

ur_result_t GetDeviceMaxSamplers(ur_device_handle_t device,
                                 uint32_t &max_samplers) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_SAMPLERS,
                                   max_samplers);
}

ur_result_t GetDeviceMaxParameterSize(ur_device_handle_t device,
                                      size_t &max_param_size) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_MAX_PARAMETER_SIZE,
                                 max_param_size);
}

ur_result_t GetDeviceMemBaseAddressAlign(ur_device_handle_t device,
                                         uint32_t &align) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN,
                                   align);
}

ur_result_t GetDeviceMemCacheType(ur_device_handle_t device,
                                  ur_device_mem_cache_type_t &cache_type) {
    return GetDeviceInfo<ur_device_mem_cache_type_t>(
        device, UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE, cache_type);
}

ur_result_t GetDeviceMemCachelineSize(ur_device_handle_t device,
                                      uint32_t &cache_line_size) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE, cache_line_size);
}

ur_result_t GetDeviceMemCacheSize(ur_device_handle_t device,
                                  uint64_t &cache_size) {
    return GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE,
                                   cache_size);
}

ur_result_t GetDeviceGlobalMemSize(ur_device_handle_t device,
                                   uint64_t &mem_size) {
    return GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_GLOBAL_MEM_SIZE,
                                   mem_size);
}

ur_result_t GetDeviceGlobalMemFree(ur_device_handle_t device,
                                   uint64_t &mem_free) {
    return GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                   mem_free);
}

ur_result_t GetDeviceMaxConstantBufferSize(ur_device_handle_t device,
                                           uint64_t &buf_size) {
    return GetDeviceInfo<uint64_t>(
        device, UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE, buf_size);
}

ur_result_t GetDeviceMaxConstantArgs(ur_device_handle_t device,
                                     uint32_t &args) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_CONSTANT_ARGS,
                                   args);
}

ur_result_t GetDeviceLocalMemType(ur_device_handle_t device,
                                  ur_device_local_mem_type_t &type) {
    return GetDeviceInfo<ur_device_local_mem_type_t>(
        device, UR_DEVICE_INFO_LOCAL_MEM_TYPE, type);
}

ur_result_t GetDeviceLocalMemSize(ur_device_handle_t device, uint64_t &size) {
    return GetDeviceInfo<uint64_t>(device, UR_DEVICE_INFO_LOCAL_MEM_SIZE, size);
}

ur_result_t GetDeviceErrorCorrectionSupport(ur_device_handle_t device,
                                            bool &ecc_support) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT,
                               ecc_support);
}

ur_result_t GetDeviceProfilingTimerResolution(ur_device_handle_t device,
                                              size_t &resolution) {
    return GetDeviceInfo<size_t>(
        device, UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION, resolution);
}

ur_result_t GetDeviceLittleEndian(ur_device_handle_t device,
                                  bool &little_endian) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_ENDIAN_LITTLE,
                               little_endian);
}

ur_result_t GetDeviceAvailable(ur_device_handle_t device, bool &available) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_AVAILABLE, available);
}

ur_result_t GetDeviceCompilerAvailable(ur_device_handle_t device,
                                       bool &available) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_COMPILER_AVAILABLE,
                               available);
}

ur_result_t GetDeviceLinkerAvailable(ur_device_handle_t device,
                                     bool &available) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_LINKER_AVAILABLE,
                               available);
}

ur_result_t GetDeviceExecutionCapabilities(
    ur_device_handle_t device,
    ur_device_exec_capability_flags_t &capabilities) {
    return GetDeviceInfo<ur_device_exec_capability_flags_t>(
        device, UR_DEVICE_INFO_EXECUTION_CAPABILITIES, capabilities);
}

ur_result_t GetDeviceQueueOnDeviceProperties(ur_device_handle_t device,
                                             ur_queue_flags_t &properties) {
    return GetDeviceInfo<ur_queue_flags_t>(
        device, UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES, properties);
}

ur_result_t GetDeviceQueueOnHostProperties(ur_device_handle_t device,
                                           ur_queue_flags_t &properties) {
    return GetDeviceInfo<ur_queue_flags_t>(
        device, UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES, properties);
}

ur_result_t GetDeviceBuiltInKernels(ur_device_handle_t device,
                                    std::vector<std::string> &names) {
    std::string kernels_str;
    ur_result_t result = GetDeviceInfo<std::string>(
        device, UR_DEVICE_INFO_BUILT_IN_KERNELS, kernels_str);
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    names = split(kernels_str, ';');
    return UR_RESULT_SUCCESS;
}

ur_result_t GetDevicePlatform(ur_device_handle_t device,
                              ur_platform_handle_t &platform) {
    return GetDeviceInfo<ur_platform_handle_t>(device, UR_DEVICE_INFO_PLATFORM,
                                               platform);
}

ur_result_t GetDeviceReferenceCount(ur_device_handle_t device,
                                    uint32_t &ref_count) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_REFERENCE_COUNT,
                                   ref_count);
}

ur_result_t GetDeviceILVersion(ur_device_handle_t device,
                               std::string &il_version) {
    return GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_IL_VERSION,
                                      il_version);
}

ur_result_t GetDeviceVendor(ur_device_handle_t device, std::string &vendor) {
    return GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_VENDOR, vendor);
}

ur_result_t GetDeviceDriverVersion(ur_device_handle_t device,
                                   std::string &driver_version) {
    return GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_DRIVER_VERSION,
                                      driver_version);
}

ur_result_t GetDeviceProfile(ur_device_handle_t device, std::string &profile) {
    return GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_PROFILE, profile);
}

ur_result_t GetDeviceVersion(ur_device_handle_t device, std::string &version) {
    return GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_VERSION, version);
}

ur_result_t GetDeviceBackendRuntimeVersion(ur_device_handle_t device,
                                           std::string &runtime_version) {
    return GetDeviceInfo<std::string>(
        device, UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION, runtime_version);
}

ur_result_t GetDeviceExtensions(ur_device_handle_t device,
                                std::vector<std::string> &extensions) {
    std::string extensions_str;
    ur_result_t result = GetDeviceInfo<std::string>(
        device, UR_DEVICE_INFO_EXTENSIONS, extensions_str);
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    extensions = split(extensions_str, ' ');
    return UR_RESULT_SUCCESS;
}

ur_result_t GetDevicePrintfBufferSize(ur_device_handle_t device, size_t &size) {
    return GetDeviceInfo<size_t>(device, UR_DEVICE_INFO_PRINTF_BUFFER_SIZE,
                                 size);
}

ur_result_t GetDevicePreferredInteropUserSync(ur_device_handle_t device,
                                              bool &sync) {
    return GetDeviceInfo<bool>(
        device, UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC, sync);
}

ur_result_t GetDeviceParentDevice(ur_device_handle_t device,
                                  ur_device_handle_t &parent) {
    return GetDeviceInfo<ur_device_handle_t>(
        device, UR_DEVICE_INFO_PARENT_DEVICE, parent);
}

ur_result_t
GetDevicePartitionProperties(ur_device_handle_t device,
                             std::vector<ur_device_partition_t> &properties) {
    return GetDeviceVectorInfo<ur_device_partition_t>(
        device, UR_DEVICE_INFO_SUPPORTED_PARTITIONS, properties);
}

ur_result_t GetDevicePartitionMaxSubDevices(ur_device_handle_t device,
                                            uint32_t &max_sub_devices) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES, max_sub_devices);
}

ur_result_t GetDevicePartitionAffinityDomainFlags(
    ur_device_handle_t device, ur_device_affinity_domain_flags_t &flags) {
    return GetDeviceInfo<ur_device_affinity_domain_flags_t>(
        device, UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN, flags);
}

ur_result_t
GetDevicePartitionType(ur_device_handle_t device,
                       std::vector<ur_device_partition_property_t> &type) {
    return GetDeviceVectorInfo<ur_device_partition_property_t>(
        device, UR_DEVICE_INFO_PARTITION_TYPE, type);
}

ur_result_t GetDeviceMaxNumberSubGroups(ur_device_handle_t device,
                                        uint32_t &max_sub_groups) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS,
                                   max_sub_groups);
}

ur_result_t
GetDeviceSubGroupIndependentForwardProgress(ur_device_handle_t device,
                                            bool &progress) {
    return GetDeviceInfo<bool>(
        device, UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
        progress);
}

ur_result_t GetDeviceSubGroupSizesIntel(ur_device_handle_t device,
                                        std::vector<uint32_t> &sizes) {
    return GetDeviceVectorInfo<uint32_t>(
        device, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL, sizes);
}

ur_result_t
GetDeviceUSMHostSupport(ur_device_handle_t device,
                        ur_device_usm_access_capability_flags_t &support) {
    return GetDeviceInfo<ur_device_usm_access_capability_flags_t>(
        device, UR_DEVICE_INFO_USM_HOST_SUPPORT, support);
}

ur_result_t
GetDeviceUSMDeviceSupport(ur_device_handle_t device,
                          ur_device_usm_access_capability_flags_t &support) {
    return GetDeviceInfo<ur_device_usm_access_capability_flags_t>(
        device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT, support);
}

ur_result_t GetDeviceUSMSingleSharedSupport(
    ur_device_handle_t device,
    ur_device_usm_access_capability_flags_t &support) {
    return GetDeviceInfo<ur_device_usm_access_capability_flags_t>(
        device, UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT, support);
}

ur_result_t GetDeviceUSMCrossSharedSupport(
    ur_device_handle_t device,
    ur_device_usm_access_capability_flags_t &support) {
    return GetDeviceInfo<ur_device_usm_access_capability_flags_t>(
        device, UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT, support);
}

ur_result_t GetDeviceUSMSystemSharedSupport(
    ur_device_handle_t device,
    ur_device_usm_access_capability_flags_t &support) {
    return GetDeviceInfo<ur_device_usm_access_capability_flags_t>(
        device, UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT, support);
}

ur_result_t GetDeviceUUID(ur_device_handle_t device, std::string &uuid) {
    return GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_UUID, uuid);
}
ur_result_t GetDevicePCIAddress(ur_device_handle_t device,
                                std::string &address) {
    return GetDeviceInfo<std::string>(device, UR_DEVICE_INFO_PCI_ADDRESS,
                                      address);
}

ur_result_t GetDeviceGPUEUCount(ur_device_handle_t device, uint32_t &count) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_GPU_EU_COUNT, count);
}
ur_result_t GetDeviceGPUEUSIMDWidth(ur_device_handle_t device,
                                    uint32_t &width) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH,
                                   width);
}

ur_result_t GetDeviceGPUEUSlices(ur_device_handle_t device, uint32_t &slices) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_GPU_EU_SLICES,
                                   slices);
}

ur_result_t GetDeviceGPUSubslicesPerSlice(ur_device_handle_t device,
                                          uint32_t &subslices) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE, subslices);
}

ur_result_t GetDeviceMaxMemoryBandwidth(ur_device_handle_t device,
                                        uint32_t &bandwidth) {
    return GetDeviceInfo<uint32_t>(device, UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH,
                                   bandwidth);
}

ur_result_t GetDeviceImageSRGB(ur_device_handle_t device, bool &support) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_IMAGE_SRGB, support);
}

ur_result_t GetDeviceAtomic64Support(ur_device_handle_t device, bool &support) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_ATOMIC_64, support);
}

ur_result_t
GetDeviceMemoryOrderCapabilities(ur_device_handle_t device,
                                 ur_memory_order_capability_flags_t &flags) {
    return GetDeviceInfo<ur_memory_order_capability_flags_t>(
        device, UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES, flags);
}

ur_result_t
GetDeviceMemoryScopeCapabilities(ur_device_handle_t device,
                                 ur_memory_scope_capability_flags_t &flags) {
    return GetDeviceInfo<ur_memory_scope_capability_flags_t>(
        device, UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES, flags);
}

ur_result_t GetDeviceBFloat16Support(ur_device_handle_t device, bool &support) {
    return GetDeviceInfo<bool>(device, UR_DEVICE_INFO_BFLOAT16, support);
}

ur_result_t GetDeviceMaxComputeQueueIndices(ur_device_handle_t device,
                                            uint32_t &max_indices) {
    return GetDeviceInfo<uint32_t>(
        device, UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES, max_indices);
}

ur_result_t GetDeviceHostPipeRWSupported(ur_device_handle_t device,
                                         bool &support) {
    return GetDeviceInfo<bool>(
        device, UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED, support);
}

ur_device_partition_property_t makePartitionByCountsDesc(uint32_t count) {
    ur_device_partition_property_t desc;
    desc.type = UR_DEVICE_PARTITION_BY_COUNTS;
    desc.value.count = count;
    return desc;
}

ur_device_partition_property_t
makePartitionEquallyDesc(uint32_t cu_per_device) {
    ur_device_partition_property_t desc;
    desc.type = UR_DEVICE_PARTITION_EQUALLY;
    desc.value.equally = cu_per_device;
    return desc;
}

ur_device_partition_property_t
makePartitionByAffinityDomain(ur_device_affinity_domain_flags_t aff_domain) {
    ur_device_partition_property_t desc;
    desc.type = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
    desc.value.affinity_domain = aff_domain;
    return desc;
}

ur_result_t MakeUSMAllocationByType(USMKind kind, ur_context_handle_t hContext,
                                    ur_device_handle_t hDevice,
                                    const ur_usm_desc_t *pUSMDesc,
                                    ur_usm_pool_handle_t hPool, size_t size,
                                    void **ppMem) {
    switch (kind) {
    case USMKind::Device:
        return urUSMDeviceAlloc(hContext, hDevice, pUSMDesc, hPool, size,
                                ppMem);
    case USMKind::Host:
        return urUSMHostAlloc(hContext, pUSMDesc, hPool, size, ppMem);
    default:
    case USMKind::Shared:
        return urUSMSharedAlloc(hContext, hDevice, pUSMDesc, hPool, size,
                                ppMem);
    }
}

} // namespace uur
