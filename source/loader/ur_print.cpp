/*
 *
 * Copyright (C) 2023-2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_print.cpp
 *
 */

#include "ur_print.h"
#include "ur_print.hpp"

#include <algorithm>
#include <sstream>
#include <string.h>

ur_result_t str_copy(std::stringstream *ss, char *buff, const size_t buff_size,
                     size_t *out_size) {
    size_t c_str_size = strlen(ss->str().c_str()) + 1;
    if (out_size) {
        *out_size = c_str_size;
    }

    if (buff) {
        if (buff_size < c_str_size) {
            return UR_RESULT_ERROR_INVALID_SIZE;
        }

#if defined(_WIN32)
        strncpy_s(buff, buff_size, ss->str().c_str(), c_str_size);
#else
        strncpy(buff, ss->str().c_str(), std::min(buff_size, c_str_size));
#endif
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t urPrintFunction(enum ur_function_t value, char *buffer,
                            const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintStructureType(enum ur_structure_type_t value, char *buffer,
                                 const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintResult(enum ur_result_t value, char *buffer,
                          const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBaseProperties(const struct ur_base_properties_t params,
                                  char *buffer, const size_t buff_size,
                                  size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBaseDesc(const struct ur_base_desc_t params, char *buffer,
                            const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintRectOffset(const struct ur_rect_offset_t params,
                              char *buffer, const size_t buff_size,
                              size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintRectRegion(const struct ur_rect_region_t params,
                              char *buffer, const size_t buff_size,
                              size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceInitFlags(enum ur_device_init_flag_t value,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintLoaderConfigInfo(enum ur_loader_config_info_t value,
                                    char *buffer, const size_t buff_size,
                                    size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCodeLocation(const struct ur_code_location_t params,
                                char *buffer, const size_t buff_size,
                                size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintAdapterInfo(enum ur_adapter_info_t value, char *buffer,
                               const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintAdapterBackend(enum ur_adapter_backend_t value, char *buffer,
                                  const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPlatformInfo(enum ur_platform_info_t value, char *buffer,
                                const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintApiVersion(enum ur_api_version_t value, char *buffer,
                              const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPlatformNativeProperties(
    const struct ur_platform_native_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPlatformBackend(enum ur_platform_backend_t value,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceBinary(const struct ur_device_binary_t params,
                                char *buffer, const size_t buff_size,
                                size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceType(enum ur_device_type_t value, char *buffer,
                              const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceInfo(enum ur_device_info_t value, char *buffer,
                              const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDeviceAffinityDomainFlags(enum ur_device_affinity_domain_flag_t value,
                                 char *buffer, const size_t buff_size,
                                 size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDevicePartition(enum ur_device_partition_t value,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDevicePartitionProperty(
    const struct ur_device_partition_property_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDevicePartitionProperties(
    const struct ur_device_partition_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDeviceFpCapabilityFlags(enum ur_device_fp_capability_flag_t value,
                               char *buffer, const size_t buff_size,
                               size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceMemCacheType(enum ur_device_mem_cache_type_t value,
                                      char *buffer, const size_t buff_size,
                                      size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceLocalMemType(enum ur_device_local_mem_type_t value,
                                      char *buffer, const size_t buff_size,
                                      size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDeviceExecCapabilityFlags(enum ur_device_exec_capability_flag_t value,
                                 char *buffer, const size_t buff_size,
                                 size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDeviceNativeProperties(const struct ur_device_native_properties_t params,
                              char *buffer, const size_t buff_size,
                              size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintMemoryOrderCapabilityFlags(enum ur_memory_order_capability_flag_t value,
                                  char *buffer, const size_t buff_size,
                                  size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintMemoryScopeCapabilityFlags(enum ur_memory_scope_capability_flag_t value,
                                  char *buffer, const size_t buff_size,
                                  size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceUsmAccessCapabilityFlags(
    enum ur_device_usm_access_capability_flag_t value, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintContextFlags(enum ur_context_flag_t value, char *buffer,
                                const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintContextProperties(const struct ur_context_properties_t params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintContextInfo(enum ur_context_info_t value, char *buffer,
                               const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintContextNativeProperties(
    const struct ur_context_native_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemFlags(enum ur_mem_flag_t value, char *buffer,
                            const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemType(enum ur_mem_type_t value, char *buffer,
                           const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemInfo(enum ur_mem_info_t value, char *buffer,
                           const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintImageChannelOrder(enum ur_image_channel_order_t value,
                                     char *buffer, const size_t buff_size,
                                     size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintImageChannelType(enum ur_image_channel_type_t value,
                                    char *buffer, const size_t buff_size,
                                    size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintImageInfo(enum ur_image_info_t value, char *buffer,
                             const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintImageFormat(const struct ur_image_format_t params,
                               char *buffer, const size_t buff_size,
                               size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintImageDesc(const struct ur_image_desc_t params, char *buffer,
                             const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBufferProperties(const struct ur_buffer_properties_t params,
                                    char *buffer, const size_t buff_size,
                                    size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBufferChannelProperties(
    const struct ur_buffer_channel_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBufferAllocLocationProperties(
    const struct ur_buffer_alloc_location_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBufferRegion(const struct ur_buffer_region_t params,
                                char *buffer, const size_t buff_size,
                                size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBufferCreateType(enum ur_buffer_create_type_t value,
                                    char *buffer, const size_t buff_size,
                                    size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintMemNativeProperties(const struct ur_mem_native_properties_t params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintSamplerFilterMode(enum ur_sampler_filter_mode_t value,
                                     char *buffer, const size_t buff_size,
                                     size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintSamplerAddressingMode(enum ur_sampler_addressing_mode_t value,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintSamplerInfo(enum ur_sampler_info_t value, char *buffer,
                               const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintSamplerDesc(const struct ur_sampler_desc_t params,
                               char *buffer, const size_t buff_size,
                               size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintSamplerNativeProperties(
    const struct ur_sampler_native_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmHostMemFlags(enum ur_usm_host_mem_flag_t value,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmDeviceMemFlags(enum ur_usm_device_mem_flag_t value,
                                     char *buffer, const size_t buff_size,
                                     size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmPoolFlags(enum ur_usm_pool_flag_t value, char *buffer,
                                const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmType(enum ur_usm_type_t value, char *buffer,
                           const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmAllocInfo(enum ur_usm_alloc_info_t value, char *buffer,
                                const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmAdviceFlags(enum ur_usm_advice_flag_t value, char *buffer,
                                  const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmDesc(const struct ur_usm_desc_t params, char *buffer,
                           const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmHostDesc(const struct ur_usm_host_desc_t params,
                               char *buffer, const size_t buff_size,
                               size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmDeviceDesc(const struct ur_usm_device_desc_t params,
                                 char *buffer, const size_t buff_size,
                                 size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmAllocLocationDesc(const struct ur_usm_alloc_location_desc_t params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmPoolDesc(const struct ur_usm_pool_desc_t params,
                               char *buffer, const size_t buff_size,
                               size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmPoolLimitsDesc(const struct ur_usm_pool_limits_desc_t params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmPoolInfo(enum ur_usm_pool_info_t value, char *buffer,
                               const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintVirtualMemGranularityInfo(enum ur_virtual_mem_granularity_info_t value,
                                 char *buffer, const size_t buff_size,
                                 size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintVirtualMemAccessFlags(enum ur_virtual_mem_access_flag_t value,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintVirtualMemInfo(enum ur_virtual_mem_info_t value,
                                  char *buffer, const size_t buff_size,
                                  size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPhysicalMemFlags(enum ur_physical_mem_flag_t value,
                                    char *buffer, const size_t buff_size,
                                    size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintPhysicalMemProperties(const struct ur_physical_mem_properties_t params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramMetadataType(enum ur_program_metadata_type_t value,
                                       char *buffer, const size_t buff_size,
                                       size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramMetadata(const struct ur_program_metadata_t params,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramProperties(const struct ur_program_properties_t params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramInfo(enum ur_program_info_t value, char *buffer,
                               const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramBuildStatus(enum ur_program_build_status_t value,
                                      char *buffer, const size_t buff_size,
                                      size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramBinaryType(enum ur_program_binary_type_t value,
                                     char *buffer, const size_t buff_size,
                                     size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramBuildInfo(enum ur_program_build_info_t value,
                                    char *buffer, const size_t buff_size,
                                    size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintSpecializationConstantInfo(
    const struct ur_specialization_constant_info_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramNativeProperties(
    const struct ur_program_native_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelArgValueProperties(
    const struct ur_kernel_arg_value_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelArgLocalProperties(
    const struct ur_kernel_arg_local_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelInfo(enum ur_kernel_info_t value, char *buffer,
                              const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelGroupInfo(enum ur_kernel_group_info_t value,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSubGroupInfo(enum ur_kernel_sub_group_info_t value,
                                      char *buffer, const size_t buff_size,
                                      size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelCacheConfig(enum ur_kernel_cache_config_t value,
                                     char *buffer, const size_t buff_size,
                                     size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelExecInfo(enum ur_kernel_exec_info_t value,
                                  char *buffer, const size_t buff_size,
                                  size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelArgPointerProperties(
    const struct ur_kernel_arg_pointer_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelExecInfoProperties(
    const struct ur_kernel_exec_info_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelArgSamplerProperties(
    const struct ur_kernel_arg_sampler_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelArgMemObjProperties(
    const struct ur_kernel_arg_mem_obj_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintKernelNativeProperties(const struct ur_kernel_native_properties_t params,
                              char *buffer, const size_t buff_size,
                              size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintQueueInfo(enum ur_queue_info_t value, char *buffer,
                             const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintQueueFlags(enum ur_queue_flag_t value, char *buffer,
                              const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintQueueProperties(const struct ur_queue_properties_t params,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueIndexProperties(const struct ur_queue_index_properties_t params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintQueueNativeDesc(const struct ur_queue_native_desc_t params,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueNativeProperties(const struct ur_queue_native_properties_t params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommand(enum ur_command_t value, char *buffer,
                           const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEventStatus(enum ur_event_status_t value, char *buffer,
                               const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEventInfo(enum ur_event_info_t value, char *buffer,
                             const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProfilingInfo(enum ur_profiling_info_t value, char *buffer,
                                 const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintEventNativeProperties(const struct ur_event_native_properties_t params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExecutionInfo(enum ur_execution_info_t value, char *buffer,
                                 const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMapFlags(enum ur_map_flag_t value, char *buffer,
                            const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmMigrationFlags(enum ur_usm_migration_flag_t value,
                                     char *buffer, const size_t buff_size,
                                     size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpImageCopyFlags(enum ur_exp_image_copy_flag_t value,
                                     char *buffer, const size_t buff_size,
                                     size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintExpFileDescriptor(const struct ur_exp_file_descriptor_t params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpWin32Handle(const struct ur_exp_win32_handle_t params,
                                  char *buffer, const size_t buff_size,
                                  size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpSamplerMipProperties(
    const struct ur_exp_sampler_mip_properties_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintExpSamplerAddrModes(const struct ur_exp_sampler_addr_modes_t params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintExpInteropMemDesc(const struct ur_exp_interop_mem_desc_t params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpInteropSemaphoreDesc(
    const struct ur_exp_interop_semaphore_desc_t params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpCommandBufferInfo(enum ur_exp_command_buffer_info_t value,
                                        char *buffer, const size_t buff_size,
                                        size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpCommandBufferCommandInfo(
    enum ur_exp_command_buffer_command_info_t value, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintExpCommandBufferDesc(const struct ur_exp_command_buffer_desc_t params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpCommandBufferUpdateMemobjArgDesc(
    const struct ur_exp_command_buffer_update_memobj_arg_desc_t params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpCommandBufferUpdatePointerArgDesc(
    const struct ur_exp_command_buffer_update_pointer_arg_desc_t params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpCommandBufferUpdateValueArgDesc(
    const struct ur_exp_command_buffer_update_value_arg_desc_t params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpCommandBufferUpdateExecInfoDesc(
    const struct ur_exp_command_buffer_update_exec_info_desc_t params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpCommandBufferUpdateKernelLaunchDesc(
    const struct ur_exp_command_buffer_update_kernel_launch_desc_t params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintExpPeerInfo(enum ur_exp_peer_info_t value, char *buffer,
                               const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << value;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintAdapterGetParams(const struct ur_adapter_get_params_t *params,
                        char *buffer, const size_t buff_size,
                        size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintAdapterReleaseParams(const struct ur_adapter_release_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintAdapterRetainParams(const struct ur_adapter_retain_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintAdapterGetLastErrorParams(
    const struct ur_adapter_get_last_error_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintAdapterGetInfoParams(const struct ur_adapter_get_info_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesUnsampledImageHandleDestroyExpParams(
    const struct ur_bindless_images_unsampled_image_handle_destroy_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesSampledImageHandleDestroyExpParams(
    const struct ur_bindless_images_sampled_image_handle_destroy_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesImageAllocateExpParams(
    const struct ur_bindless_images_image_allocate_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesImageFreeExpParams(
    const struct ur_bindless_images_image_free_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesUnsampledImageCreateExpParams(
    const struct ur_bindless_images_unsampled_image_create_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesSampledImageCreateExpParams(
    const struct ur_bindless_images_sampled_image_create_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesImageCopyExpParams(
    const struct ur_bindless_images_image_copy_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesImageGetInfoExpParams(
    const struct ur_bindless_images_image_get_info_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesMipmapGetLevelExpParams(
    const struct ur_bindless_images_mipmap_get_level_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesMipmapFreeExpParams(
    const struct ur_bindless_images_mipmap_free_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesImportOpaqueFdExpParams(
    const struct ur_bindless_images_import_opaque_fd_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesMapExternalArrayExpParams(
    const struct ur_bindless_images_map_external_array_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesReleaseInteropExpParams(
    const struct ur_bindless_images_release_interop_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesImportExternalSemaphoreOpaqueFdExpParams(
    const struct
    ur_bindless_images_import_external_semaphore_opaque_fd_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesDestroyExternalSemaphoreExpParams(
    const struct ur_bindless_images_destroy_external_semaphore_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesWaitExternalSemaphoreExpParams(
    const struct ur_bindless_images_wait_external_semaphore_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintBindlessImagesSignalExternalSemaphoreExpParams(
    const struct ur_bindless_images_signal_external_semaphore_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferCreateExpParams(
    const struct ur_command_buffer_create_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferRetainExpParams(
    const struct ur_command_buffer_retain_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferReleaseExpParams(
    const struct ur_command_buffer_release_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferFinalizeExpParams(
    const struct ur_command_buffer_finalize_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendKernelLaunchExpParams(
    const struct ur_command_buffer_append_kernel_launch_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendUsmMemcpyExpParams(
    const struct ur_command_buffer_append_usm_memcpy_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendUsmFillExpParams(
    const struct ur_command_buffer_append_usm_fill_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendMemBufferCopyExpParams(
    const struct ur_command_buffer_append_mem_buffer_copy_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendMemBufferWriteExpParams(
    const struct ur_command_buffer_append_mem_buffer_write_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendMemBufferReadExpParams(
    const struct ur_command_buffer_append_mem_buffer_read_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendMemBufferCopyRectExpParams(
    const struct ur_command_buffer_append_mem_buffer_copy_rect_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendMemBufferWriteRectExpParams(
    const struct ur_command_buffer_append_mem_buffer_write_rect_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendMemBufferReadRectExpParams(
    const struct ur_command_buffer_append_mem_buffer_read_rect_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendMemBufferFillExpParams(
    const struct ur_command_buffer_append_mem_buffer_fill_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendUsmPrefetchExpParams(
    const struct ur_command_buffer_append_usm_prefetch_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferAppendUsmAdviseExpParams(
    const struct ur_command_buffer_append_usm_advise_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferEnqueueExpParams(
    const struct ur_command_buffer_enqueue_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferRetainCommandExpParams(
    const struct ur_command_buffer_retain_command_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferReleaseCommandExpParams(
    const struct ur_command_buffer_release_command_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferUpdateKernelLaunchExpParams(
    const struct ur_command_buffer_update_kernel_launch_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferGetInfoExpParams(
    const struct ur_command_buffer_get_info_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintCommandBufferCommandGetInfoExpParams(
    const struct ur_command_buffer_command_get_info_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintContextCreateParams(const struct ur_context_create_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintContextRetainParams(const struct ur_context_retain_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintContextReleaseParams(const struct ur_context_release_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintContextGetInfoParams(const struct ur_context_get_info_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintContextGetNativeHandleParams(
    const struct ur_context_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintContextCreateWithNativeHandleParams(
    const struct ur_context_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintContextSetExtendedDeleterParams(
    const struct ur_context_set_extended_deleter_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueKernelLaunchParams(
    const struct ur_enqueue_kernel_launch_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueEventsWaitParams(
    const struct ur_enqueue_events_wait_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueEventsWaitWithBarrierParams(
    const struct ur_enqueue_events_wait_with_barrier_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferReadParams(
    const struct ur_enqueue_mem_buffer_read_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferWriteParams(
    const struct ur_enqueue_mem_buffer_write_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferReadRectParams(
    const struct ur_enqueue_mem_buffer_read_rect_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferWriteRectParams(
    const struct ur_enqueue_mem_buffer_write_rect_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferCopyParams(
    const struct ur_enqueue_mem_buffer_copy_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferCopyRectParams(
    const struct ur_enqueue_mem_buffer_copy_rect_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferFillParams(
    const struct ur_enqueue_mem_buffer_fill_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemImageReadParams(
    const struct ur_enqueue_mem_image_read_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemImageWriteParams(
    const struct ur_enqueue_mem_image_write_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemImageCopyParams(
    const struct ur_enqueue_mem_image_copy_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueMemBufferMapParams(
    const struct ur_enqueue_mem_buffer_map_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintEnqueueMemUnmapParams(const struct ur_enqueue_mem_unmap_params_t *params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintEnqueueUsmFillParams(const struct ur_enqueue_usm_fill_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueUsmMemcpyParams(
    const struct ur_enqueue_usm_memcpy_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueUsmPrefetchParams(
    const struct ur_enqueue_usm_prefetch_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueUsmAdviseParams(
    const struct ur_enqueue_usm_advise_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueUsmFill_2dParams(
    const struct ur_enqueue_usm_fill_2d_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueUsmMemcpy_2dParams(
    const struct ur_enqueue_usm_memcpy_2d_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueDeviceGlobalVariableWriteParams(
    const struct ur_enqueue_device_global_variable_write_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueDeviceGlobalVariableReadParams(
    const struct ur_enqueue_device_global_variable_read_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueReadHostPipeParams(
    const struct ur_enqueue_read_host_pipe_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueWriteHostPipeParams(
    const struct ur_enqueue_write_host_pipe_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEnqueueCooperativeKernelLaunchExpParams(
    const struct ur_enqueue_cooperative_kernel_launch_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintEventGetInfoParams(const struct ur_event_get_info_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEventGetProfilingInfoParams(
    const struct ur_event_get_profiling_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEventWaitParams(const struct ur_event_wait_params_t *params,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintEventRetainParams(const struct ur_event_retain_params_t *params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintEventReleaseParams(const struct ur_event_release_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEventGetNativeHandleParams(
    const struct ur_event_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEventCreateWithNativeHandleParams(
    const struct ur_event_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintEventSetCallbackParams(
    const struct ur_event_set_callback_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintKernelCreateParams(const struct ur_kernel_create_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintKernelGetInfoParams(const struct ur_kernel_get_info_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelGetGroupInfoParams(
    const struct ur_kernel_get_group_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelGetSubGroupInfoParams(
    const struct ur_kernel_get_sub_group_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintKernelRetainParams(const struct ur_kernel_retain_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintKernelReleaseParams(const struct ur_kernel_release_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelGetNativeHandleParams(
    const struct ur_kernel_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelCreateWithNativeHandleParams(
    const struct ur_kernel_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSetArgValueParams(
    const struct ur_kernel_set_arg_value_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSetArgLocalParams(
    const struct ur_kernel_set_arg_local_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSetArgPointerParams(
    const struct ur_kernel_set_arg_pointer_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSetExecInfoParams(
    const struct ur_kernel_set_exec_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSetArgSamplerParams(
    const struct ur_kernel_set_arg_sampler_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSetArgMemObjParams(
    const struct ur_kernel_set_arg_mem_obj_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSetSpecializationConstantsParams(
    const struct ur_kernel_set_specialization_constants_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintKernelSuggestMaxCooperativeGroupCountExpParams(
    const struct ur_kernel_suggest_max_cooperative_group_count_exp_params_t
        *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintLoaderInitParams(const struct ur_loader_init_params_t *params,
                        char *buffer, const size_t buff_size,
                        size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintLoaderTearDownParams(const struct ur_loader_tear_down_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintLoaderConfigCreateParams(
    const struct ur_loader_config_create_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintLoaderConfigRetainParams(
    const struct ur_loader_config_retain_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintLoaderConfigReleaseParams(
    const struct ur_loader_config_release_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintLoaderConfigGetInfoParams(
    const struct ur_loader_config_get_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintLoaderConfigEnableLayerParams(
    const struct ur_loader_config_enable_layer_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintLoaderConfigSetCodeLocationCallbackParams(
    const struct ur_loader_config_set_code_location_callback_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintMemImageCreateParams(const struct ur_mem_image_create_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintMemBufferCreateParams(const struct ur_mem_buffer_create_params_t *params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemRetainParams(const struct ur_mem_retain_params_t *params,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintMemReleaseParams(const struct ur_mem_release_params_t *params,
                        char *buffer, const size_t buff_size,
                        size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemBufferPartitionParams(
    const struct ur_mem_buffer_partition_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemGetNativeHandleParams(
    const struct ur_mem_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemBufferCreateWithNativeHandleParams(
    const struct ur_mem_buffer_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemImageCreateWithNativeHandleParams(
    const struct ur_mem_image_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintMemGetInfoParams(const struct ur_mem_get_info_params_t *params,
                        char *buffer, const size_t buff_size,
                        size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintMemImageGetInfoParams(
    const struct ur_mem_image_get_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPhysicalMemCreateParams(
    const struct ur_physical_mem_create_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPhysicalMemRetainParams(
    const struct ur_physical_mem_retain_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPhysicalMemReleaseParams(
    const struct ur_physical_mem_release_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintPlatformGetParams(const struct ur_platform_get_params_t *params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintPlatformGetInfoParams(const struct ur_platform_get_info_params_t *params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPlatformGetNativeHandleParams(
    const struct ur_platform_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPlatformCreateWithNativeHandleParams(
    const struct ur_platform_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPlatformGetApiVersionParams(
    const struct ur_platform_get_api_version_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintPlatformGetBackendOptionParams(
    const struct ur_platform_get_backend_option_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramCreateWithIlParams(
    const struct ur_program_create_with_il_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramCreateWithBinaryParams(
    const struct ur_program_create_with_binary_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramBuildParams(const struct ur_program_build_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramBuildExpParams(const struct ur_program_build_exp_params_t *params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramCompileParams(const struct ur_program_compile_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramCompileExpParams(
    const struct ur_program_compile_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramLinkParams(const struct ur_program_link_params_t *params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramLinkExpParams(const struct ur_program_link_exp_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramRetainParams(const struct ur_program_retain_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramReleaseParams(const struct ur_program_release_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramGetFunctionPointerParams(
    const struct ur_program_get_function_pointer_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintProgramGetInfoParams(const struct ur_program_get_info_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramGetBuildInfoParams(
    const struct ur_program_get_build_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramSetSpecializationConstantsParams(
    const struct ur_program_set_specialization_constants_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramGetNativeHandleParams(
    const struct ur_program_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintProgramCreateWithNativeHandleParams(
    const struct ur_program_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueGetInfoParams(const struct ur_queue_get_info_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueCreateParams(const struct ur_queue_create_params_t *params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueRetainParams(const struct ur_queue_retain_params_t *params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueReleaseParams(const struct ur_queue_release_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintQueueGetNativeHandleParams(
    const struct ur_queue_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintQueueCreateWithNativeHandleParams(
    const struct ur_queue_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueFinishParams(const struct ur_queue_finish_params_t *params,
                         char *buffer, const size_t buff_size,
                         size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintQueueFlushParams(const struct ur_queue_flush_params_t *params,
                        char *buffer, const size_t buff_size,
                        size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintSamplerCreateParams(const struct ur_sampler_create_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintSamplerRetainParams(const struct ur_sampler_retain_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintSamplerReleaseParams(const struct ur_sampler_release_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintSamplerGetInfoParams(const struct ur_sampler_get_info_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintSamplerGetNativeHandleParams(
    const struct ur_sampler_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintSamplerCreateWithNativeHandleParams(
    const struct ur_sampler_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmHostAllocParams(const struct ur_usm_host_alloc_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmDeviceAllocParams(const struct ur_usm_device_alloc_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmSharedAllocParams(const struct ur_usm_shared_alloc_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmFreeParams(const struct ur_usm_free_params_t *params,
                                 char *buffer, const size_t buff_size,
                                 size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmGetMemAllocInfoParams(
    const struct ur_usm_get_mem_alloc_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmPoolCreateParams(const struct ur_usm_pool_create_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmPoolRetainParams(const struct ur_usm_pool_retain_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmPoolReleaseParams(const struct ur_usm_pool_release_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmPoolGetInfoParams(const struct ur_usm_pool_get_info_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmPitchedAllocExpParams(
    const struct ur_usm_pitched_alloc_exp_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmImportExpParams(const struct ur_usm_import_exp_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintUsmReleaseExpParams(const struct ur_usm_release_exp_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmP2pEnablePeerAccessExpParams(
    const struct ur_usm_p2p_enable_peer_access_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmP2pDisablePeerAccessExpParams(
    const struct ur_usm_p2p_disable_peer_access_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintUsmP2pPeerAccessGetInfoExpParams(
    const struct ur_usm_p2p_peer_access_get_info_exp_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintVirtualMemGranularityGetInfoParams(
    const struct ur_virtual_mem_granularity_get_info_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintVirtualMemReserveParams(
    const struct ur_virtual_mem_reserve_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintVirtualMemFreeParams(const struct ur_virtual_mem_free_params_t *params,
                            char *buffer, const size_t buff_size,
                            size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintVirtualMemMapParams(const struct ur_virtual_mem_map_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintVirtualMemUnmapParams(const struct ur_virtual_mem_unmap_params_t *params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintVirtualMemSetAccessParams(
    const struct ur_virtual_mem_set_access_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintVirtualMemGetInfoParams(
    const struct ur_virtual_mem_get_info_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceGetParams(const struct ur_device_get_params_t *params,
                                   char *buffer, const size_t buff_size,
                                   size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDeviceGetInfoParams(const struct ur_device_get_info_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDeviceRetainParams(const struct ur_device_retain_params_t *params,
                          char *buffer, const size_t buff_size,
                          size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDeviceReleaseParams(const struct ur_device_release_params_t *params,
                           char *buffer, const size_t buff_size,
                           size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t
urPrintDevicePartitionParams(const struct ur_device_partition_params_t *params,
                             char *buffer, const size_t buff_size,
                             size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceSelectBinaryParams(
    const struct ur_device_select_binary_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceGetNativeHandleParams(
    const struct ur_device_get_native_handle_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceCreateWithNativeHandleParams(
    const struct ur_device_create_with_native_handle_params_t *params,
    char *buffer, const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintDeviceGetGlobalTimestampsParams(
    const struct ur_device_get_global_timestamps_params_t *params, char *buffer,
    const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ss << params;
    return str_copy(&ss, buffer, buff_size, out_size);
}

ur_result_t urPrintFunctionParams(enum ur_function_t function,
                                  const void *params, char *buffer,
                                  const size_t buff_size, size_t *out_size) {
    std::stringstream ss;
    ur_result_t result = ur::extras::printFunctionParams(ss, function, params);
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    return str_copy(&ss, buffer, buff_size, out_size);
}
