/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_print.h
 *
 */
#ifndef UR_PRINT_H
#define UR_PRINT_H 1

#include "ur_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_function_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintFunction(enum ur_function_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_structure_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintStructureType(enum ur_structure_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_result_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintResult(enum ur_result_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_base_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBaseProperties(const struct ur_base_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_base_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBaseDesc(const struct ur_base_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_rect_offset_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintRectOffset(const struct ur_rect_offset_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_rect_region_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintRectRegion(const struct ur_rect_region_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_init_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceInitFlags(enum ur_device_init_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_config_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderConfigInfo(enum ur_loader_config_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_code_location_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCodeLocation(const struct ur_code_location_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_adapter_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintAdapterInfo(enum ur_adapter_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_adapter_backend_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintAdapterBackend(enum ur_adapter_backend_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformInfo(enum ur_platform_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_api_version_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintApiVersion(enum ur_api_version_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformNativeProperties(const struct ur_platform_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_backend_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformBackend(enum ur_platform_backend_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_binary_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceBinary(const struct ur_device_binary_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceType(enum ur_device_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceInfo(enum ur_device_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_affinity_domain_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceAffinityDomainFlags(enum ur_device_affinity_domain_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_partition_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDevicePartition(enum ur_device_partition_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_partition_property_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDevicePartitionProperty(const struct ur_device_partition_property_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_partition_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDevicePartitionProperties(const struct ur_device_partition_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_fp_capability_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceFpCapabilityFlags(enum ur_device_fp_capability_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_mem_cache_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceMemCacheType(enum ur_device_mem_cache_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_local_mem_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceLocalMemType(enum ur_device_local_mem_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_exec_capability_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceExecCapabilityFlags(enum ur_device_exec_capability_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceNativeProperties(const struct ur_device_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_memory_order_capability_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemoryOrderCapabilityFlags(enum ur_memory_order_capability_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_memory_scope_capability_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemoryScopeCapabilityFlags(enum ur_memory_scope_capability_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_usm_access_capability_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceUsmAccessCapabilityFlags(enum ur_device_usm_access_capability_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextFlags(enum ur_context_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextProperties(const struct ur_context_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextInfo(enum ur_context_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextNativeProperties(const struct ur_context_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemFlags(enum ur_mem_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemType(enum ur_mem_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemInfo(enum ur_mem_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_image_channel_order_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintImageChannelOrder(enum ur_image_channel_order_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_image_channel_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintImageChannelType(enum ur_image_channel_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_image_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintImageInfo(enum ur_image_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_image_format_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintImageFormat(const struct ur_image_format_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_image_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintImageDesc(const struct ur_image_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_buffer_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBufferProperties(const struct ur_buffer_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_buffer_channel_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBufferChannelProperties(const struct ur_buffer_channel_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_buffer_alloc_location_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBufferAllocLocationProperties(const struct ur_buffer_alloc_location_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_buffer_region_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBufferRegion(const struct ur_buffer_region_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_buffer_create_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBufferCreateType(enum ur_buffer_create_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemNativeProperties(const struct ur_mem_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_filter_mode_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerFilterMode(enum ur_sampler_filter_mode_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_addressing_mode_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerAddressingMode(enum ur_sampler_addressing_mode_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerInfo(enum ur_sampler_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerDesc(const struct ur_sampler_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerNativeProperties(const struct ur_sampler_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_host_mem_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmHostMemFlags(enum ur_usm_host_mem_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_device_mem_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmDeviceMemFlags(enum ur_usm_device_mem_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolFlags(enum ur_usm_pool_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmType(enum ur_usm_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_alloc_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmAllocInfo(enum ur_usm_alloc_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_advice_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmAdviceFlags(enum ur_usm_advice_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmDesc(const struct ur_usm_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_host_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmHostDesc(const struct ur_usm_host_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_device_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmDeviceDesc(const struct ur_usm_device_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_alloc_location_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmAllocLocationDesc(const struct ur_usm_alloc_location_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolDesc(const struct ur_usm_pool_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_limits_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolLimitsDesc(const struct ur_usm_pool_limits_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolInfo(enum ur_usm_pool_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_granularity_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemGranularityInfo(enum ur_virtual_mem_granularity_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_access_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemAccessFlags(enum ur_virtual_mem_access_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemInfo(enum ur_virtual_mem_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_physical_mem_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPhysicalMemFlags(enum ur_physical_mem_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_physical_mem_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPhysicalMemProperties(const struct ur_physical_mem_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_metadata_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramMetadataType(enum ur_program_metadata_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_metadata_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramMetadata(const struct ur_program_metadata_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramProperties(const struct ur_program_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramInfo(enum ur_program_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_build_status_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramBuildStatus(enum ur_program_build_status_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_binary_type_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramBinaryType(enum ur_program_binary_type_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_build_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramBuildInfo(enum ur_program_build_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_specialization_constant_info_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSpecializationConstantInfo(const struct ur_specialization_constant_info_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramNativeProperties(const struct ur_program_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_arg_value_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelArgValueProperties(const struct ur_kernel_arg_value_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_arg_local_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelArgLocalProperties(const struct ur_kernel_arg_local_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelInfo(enum ur_kernel_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_group_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelGroupInfo(enum ur_kernel_group_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_sub_group_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSubGroupInfo(enum ur_kernel_sub_group_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_cache_config_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelCacheConfig(enum ur_kernel_cache_config_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_exec_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelExecInfo(enum ur_kernel_exec_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_arg_pointer_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelArgPointerProperties(const struct ur_kernel_arg_pointer_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_exec_info_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelExecInfoProperties(const struct ur_kernel_exec_info_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_arg_sampler_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelArgSamplerProperties(const struct ur_kernel_arg_sampler_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_arg_mem_obj_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelArgMemObjProperties(const struct ur_kernel_arg_mem_obj_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelNativeProperties(const struct ur_kernel_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueInfo(enum ur_queue_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueFlags(enum ur_queue_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueProperties(const struct ur_queue_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_index_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueIndexProperties(const struct ur_queue_index_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_native_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueNativeDesc(const struct ur_queue_native_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueNativeProperties(const struct ur_queue_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommand(enum ur_command_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_status_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventStatus(enum ur_event_status_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventInfo(enum ur_event_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_profiling_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProfilingInfo(enum ur_profiling_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_native_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventNativeProperties(const struct ur_event_native_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_execution_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExecutionInfo(enum ur_execution_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_map_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMapFlags(enum ur_map_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_migration_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmMigrationFlags(enum ur_usm_migration_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_image_copy_flag_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpImageCopyFlags(enum ur_exp_image_copy_flag_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_file_descriptor_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpFileDescriptor(const struct ur_exp_file_descriptor_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_win32_handle_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpWin32Handle(const struct ur_exp_win32_handle_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_sampler_mip_properties_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpSamplerMipProperties(const struct ur_exp_sampler_mip_properties_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_sampler_addr_modes_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpSamplerAddrModes(const struct ur_exp_sampler_addr_modes_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_interop_mem_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpInteropMemDesc(const struct ur_exp_interop_mem_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_interop_semaphore_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpInteropSemaphoreDesc(const struct ur_exp_interop_semaphore_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_command_buffer_desc_t struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpCommandBufferDesc(const struct ur_exp_command_buffer_desc_t params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_exp_peer_info_t enum
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintExpPeerInfo(enum ur_exp_peer_info_t value, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_config_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderConfigCreateParams(const struct ur_loader_config_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_config_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderConfigRetainParams(const struct ur_loader_config_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_config_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderConfigReleaseParams(const struct ur_loader_config_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_config_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderConfigGetInfoParams(const struct ur_loader_config_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_config_enable_layer_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderConfigEnableLayerParams(const struct ur_loader_config_enable_layer_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_config_set_code_location_callback_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderConfigSetCodeLocationCallbackParams(const struct ur_loader_config_set_code_location_callback_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_get_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformGetParams(const struct ur_platform_get_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformGetInfoParams(const struct ur_platform_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformGetNativeHandleParams(const struct ur_platform_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformCreateWithNativeHandleParams(const struct ur_platform_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_get_api_version_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformGetApiVersionParams(const struct ur_platform_get_api_version_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_platform_get_backend_option_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPlatformGetBackendOptionParams(const struct ur_platform_get_backend_option_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextCreateParams(const struct ur_context_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextRetainParams(const struct ur_context_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextReleaseParams(const struct ur_context_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextGetInfoParams(const struct ur_context_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextGetNativeHandleParams(const struct ur_context_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextCreateWithNativeHandleParams(const struct ur_context_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_context_set_extended_deleter_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintContextSetExtendedDeleterParams(const struct ur_context_set_extended_deleter_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventGetInfoParams(const struct ur_event_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_get_profiling_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventGetProfilingInfoParams(const struct ur_event_get_profiling_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_wait_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventWaitParams(const struct ur_event_wait_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventRetainParams(const struct ur_event_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventReleaseParams(const struct ur_event_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventGetNativeHandleParams(const struct ur_event_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventCreateWithNativeHandleParams(const struct ur_event_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_event_set_callback_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEventSetCallbackParams(const struct ur_event_set_callback_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_create_with_il_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramCreateWithIlParams(const struct ur_program_create_with_il_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_create_with_binary_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramCreateWithBinaryParams(const struct ur_program_create_with_binary_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_build_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramBuildParams(const struct ur_program_build_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_build_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramBuildExpParams(const struct ur_program_build_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_compile_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramCompileParams(const struct ur_program_compile_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_compile_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramCompileExpParams(const struct ur_program_compile_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_link_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramLinkParams(const struct ur_program_link_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_link_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramLinkExpParams(const struct ur_program_link_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramRetainParams(const struct ur_program_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramReleaseParams(const struct ur_program_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_get_function_pointer_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramGetFunctionPointerParams(const struct ur_program_get_function_pointer_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramGetInfoParams(const struct ur_program_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_get_build_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramGetBuildInfoParams(const struct ur_program_get_build_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_set_specialization_constants_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramSetSpecializationConstantsParams(const struct ur_program_set_specialization_constants_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramGetNativeHandleParams(const struct ur_program_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_program_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintProgramCreateWithNativeHandleParams(const struct ur_program_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelCreateParams(const struct ur_kernel_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelGetInfoParams(const struct ur_kernel_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_get_group_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelGetGroupInfoParams(const struct ur_kernel_get_group_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_get_sub_group_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelGetSubGroupInfoParams(const struct ur_kernel_get_sub_group_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelRetainParams(const struct ur_kernel_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelReleaseParams(const struct ur_kernel_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelGetNativeHandleParams(const struct ur_kernel_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelCreateWithNativeHandleParams(const struct ur_kernel_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_set_arg_value_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSetArgValueParams(const struct ur_kernel_set_arg_value_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_set_arg_local_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSetArgLocalParams(const struct ur_kernel_set_arg_local_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_set_arg_pointer_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSetArgPointerParams(const struct ur_kernel_set_arg_pointer_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_set_exec_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSetExecInfoParams(const struct ur_kernel_set_exec_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_set_arg_sampler_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSetArgSamplerParams(const struct ur_kernel_set_arg_sampler_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_set_arg_mem_obj_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSetArgMemObjParams(const struct ur_kernel_set_arg_mem_obj_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_set_specialization_constants_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSetSpecializationConstantsParams(const struct ur_kernel_set_specialization_constants_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_kernel_suggest_max_cooperative_group_count_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintKernelSuggestMaxCooperativeGroupCountExpParams(const struct ur_kernel_suggest_max_cooperative_group_count_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerCreateParams(const struct ur_sampler_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerRetainParams(const struct ur_sampler_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerReleaseParams(const struct ur_sampler_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerGetInfoParams(const struct ur_sampler_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerGetNativeHandleParams(const struct ur_sampler_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_sampler_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintSamplerCreateWithNativeHandleParams(const struct ur_sampler_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_image_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemImageCreateParams(const struct ur_mem_image_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_buffer_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemBufferCreateParams(const struct ur_mem_buffer_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemRetainParams(const struct ur_mem_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemReleaseParams(const struct ur_mem_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_buffer_partition_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemBufferPartitionParams(const struct ur_mem_buffer_partition_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemGetNativeHandleParams(const struct ur_mem_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_buffer_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemBufferCreateWithNativeHandleParams(const struct ur_mem_buffer_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_image_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemImageCreateWithNativeHandleParams(const struct ur_mem_image_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemGetInfoParams(const struct ur_mem_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_mem_image_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintMemImageGetInfoParams(const struct ur_mem_image_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_physical_mem_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPhysicalMemCreateParams(const struct ur_physical_mem_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_physical_mem_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPhysicalMemRetainParams(const struct ur_physical_mem_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_physical_mem_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintPhysicalMemReleaseParams(const struct ur_physical_mem_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_adapter_get_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintAdapterGetParams(const struct ur_adapter_get_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_adapter_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintAdapterReleaseParams(const struct ur_adapter_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_adapter_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintAdapterRetainParams(const struct ur_adapter_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_adapter_get_last_error_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintAdapterGetLastErrorParams(const struct ur_adapter_get_last_error_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_adapter_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintAdapterGetInfoParams(const struct ur_adapter_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_kernel_launch_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueKernelLaunchParams(const struct ur_enqueue_kernel_launch_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_events_wait_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueEventsWaitParams(const struct ur_enqueue_events_wait_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_events_wait_with_barrier_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueEventsWaitWithBarrierParams(const struct ur_enqueue_events_wait_with_barrier_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_read_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferReadParams(const struct ur_enqueue_mem_buffer_read_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_write_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferWriteParams(const struct ur_enqueue_mem_buffer_write_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_read_rect_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferReadRectParams(const struct ur_enqueue_mem_buffer_read_rect_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_write_rect_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferWriteRectParams(const struct ur_enqueue_mem_buffer_write_rect_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_copy_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferCopyParams(const struct ur_enqueue_mem_buffer_copy_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_copy_rect_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferCopyRectParams(const struct ur_enqueue_mem_buffer_copy_rect_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_fill_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferFillParams(const struct ur_enqueue_mem_buffer_fill_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_image_read_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemImageReadParams(const struct ur_enqueue_mem_image_read_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_image_write_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemImageWriteParams(const struct ur_enqueue_mem_image_write_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_image_copy_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemImageCopyParams(const struct ur_enqueue_mem_image_copy_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_buffer_map_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemBufferMapParams(const struct ur_enqueue_mem_buffer_map_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_mem_unmap_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueMemUnmapParams(const struct ur_enqueue_mem_unmap_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_usm_fill_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueUsmFillParams(const struct ur_enqueue_usm_fill_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_usm_memcpy_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueUsmMemcpyParams(const struct ur_enqueue_usm_memcpy_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_usm_prefetch_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueUsmPrefetchParams(const struct ur_enqueue_usm_prefetch_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_usm_advise_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueUsmAdviseParams(const struct ur_enqueue_usm_advise_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_usm_fill_2d_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueUsmFill_2dParams(const struct ur_enqueue_usm_fill_2d_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_usm_memcpy_2d_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueUsmMemcpy_2dParams(const struct ur_enqueue_usm_memcpy_2d_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_device_global_variable_write_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueDeviceGlobalVariableWriteParams(const struct ur_enqueue_device_global_variable_write_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_device_global_variable_read_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueDeviceGlobalVariableReadParams(const struct ur_enqueue_device_global_variable_read_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_read_host_pipe_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueReadHostPipeParams(const struct ur_enqueue_read_host_pipe_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_write_host_pipe_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueWriteHostPipeParams(const struct ur_enqueue_write_host_pipe_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_enqueue_cooperative_kernel_launch_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintEnqueueCooperativeKernelLaunchExpParams(const struct ur_enqueue_cooperative_kernel_launch_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueGetInfoParams(const struct ur_queue_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueCreateParams(const struct ur_queue_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueRetainParams(const struct ur_queue_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueReleaseParams(const struct ur_queue_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueGetNativeHandleParams(const struct ur_queue_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueCreateWithNativeHandleParams(const struct ur_queue_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_finish_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueFinishParams(const struct ur_queue_finish_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_queue_flush_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintQueueFlushParams(const struct ur_queue_flush_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_unsampled_image_handle_destroy_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesUnsampledImageHandleDestroyExpParams(const struct ur_bindless_images_unsampled_image_handle_destroy_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_sampled_image_handle_destroy_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesSampledImageHandleDestroyExpParams(const struct ur_bindless_images_sampled_image_handle_destroy_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_image_allocate_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesImageAllocateExpParams(const struct ur_bindless_images_image_allocate_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_image_free_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesImageFreeExpParams(const struct ur_bindless_images_image_free_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_unsampled_image_create_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesUnsampledImageCreateExpParams(const struct ur_bindless_images_unsampled_image_create_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_sampled_image_create_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesSampledImageCreateExpParams(const struct ur_bindless_images_sampled_image_create_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_image_copy_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesImageCopyExpParams(const struct ur_bindless_images_image_copy_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_image_get_info_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesImageGetInfoExpParams(const struct ur_bindless_images_image_get_info_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_mipmap_get_level_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesMipmapGetLevelExpParams(const struct ur_bindless_images_mipmap_get_level_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_mipmap_free_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesMipmapFreeExpParams(const struct ur_bindless_images_mipmap_free_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_import_opaque_fd_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesImportOpaqueFdExpParams(const struct ur_bindless_images_import_opaque_fd_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_map_external_array_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesMapExternalArrayExpParams(const struct ur_bindless_images_map_external_array_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_release_interop_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesReleaseInteropExpParams(const struct ur_bindless_images_release_interop_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_import_external_semaphore_opaque_fd_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesImportExternalSemaphoreOpaqueFdExpParams(const struct ur_bindless_images_import_external_semaphore_opaque_fd_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_destroy_external_semaphore_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesDestroyExternalSemaphoreExpParams(const struct ur_bindless_images_destroy_external_semaphore_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_wait_external_semaphore_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesWaitExternalSemaphoreExpParams(const struct ur_bindless_images_wait_external_semaphore_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_bindless_images_signal_external_semaphore_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintBindlessImagesSignalExternalSemaphoreExpParams(const struct ur_bindless_images_signal_external_semaphore_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_host_alloc_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmHostAllocParams(const struct ur_usm_host_alloc_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_device_alloc_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmDeviceAllocParams(const struct ur_usm_device_alloc_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_shared_alloc_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmSharedAllocParams(const struct ur_usm_shared_alloc_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_free_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmFreeParams(const struct ur_usm_free_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_get_mem_alloc_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmGetMemAllocInfoParams(const struct ur_usm_get_mem_alloc_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_create_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolCreateParams(const struct ur_usm_pool_create_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolRetainParams(const struct ur_usm_pool_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolReleaseParams(const struct ur_usm_pool_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pool_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPoolGetInfoParams(const struct ur_usm_pool_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_pitched_alloc_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmPitchedAllocExpParams(const struct ur_usm_pitched_alloc_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_import_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmImportExpParams(const struct ur_usm_import_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_release_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmReleaseExpParams(const struct ur_usm_release_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_create_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferCreateExpParams(const struct ur_command_buffer_create_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_retain_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferRetainExpParams(const struct ur_command_buffer_retain_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_release_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferReleaseExpParams(const struct ur_command_buffer_release_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_finalize_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferFinalizeExpParams(const struct ur_command_buffer_finalize_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_kernel_launch_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendKernelLaunchExpParams(const struct ur_command_buffer_append_kernel_launch_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_usm_memcpy_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendUsmMemcpyExpParams(const struct ur_command_buffer_append_usm_memcpy_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_usm_fill_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendUsmFillExpParams(const struct ur_command_buffer_append_usm_fill_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_mem_buffer_copy_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendMemBufferCopyExpParams(const struct ur_command_buffer_append_mem_buffer_copy_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_mem_buffer_write_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendMemBufferWriteExpParams(const struct ur_command_buffer_append_mem_buffer_write_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_mem_buffer_read_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendMemBufferReadExpParams(const struct ur_command_buffer_append_mem_buffer_read_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_mem_buffer_copy_rect_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendMemBufferCopyRectExpParams(const struct ur_command_buffer_append_mem_buffer_copy_rect_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_mem_buffer_write_rect_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendMemBufferWriteRectExpParams(const struct ur_command_buffer_append_mem_buffer_write_rect_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_mem_buffer_read_rect_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendMemBufferReadRectExpParams(const struct ur_command_buffer_append_mem_buffer_read_rect_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_mem_buffer_fill_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendMemBufferFillExpParams(const struct ur_command_buffer_append_mem_buffer_fill_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_usm_prefetch_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendUsmPrefetchExpParams(const struct ur_command_buffer_append_usm_prefetch_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_append_usm_advise_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferAppendUsmAdviseExpParams(const struct ur_command_buffer_append_usm_advise_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_command_buffer_enqueue_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintCommandBufferEnqueueExpParams(const struct ur_command_buffer_enqueue_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_p2p_enable_peer_access_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmP2pEnablePeerAccessExpParams(const struct ur_usm_p2p_enable_peer_access_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_p2p_disable_peer_access_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmP2pDisablePeerAccessExpParams(const struct ur_usm_p2p_disable_peer_access_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_usm_p2p_peer_access_get_info_exp_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintUsmP2pPeerAccessGetInfoExpParams(const struct ur_usm_p2p_peer_access_get_info_exp_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_init_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderInitParams(const struct ur_loader_init_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_loader_tear_down_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintLoaderTearDownParams(const struct ur_loader_tear_down_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_granularity_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemGranularityGetInfoParams(const struct ur_virtual_mem_granularity_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_reserve_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemReserveParams(const struct ur_virtual_mem_reserve_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_free_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemFreeParams(const struct ur_virtual_mem_free_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_map_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemMapParams(const struct ur_virtual_mem_map_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_unmap_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemUnmapParams(const struct ur_virtual_mem_unmap_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_set_access_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemSetAccessParams(const struct ur_virtual_mem_set_access_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_virtual_mem_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintVirtualMemGetInfoParams(const struct ur_virtual_mem_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_get_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceGetParams(const struct ur_device_get_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_get_info_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceGetInfoParams(const struct ur_device_get_info_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_retain_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceRetainParams(const struct ur_device_retain_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_release_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceReleaseParams(const struct ur_device_release_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_partition_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDevicePartitionParams(const struct ur_device_partition_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_select_binary_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceSelectBinaryParams(const struct ur_device_select_binary_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_get_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceGetNativeHandleParams(const struct ur_device_get_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_create_with_native_handle_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceCreateWithNativeHandleParams(const struct ur_device_create_with_native_handle_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print ur_device_get_global_timestamps_params_t params struct
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintDeviceGetGlobalTimestampsParams(const struct ur_device_get_global_timestamps_params_t *params, char *buffer, const size_t buff_size, size_t *out_size);

///////////////////////////////////////////////////////////////////////////////
/// @brief Print function parameters
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         - `NULL == params`
///         - `NULL == buffer`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         - `buff_size < out_size`
UR_APIEXPORT ur_result_t UR_APICALL urPrintFunctionParams(enum ur_function_t function, const void *params, char *buffer, const size_t buff_size, size_t *out_size);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* UR_PRINT_H */
