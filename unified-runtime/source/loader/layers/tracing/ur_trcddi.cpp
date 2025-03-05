/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_trcddi.cpp
 *
 */

#include "ur_tracing_layer.hpp"
#include <iostream>
#include <stdio.h>

namespace ur_tracing_layer {
///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGet
__urdlllocal ur_result_t UR_APICALL urAdapterGet(
    /// [in] the number of adapters to be added to phAdapters.
    /// If phAdapters is not NULL, then NumEntries should be greater than
    /// zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
    /// will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)][alloc] array of handle of
    /// adapters. If NumEntries is less than the number of adapters available,
    /// then
    /// ::urAdapterGet shall only retrieve that number of adapters.
    ur_adapter_handle_t *phAdapters,
    /// [out][optional] returns the total number of adapters available.
    uint32_t *pNumAdapters) {
  auto pfnAdapterGet = getContext()->urDdiTable.Global.pfnAdapterGet;

  if (nullptr == pfnAdapterGet)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_adapter_get_params_t params = {&NumEntries, &phAdapters, &pNumAdapters};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ADAPTER_GET,
                                                 "urAdapterGet", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urAdapterGet\n");

  ur_result_t result = pfnAdapterGet(NumEntries, phAdapters, pNumAdapters);

  getContext()->notify_end(UR_FUNCTION_ADAPTER_GET, "urAdapterGet", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ADAPTER_GET, &params);
    logger.info("   <--- urAdapterGet({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRelease
__urdlllocal ur_result_t UR_APICALL urAdapterRelease(
    /// [in][release] Adapter handle to release
    ur_adapter_handle_t hAdapter) {
  auto pfnAdapterRelease = getContext()->urDdiTable.Global.pfnAdapterRelease;

  if (nullptr == pfnAdapterRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_adapter_release_params_t params = {&hAdapter};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ADAPTER_RELEASE,
                                                 "urAdapterRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urAdapterRelease\n");

  ur_result_t result = pfnAdapterRelease(hAdapter);

  getContext()->notify_end(UR_FUNCTION_ADAPTER_RELEASE, "urAdapterRelease",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ADAPTER_RELEASE,
                                    &params);
    logger.info("   <--- urAdapterRelease({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRetain
__urdlllocal ur_result_t UR_APICALL urAdapterRetain(
    /// [in][retain] Adapter handle to retain
    ur_adapter_handle_t hAdapter) {
  auto pfnAdapterRetain = getContext()->urDdiTable.Global.pfnAdapterRetain;

  if (nullptr == pfnAdapterRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_adapter_retain_params_t params = {&hAdapter};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ADAPTER_RETAIN,
                                                 "urAdapterRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urAdapterRetain\n");

  ur_result_t result = pfnAdapterRetain(hAdapter);

  getContext()->notify_end(UR_FUNCTION_ADAPTER_RETAIN, "urAdapterRetain",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ADAPTER_RETAIN,
                                    &params);
    logger.info("   <--- urAdapterRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGetLastError
__urdlllocal ur_result_t UR_APICALL urAdapterGetLastError(
    /// [in] handle of the adapter instance
    ur_adapter_handle_t hAdapter,
    /// [out] pointer to a C string where the adapter specific error message
    /// will be stored.
    const char **ppMessage,
    /// [out] pointer to an integer where the adapter specific error code will
    /// be stored.
    int32_t *pError) {
  auto pfnAdapterGetLastError =
      getContext()->urDdiTable.Global.pfnAdapterGetLastError;

  if (nullptr == pfnAdapterGetLastError)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_adapter_get_last_error_params_t params = {&hAdapter, &ppMessage, &pError};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ADAPTER_GET_LAST_ERROR, "urAdapterGetLastError", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urAdapterGetLastError\n");

  ur_result_t result = pfnAdapterGetLastError(hAdapter, ppMessage, pError);

  getContext()->notify_end(UR_FUNCTION_ADAPTER_GET_LAST_ERROR,
                           "urAdapterGetLastError", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ADAPTER_GET_LAST_ERROR, &params);
    logger.info("   <--- urAdapterGetLastError({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGetInfo
__urdlllocal ur_result_t UR_APICALL urAdapterGetInfo(
    /// [in] handle of the adapter
    ur_adapter_handle_t hAdapter,
    /// [in] type of the info to retrieve
    ur_adapter_info_t propName,
    /// [in] the number of bytes pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info.
    /// If Size is not equal to or greater to the real number of bytes needed
    /// to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    /// returned and pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual number of bytes being queried by
    /// pPropValue.
    size_t *pPropSizeRet) {
  auto pfnAdapterGetInfo = getContext()->urDdiTable.Global.pfnAdapterGetInfo;

  if (nullptr == pfnAdapterGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_adapter_get_info_params_t params = {&hAdapter, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ADAPTER_GET_INFO,
                                                 "urAdapterGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urAdapterGetInfo\n");

  ur_result_t result =
      pfnAdapterGetInfo(hAdapter, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_ADAPTER_GET_INFO, "urAdapterGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ADAPTER_GET_INFO,
                                    &params);
    logger.info("   <--- urAdapterGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGet
__urdlllocal ur_result_t UR_APICALL urPlatformGet(
    /// [in][range(0, NumAdapters)] array of adapters to query for platforms.
    ur_adapter_handle_t *phAdapters,
    /// [in] number of adapters pointed to by phAdapters
    uint32_t NumAdapters,
    /// [in] the number of platforms to be added to phPlatforms.
    /// If phPlatforms is not NULL, then NumEntries should be greater than
    /// zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
    /// will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)] array of handle of platforms.
    /// If NumEntries is less than the number of platforms available, then
    /// ::urPlatformGet shall only retrieve that number of platforms.
    ur_platform_handle_t *phPlatforms,
    /// [out][optional] returns the total number of platforms available.
    uint32_t *pNumPlatforms) {
  auto pfnGet = getContext()->urDdiTable.Platform.pfnGet;

  if (nullptr == pfnGet)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_platform_get_params_t params = {&phAdapters, &NumAdapters, &NumEntries,
                                     &phPlatforms, &pNumPlatforms};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PLATFORM_GET,
                                                 "urPlatformGet", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPlatformGet\n");

  ur_result_t result =
      pfnGet(phAdapters, NumAdapters, NumEntries, phPlatforms, pNumPlatforms);

  getContext()->notify_end(UR_FUNCTION_PLATFORM_GET, "urPlatformGet", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PLATFORM_GET,
                                    &params);
    logger.info("   <--- urPlatformGet({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetInfo
__urdlllocal ur_result_t UR_APICALL urPlatformGetInfo(
    /// [in] handle of the platform
    ur_platform_handle_t hPlatform,
    /// [in] type of the info to retrieve
    ur_platform_info_t propName,
    /// [in] the number of bytes pointed to by pPlatformInfo.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info.
    /// If Size is not equal to or greater to the real number of bytes needed
    /// to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    /// returned and pPlatformInfo is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual number of bytes being queried by
    /// pPlatformInfo.
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Platform.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_platform_get_info_params_t params = {&hPlatform, &propName, &propSize,
                                          &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PLATFORM_GET_INFO,
                                                 "urPlatformGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPlatformGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hPlatform, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_PLATFORM_GET_INFO, "urPlatformGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PLATFORM_GET_INFO,
                                    &params);
    logger.info("   <--- urPlatformGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetApiVersion
__urdlllocal ur_result_t UR_APICALL urPlatformGetApiVersion(
    /// [in] handle of the platform
    ur_platform_handle_t hPlatform,
    /// [out] api version
    ur_api_version_t *pVersion) {
  auto pfnGetApiVersion = getContext()->urDdiTable.Platform.pfnGetApiVersion;

  if (nullptr == pfnGetApiVersion)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_platform_get_api_version_params_t params = {&hPlatform, &pVersion};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PLATFORM_GET_API_VERSION, "urPlatformGetApiVersion", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPlatformGetApiVersion\n");

  ur_result_t result = pfnGetApiVersion(hPlatform, pVersion);

  getContext()->notify_end(UR_FUNCTION_PLATFORM_GET_API_VERSION,
                           "urPlatformGetApiVersion", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PLATFORM_GET_API_VERSION, &params);
    logger.info("   <--- urPlatformGetApiVersion({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformGetNativeHandle(
    /// [in] handle of the platform.
    ur_platform_handle_t hPlatform,
    /// [out] a pointer to the native handle of the platform.
    ur_native_handle_t *phNativePlatform) {
  auto pfnGetNativeHandle =
      getContext()->urDdiTable.Platform.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_platform_get_native_handle_params_t params = {&hPlatform,
                                                   &phNativePlatform};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_PLATFORM_GET_NATIVE_HANDLE,
                                 "urPlatformGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPlatformGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hPlatform, phNativePlatform);

  getContext()->notify_end(UR_FUNCTION_PLATFORM_GET_NATIVE_HANDLE,
                           "urPlatformGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PLATFORM_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urPlatformGetNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the platform.
    ur_native_handle_t hNativePlatform,
    /// [in] handle of the adapter associated with the native backend.
    ur_adapter_handle_t hAdapter,
    /// [in][optional] pointer to native platform properties struct.
    const ur_platform_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the platform object created.
    ur_platform_handle_t *phPlatform) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Platform.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_platform_create_with_native_handle_params_t params = {
      &hNativePlatform, &hAdapter, &pProperties, &phPlatform};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_PLATFORM_CREATE_WITH_NATIVE_HANDLE,
                                 "urPlatformCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPlatformCreateWithNativeHandle\n");

  ur_result_t result = pfnCreateWithNativeHandle(hNativePlatform, hAdapter,
                                                 pProperties, phPlatform);

  getContext()->notify_end(UR_FUNCTION_PLATFORM_CREATE_WITH_NATIVE_HANDLE,
                           "urPlatformCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PLATFORM_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urPlatformCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetBackendOption
__urdlllocal ur_result_t UR_APICALL urPlatformGetBackendOption(
    /// [in] handle of the platform instance.
    ur_platform_handle_t hPlatform,
    /// [in] string containing the frontend option.
    const char *pFrontendOption,
    /// [out] returns the correct platform specific compiler option based on
    /// the frontend option.
    const char **ppPlatformOption) {
  auto pfnGetBackendOption =
      getContext()->urDdiTable.Platform.pfnGetBackendOption;

  if (nullptr == pfnGetBackendOption)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_platform_get_backend_option_params_t params = {
      &hPlatform, &pFrontendOption, &ppPlatformOption};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_PLATFORM_GET_BACKEND_OPTION,
                                 "urPlatformGetBackendOption", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPlatformGetBackendOption\n");

  ur_result_t result =
      pfnGetBackendOption(hPlatform, pFrontendOption, ppPlatformOption);

  getContext()->notify_end(UR_FUNCTION_PLATFORM_GET_BACKEND_OPTION,
                           "urPlatformGetBackendOption", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PLATFORM_GET_BACKEND_OPTION, &params);
    logger.info("   <--- urPlatformGetBackendOption({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGet
__urdlllocal ur_result_t UR_APICALL urDeviceGet(
    /// [in] handle of the platform instance
    ur_platform_handle_t hPlatform,
    /// [in] the type of the devices.
    ur_device_type_t DeviceType,
    /// [in] the number of devices to be added to phDevices.
    /// If phDevices is not NULL, then NumEntries should be greater than zero.
    /// Otherwise ::UR_RESULT_ERROR_INVALID_SIZE
    /// will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)][alloc] array of handle of devices.
    /// If NumEntries is less than the number of devices available, then
    /// platform shall only retrieve that number of devices.
    ur_device_handle_t *phDevices,
    /// [out][optional] pointer to the number of devices.
    /// pNumDevices will be updated with the total number of devices available.
    uint32_t *pNumDevices) {
  auto pfnGet = getContext()->urDdiTable.Device.pfnGet;

  if (nullptr == pfnGet)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_get_params_t params = {&hPlatform, &DeviceType, &NumEntries,
                                   &phDevices, &pNumDevices};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_DEVICE_GET,
                                                 "urDeviceGet", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceGet\n");

  ur_result_t result =
      pfnGet(hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);

  getContext()->notify_end(UR_FUNCTION_DEVICE_GET, "urDeviceGet", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_DEVICE_GET, &params);
    logger.info("   <--- urDeviceGet({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetInfo
__urdlllocal ur_result_t UR_APICALL urDeviceGetInfo(
    /// [in] handle of the device instance
    ur_device_handle_t hDevice,
    /// [in] type of the info to retrieve
    ur_device_info_t propName,
    /// [in] the number of bytes pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info.
    /// If propSize is not equal to or greater than the real number of bytes
    /// needed to return the info
    /// then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName.
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Device.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_get_info_params_t params = {&hDevice, &propName, &propSize,
                                        &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_DEVICE_GET_INFO,
                                                 "urDeviceGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hDevice, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_DEVICE_GET_INFO, "urDeviceGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_DEVICE_GET_INFO,
                                    &params);
    logger.info("   <--- urDeviceGetInfo({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRetain
__urdlllocal ur_result_t UR_APICALL urDeviceRetain(
    /// [in][retain] handle of the device to get a reference of.
    ur_device_handle_t hDevice) {
  auto pfnRetain = getContext()->urDdiTable.Device.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_retain_params_t params = {&hDevice};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_DEVICE_RETAIN,
                                                 "urDeviceRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceRetain\n");

  ur_result_t result = pfnRetain(hDevice);

  getContext()->notify_end(UR_FUNCTION_DEVICE_RETAIN, "urDeviceRetain", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_DEVICE_RETAIN,
                                    &params);
    logger.info("   <--- urDeviceRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRelease
__urdlllocal ur_result_t UR_APICALL urDeviceRelease(
    /// [in][release] handle of the device to release.
    ur_device_handle_t hDevice) {
  auto pfnRelease = getContext()->urDdiTable.Device.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_release_params_t params = {&hDevice};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_DEVICE_RELEASE,
                                                 "urDeviceRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceRelease\n");

  ur_result_t result = pfnRelease(hDevice);

  getContext()->notify_end(UR_FUNCTION_DEVICE_RELEASE, "urDeviceRelease",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_DEVICE_RELEASE,
                                    &params);
    logger.info("   <--- urDeviceRelease({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDevicePartition
__urdlllocal ur_result_t UR_APICALL urDevicePartition(
    /// [in] handle of the device to partition.
    ur_device_handle_t hDevice,
    /// [in] Device partition properties.
    const ur_device_partition_properties_t *pProperties,
    /// [in] the number of sub-devices.
    uint32_t NumDevices,
    /// [out][optional][range(0, NumDevices)] array of handle of devices.
    /// If NumDevices is less than the number of sub-devices available, then
    /// the function shall only retrieve that number of sub-devices.
    ur_device_handle_t *phSubDevices,
    /// [out][optional] pointer to the number of sub-devices the device can be
    /// partitioned into according to the partitioning property.
    uint32_t *pNumDevicesRet) {
  auto pfnPartition = getContext()->urDdiTable.Device.pfnPartition;

  if (nullptr == pfnPartition)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_partition_params_t params = {&hDevice, &pProperties, &NumDevices,
                                         &phSubDevices, &pNumDevicesRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_DEVICE_PARTITION,
                                                 "urDevicePartition", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDevicePartition\n");

  ur_result_t result = pfnPartition(hDevice, pProperties, NumDevices,
                                    phSubDevices, pNumDevicesRet);

  getContext()->notify_end(UR_FUNCTION_DEVICE_PARTITION, "urDevicePartition",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_DEVICE_PARTITION,
                                    &params);
    logger.info("   <--- urDevicePartition({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceSelectBinary
__urdlllocal ur_result_t UR_APICALL urDeviceSelectBinary(
    /// [in] handle of the device to select binary for.
    ur_device_handle_t hDevice,
    /// [in] the array of binaries to select from.
    const ur_device_binary_t *pBinaries,
    /// [in] the number of binaries passed in ppBinaries.
    /// Must greater than or equal to zero otherwise
    /// ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t NumBinaries,
    /// [out] the index of the selected binary in the input array of binaries.
    /// If a suitable binary was not found the function returns
    /// ::UR_RESULT_ERROR_INVALID_BINARY.
    uint32_t *pSelectedBinary) {
  auto pfnSelectBinary = getContext()->urDdiTable.Device.pfnSelectBinary;

  if (nullptr == pfnSelectBinary)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_select_binary_params_t params = {&hDevice, &pBinaries, &NumBinaries,
                                             &pSelectedBinary};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_DEVICE_SELECT_BINARY, "urDeviceSelectBinary", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceSelectBinary\n");

  ur_result_t result =
      pfnSelectBinary(hDevice, pBinaries, NumBinaries, pSelectedBinary);

  getContext()->notify_end(UR_FUNCTION_DEVICE_SELECT_BINARY,
                           "urDeviceSelectBinary", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_DEVICE_SELECT_BINARY,
                                    &params);
    logger.info("   <--- urDeviceSelectBinary({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceGetNativeHandle(
    /// [in] handle of the device.
    ur_device_handle_t hDevice,
    /// [out] a pointer to the native handle of the device.
    ur_native_handle_t *phNativeDevice) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Device.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_get_native_handle_params_t params = {&hDevice, &phNativeDevice};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_DEVICE_GET_NATIVE_HANDLE, "urDeviceGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hDevice, phNativeDevice);

  getContext()->notify_end(UR_FUNCTION_DEVICE_GET_NATIVE_HANDLE,
                           "urDeviceGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_DEVICE_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urDeviceGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the device.
    ur_native_handle_t hNativeDevice,
    /// [in] handle of the adapter to which `hNativeDevice` belongs
    ur_adapter_handle_t hAdapter,
    /// [in][optional] pointer to native device properties struct.
    const ur_device_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the device object created.
    ur_device_handle_t *phDevice) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Device.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_create_with_native_handle_params_t params = {
      &hNativeDevice, &hAdapter, &pProperties, &phDevice};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_DEVICE_CREATE_WITH_NATIVE_HANDLE,
                                 "urDeviceCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceCreateWithNativeHandle\n");

  ur_result_t result =
      pfnCreateWithNativeHandle(hNativeDevice, hAdapter, pProperties, phDevice);

  getContext()->notify_end(UR_FUNCTION_DEVICE_CREATE_WITH_NATIVE_HANDLE,
                           "urDeviceCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_DEVICE_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urDeviceCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetGlobalTimestamps
__urdlllocal ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(
    /// [in] handle of the device instance
    ur_device_handle_t hDevice,
    /// [out][optional] pointer to the Device's global timestamp that
    /// correlates with the Host's global timestamp value
    uint64_t *pDeviceTimestamp,
    /// [out][optional] pointer to the Host's global timestamp that
    /// correlates with the Device's global timestamp value
    uint64_t *pHostTimestamp) {
  auto pfnGetGlobalTimestamps =
      getContext()->urDdiTable.Device.pfnGetGlobalTimestamps;

  if (nullptr == pfnGetGlobalTimestamps)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_device_get_global_timestamps_params_t params = {
      &hDevice, &pDeviceTimestamp, &pHostTimestamp};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_DEVICE_GET_GLOBAL_TIMESTAMPS,
                                 "urDeviceGetGlobalTimestamps", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urDeviceGetGlobalTimestamps\n");

  ur_result_t result =
      pfnGetGlobalTimestamps(hDevice, pDeviceTimestamp, pHostTimestamp);

  getContext()->notify_end(UR_FUNCTION_DEVICE_GET_GLOBAL_TIMESTAMPS,
                           "urDeviceGetGlobalTimestamps", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_DEVICE_GET_GLOBAL_TIMESTAMPS, &params);
    logger.info("   <--- urDeviceGetGlobalTimestamps({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreate
__urdlllocal ur_result_t UR_APICALL urContextCreate(
    /// [in] the number of devices given in phDevices
    uint32_t DeviceCount,
    /// [in][range(0, DeviceCount)] array of handle of devices.
    const ur_device_handle_t *phDevices,
    /// [in][optional] pointer to context creation properties.
    const ur_context_properties_t *pProperties,
    /// [out][alloc] pointer to handle of context object created
    ur_context_handle_t *phContext) {
  auto pfnCreate = getContext()->urDdiTable.Context.pfnCreate;

  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_context_create_params_t params = {&DeviceCount, &phDevices, &pProperties,
                                       &phContext};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_CONTEXT_CREATE,
                                                 "urContextCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urContextCreate\n");

  ur_result_t result =
      pfnCreate(DeviceCount, phDevices, pProperties, phContext);

  getContext()->notify_end(UR_FUNCTION_CONTEXT_CREATE, "urContextCreate",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_CONTEXT_CREATE,
                                    &params);
    logger.info("   <--- urContextCreate({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL urContextRetain(
    /// [in][retain] handle of the context to get a reference of.
    ur_context_handle_t hContext) {
  auto pfnRetain = getContext()->urDdiTable.Context.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_context_retain_params_t params = {&hContext};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_CONTEXT_RETAIN,
                                                 "urContextRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urContextRetain\n");

  ur_result_t result = pfnRetain(hContext);

  getContext()->notify_end(UR_FUNCTION_CONTEXT_RETAIN, "urContextRetain",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_CONTEXT_RETAIN,
                                    &params);
    logger.info("   <--- urContextRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL urContextRelease(
    /// [in][release] handle of the context to release.
    ur_context_handle_t hContext) {
  auto pfnRelease = getContext()->urDdiTable.Context.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_context_release_params_t params = {&hContext};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_CONTEXT_RELEASE,
                                                 "urContextRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urContextRelease\n");

  ur_result_t result = pfnRelease(hContext);

  getContext()->notify_end(UR_FUNCTION_CONTEXT_RELEASE, "urContextRelease",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_CONTEXT_RELEASE,
                                    &params);
    logger.info("   <--- urContextRelease({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetInfo
__urdlllocal ur_result_t UR_APICALL urContextGetInfo(
    /// [in] handle of the context
    ur_context_handle_t hContext,
    /// [in] type of the info to retrieve
    ur_context_info_t propName,
    /// [in] the number of bytes of memory pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info.
    /// if propSize is not equal to or greater than the real number of bytes
    /// needed to return
    /// the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName.
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Context.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_context_get_info_params_t params = {&hContext, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_CONTEXT_GET_INFO,
                                                 "urContextGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urContextGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hContext, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_CONTEXT_GET_INFO, "urContextGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_CONTEXT_GET_INFO,
                                    &params);
    logger.info("   <--- urContextGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextGetNativeHandle(
    /// [in] handle of the context.
    ur_context_handle_t hContext,
    /// [out] a pointer to the native handle of the context.
    ur_native_handle_t *phNativeContext) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Context.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_context_get_native_handle_params_t params = {&hContext, &phNativeContext};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_CONTEXT_GET_NATIVE_HANDLE,
                                 "urContextGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urContextGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hContext, phNativeContext);

  getContext()->notify_end(UR_FUNCTION_CONTEXT_GET_NATIVE_HANDLE,
                           "urContextGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_CONTEXT_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urContextGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the context.
    ur_native_handle_t hNativeContext,
    /// [in] handle of the adapter that owns the native handle
    ur_adapter_handle_t hAdapter,
    /// [in] number of devices associated with the context
    uint32_t numDevices,
    /// [in][optional][range(0, numDevices)] list of devices associated with
    /// the context
    const ur_device_handle_t *phDevices,
    /// [in][optional] pointer to native context properties struct
    const ur_context_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the context object created.
    ur_context_handle_t *phContext) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Context.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_context_create_with_native_handle_params_t params = {
      &hNativeContext, &hAdapter,    &numDevices,
      &phDevices,      &pProperties, &phContext};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_CONTEXT_CREATE_WITH_NATIVE_HANDLE,
                                 "urContextCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urContextCreateWithNativeHandle\n");

  ur_result_t result = pfnCreateWithNativeHandle(
      hNativeContext, hAdapter, numDevices, phDevices, pProperties, phContext);

  getContext()->notify_end(UR_FUNCTION_CONTEXT_CREATE_WITH_NATIVE_HANDLE,
                           "urContextCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_CONTEXT_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urContextCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextSetExtendedDeleter
__urdlllocal ur_result_t UR_APICALL urContextSetExtendedDeleter(
    /// [in] handle of the context.
    ur_context_handle_t hContext,
    /// [in] Function pointer to extended deleter.
    ur_context_extended_deleter_t pfnDeleter,
    /// [in][out][optional] pointer to data to be passed to callback.
    void *pUserData) {
  auto pfnSetExtendedDeleter =
      getContext()->urDdiTable.Context.pfnSetExtendedDeleter;

  if (nullptr == pfnSetExtendedDeleter)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_context_set_extended_deleter_params_t params = {&hContext, &pfnDeleter,
                                                     &pUserData};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_CONTEXT_SET_EXTENDED_DELETER,
                                 "urContextSetExtendedDeleter", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urContextSetExtendedDeleter\n");

  ur_result_t result = pfnSetExtendedDeleter(hContext, pfnDeleter, pUserData);

  getContext()->notify_end(UR_FUNCTION_CONTEXT_SET_EXTENDED_DELETER,
                           "urContextSetExtendedDeleter", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_CONTEXT_SET_EXTENDED_DELETER, &params);
    logger.info("   <--- urContextSetExtendedDeleter({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageCreate
__urdlllocal ur_result_t UR_APICALL urMemImageCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] allocation and usage information flags
    ur_mem_flags_t flags,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [in][optional] pointer to the buffer data
    void *pHost,
    /// [out][alloc] pointer to handle of image object created
    ur_mem_handle_t *phMem) {
  auto pfnImageCreate = getContext()->urDdiTable.Mem.pfnImageCreate;

  if (nullptr == pfnImageCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_image_create_params_t params = {&hContext,   &flags, &pImageFormat,
                                         &pImageDesc, &pHost, &phMem};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_MEM_IMAGE_CREATE,
                                                 "urMemImageCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemImageCreate\n");

  ur_result_t result =
      pfnImageCreate(hContext, flags, pImageFormat, pImageDesc, pHost, phMem);

  getContext()->notify_end(UR_FUNCTION_MEM_IMAGE_CREATE, "urMemImageCreate",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_IMAGE_CREATE,
                                    &params);
    logger.info("   <--- urMemImageCreate({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreate
__urdlllocal ur_result_t UR_APICALL urMemBufferCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] allocation and usage information flags
    ur_mem_flags_t flags,
    /// [in] size in bytes of the memory object to be allocated
    size_t size,
    /// [in][optional] pointer to buffer creation properties
    const ur_buffer_properties_t *pProperties,
    /// [out][alloc] pointer to handle of the memory buffer created
    ur_mem_handle_t *phBuffer) {
  auto pfnBufferCreate = getContext()->urDdiTable.Mem.pfnBufferCreate;

  if (nullptr == pfnBufferCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_buffer_create_params_t params = {&hContext, &flags, &size,
                                          &pProperties, &phBuffer};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_MEM_BUFFER_CREATE,
                                                 "urMemBufferCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemBufferCreate\n");

  ur_result_t result =
      pfnBufferCreate(hContext, flags, size, pProperties, phBuffer);

  getContext()->notify_end(UR_FUNCTION_MEM_BUFFER_CREATE, "urMemBufferCreate",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_BUFFER_CREATE,
                                    &params);
    logger.info("   <--- urMemBufferCreate({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL urMemRetain(
    /// [in][retain] handle of the memory object to get access
    ur_mem_handle_t hMem) {
  auto pfnRetain = getContext()->urDdiTable.Mem.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_retain_params_t params = {&hMem};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_MEM_RETAIN,
                                                 "urMemRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemRetain\n");

  ur_result_t result = pfnRetain(hMem);

  getContext()->notify_end(UR_FUNCTION_MEM_RETAIN, "urMemRetain", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_RETAIN, &params);
    logger.info("   <--- urMemRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL urMemRelease(
    /// [in][release] handle of the memory object to release
    ur_mem_handle_t hMem) {
  auto pfnRelease = getContext()->urDdiTable.Mem.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_release_params_t params = {&hMem};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_MEM_RELEASE,
                                                 "urMemRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemRelease\n");

  ur_result_t result = pfnRelease(hMem);

  getContext()->notify_end(UR_FUNCTION_MEM_RELEASE, "urMemRelease", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_RELEASE, &params);
    logger.info("   <--- urMemRelease({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferPartition
__urdlllocal ur_result_t UR_APICALL urMemBufferPartition(
    /// [in] handle of the buffer object to allocate from
    ur_mem_handle_t hBuffer,
    /// [in] allocation and usage information flags
    ur_mem_flags_t flags,
    /// [in] buffer creation type
    ur_buffer_create_type_t bufferCreateType,
    /// [in] pointer to buffer create region information
    const ur_buffer_region_t *pRegion,
    /// [out] pointer to the handle of sub buffer created
    ur_mem_handle_t *phMem) {
  auto pfnBufferPartition = getContext()->urDdiTable.Mem.pfnBufferPartition;

  if (nullptr == pfnBufferPartition)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_buffer_partition_params_t params = {
      &hBuffer, &flags, &bufferCreateType, &pRegion, &phMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_MEM_BUFFER_PARTITION, "urMemBufferPartition", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemBufferPartition\n");

  ur_result_t result =
      pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion, phMem);

  getContext()->notify_end(UR_FUNCTION_MEM_BUFFER_PARTITION,
                           "urMemBufferPartition", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_BUFFER_PARTITION,
                                    &params);
    logger.info("   <--- urMemBufferPartition({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemGetNativeHandle(
    /// [in] handle of the mem.
    ur_mem_handle_t hMem,
    /// [in][optional] handle of the device that the native handle will be
    /// resident on.
    ur_device_handle_t hDevice,
    /// [out] a pointer to the native handle of the mem.
    ur_native_handle_t *phNativeMem) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Mem.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_get_native_handle_params_t params = {&hMem, &hDevice, &phNativeMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_MEM_GET_NATIVE_HANDLE, "urMemGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hMem, hDevice, phNativeMem);

  getContext()->notify_end(UR_FUNCTION_MEM_GET_NATIVE_HANDLE,
                           "urMemGetNativeHandle", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_GET_NATIVE_HANDLE,
                                    &params);
    logger.info("   <--- urMemGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    /// [in][nocheck] the native handle to the memory.
    ur_native_handle_t hNativeMem,
    /// [in] handle of the context object.
    ur_context_handle_t hContext,
    /// [in][optional] pointer to native memory creation properties.
    const ur_mem_native_properties_t *pProperties,
    /// [out][alloc] pointer to handle of buffer memory object created.
    ur_mem_handle_t *phMem) {
  auto pfnBufferCreateWithNativeHandle =
      getContext()->urDdiTable.Mem.pfnBufferCreateWithNativeHandle;

  if (nullptr == pfnBufferCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_buffer_create_with_native_handle_params_t params = {
      &hNativeMem, &hContext, &pProperties, &phMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_MEM_BUFFER_CREATE_WITH_NATIVE_HANDLE,
      "urMemBufferCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemBufferCreateWithNativeHandle\n");

  ur_result_t result =
      pfnBufferCreateWithNativeHandle(hNativeMem, hContext, pProperties, phMem);

  getContext()->notify_end(UR_FUNCTION_MEM_BUFFER_CREATE_WITH_NATIVE_HANDLE,
                           "urMemBufferCreateWithNativeHandle", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_MEM_BUFFER_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urMemBufferCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    /// [in][nocheck] the native handle to the memory.
    ur_native_handle_t hNativeMem,
    /// [in] handle of the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to image format specification.
    const ur_image_format_t *pImageFormat,
    /// [in] pointer to image description.
    const ur_image_desc_t *pImageDesc,
    /// [in][optional] pointer to native memory creation properties.
    const ur_mem_native_properties_t *pProperties,
    /// [out][alloc pointer to handle of image memory object created.
    ur_mem_handle_t *phMem) {
  auto pfnImageCreateWithNativeHandle =
      getContext()->urDdiTable.Mem.pfnImageCreateWithNativeHandle;

  if (nullptr == pfnImageCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_image_create_with_native_handle_params_t params = {
      &hNativeMem, &hContext, &pImageFormat, &pImageDesc, &pProperties, &phMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_MEM_IMAGE_CREATE_WITH_NATIVE_HANDLE,
      "urMemImageCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemImageCreateWithNativeHandle\n");

  ur_result_t result = pfnImageCreateWithNativeHandle(
      hNativeMem, hContext, pImageFormat, pImageDesc, pProperties, phMem);

  getContext()->notify_end(UR_FUNCTION_MEM_IMAGE_CREATE_WITH_NATIVE_HANDLE,
                           "urMemImageCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_MEM_IMAGE_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urMemImageCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetInfo
__urdlllocal ur_result_t UR_APICALL urMemGetInfo(
    /// [in] handle to the memory object being queried.
    ur_mem_handle_t hMemory,
    /// [in] type of the info to retrieve.
    ur_mem_info_t propName,
    /// [in] the number of bytes of memory pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info.
    /// If propSize is less than the real number of bytes needed to return
    /// the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName.
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Mem.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_get_info_params_t params = {&hMemory, &propName, &propSize,
                                     &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_MEM_GET_INFO,
                                                 "urMemGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_MEM_GET_INFO, "urMemGetInfo", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_GET_INFO,
                                    &params);
    logger.info("   <--- urMemGetInfo({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemImageGetInfo
__urdlllocal ur_result_t UR_APICALL urMemImageGetInfo(
    /// [in] handle to the image object being queried.
    ur_mem_handle_t hMemory,
    /// [in] type of image info to retrieve.
    ur_image_info_t propName,
    /// [in] the number of bytes of memory pointer to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info.
    /// If propSize is less than the real number of bytes needed to return
    /// the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName.
    size_t *pPropSizeRet) {
  auto pfnImageGetInfo = getContext()->urDdiTable.Mem.pfnImageGetInfo;

  if (nullptr == pfnImageGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_mem_image_get_info_params_t params = {&hMemory, &propName, &propSize,
                                           &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_MEM_IMAGE_GET_INFO,
                                                 "urMemImageGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urMemImageGetInfo\n");

  ur_result_t result =
      pfnImageGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_MEM_IMAGE_GET_INFO, "urMemImageGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_MEM_IMAGE_GET_INFO,
                                    &params);
    logger.info("   <--- urMemImageGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreate
__urdlllocal ur_result_t UR_APICALL urSamplerCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to the sampler description
    const ur_sampler_desc_t *pDesc,
    /// [out][alloc] pointer to handle of sampler object created
    ur_sampler_handle_t *phSampler) {
  auto pfnCreate = getContext()->urDdiTable.Sampler.pfnCreate;

  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_sampler_create_params_t params = {&hContext, &pDesc, &phSampler};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_SAMPLER_CREATE,
                                                 "urSamplerCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urSamplerCreate\n");

  ur_result_t result = pfnCreate(hContext, pDesc, phSampler);

  getContext()->notify_end(UR_FUNCTION_SAMPLER_CREATE, "urSamplerCreate",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_SAMPLER_CREATE,
                                    &params);
    logger.info("   <--- urSamplerCreate({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRetain
__urdlllocal ur_result_t UR_APICALL urSamplerRetain(
    /// [in][retain] handle of the sampler object to get access
    ur_sampler_handle_t hSampler) {
  auto pfnRetain = getContext()->urDdiTable.Sampler.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_sampler_retain_params_t params = {&hSampler};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_SAMPLER_RETAIN,
                                                 "urSamplerRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urSamplerRetain\n");

  ur_result_t result = pfnRetain(hSampler);

  getContext()->notify_end(UR_FUNCTION_SAMPLER_RETAIN, "urSamplerRetain",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_SAMPLER_RETAIN,
                                    &params);
    logger.info("   <--- urSamplerRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRelease
__urdlllocal ur_result_t UR_APICALL urSamplerRelease(
    /// [in][release] handle of the sampler object to release
    ur_sampler_handle_t hSampler) {
  auto pfnRelease = getContext()->urDdiTable.Sampler.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_sampler_release_params_t params = {&hSampler};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_SAMPLER_RELEASE,
                                                 "urSamplerRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urSamplerRelease\n");

  ur_result_t result = pfnRelease(hSampler);

  getContext()->notify_end(UR_FUNCTION_SAMPLER_RELEASE, "urSamplerRelease",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_SAMPLER_RELEASE,
                                    &params);
    logger.info("   <--- urSamplerRelease({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetInfo
__urdlllocal ur_result_t UR_APICALL urSamplerGetInfo(
    /// [in] handle of the sampler object
    ur_sampler_handle_t hSampler,
    /// [in] name of the sampler property to query
    ur_sampler_info_t propName,
    /// [in] size in bytes of the sampler property value provided
    size_t propSize,
    /// [out][typename(propName, propSize)][optional] value of the sampler
    /// property
    void *pPropValue,
    /// [out][optional] size in bytes returned in sampler property value
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Sampler.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_sampler_get_info_params_t params = {&hSampler, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_SAMPLER_GET_INFO,
                                                 "urSamplerGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urSamplerGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hSampler, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_SAMPLER_GET_INFO, "urSamplerGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_SAMPLER_GET_INFO,
                                    &params);
    logger.info("   <--- urSamplerGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerGetNativeHandle(
    /// [in] handle of the sampler.
    ur_sampler_handle_t hSampler,
    /// [out] a pointer to the native handle of the sampler.
    ur_native_handle_t *phNativeSampler) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Sampler.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_sampler_get_native_handle_params_t params = {&hSampler, &phNativeSampler};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_SAMPLER_GET_NATIVE_HANDLE,
                                 "urSamplerGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urSamplerGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hSampler, phNativeSampler);

  getContext()->notify_end(UR_FUNCTION_SAMPLER_GET_NATIVE_HANDLE,
                           "urSamplerGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_SAMPLER_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urSamplerGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the sampler.
    ur_native_handle_t hNativeSampler,
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in][optional] pointer to native sampler properties struct.
    const ur_sampler_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the sampler object created.
    ur_sampler_handle_t *phSampler) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Sampler.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_sampler_create_with_native_handle_params_t params = {
      &hNativeSampler, &hContext, &pProperties, &phSampler};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_SAMPLER_CREATE_WITH_NATIVE_HANDLE,
                                 "urSamplerCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urSamplerCreateWithNativeHandle\n");

  ur_result_t result = pfnCreateWithNativeHandle(hNativeSampler, hContext,
                                                 pProperties, phSampler);

  getContext()->notify_end(UR_FUNCTION_SAMPLER_CREATE_WITH_NATIVE_HANDLE,
                           "urSamplerCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_SAMPLER_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urSamplerCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMHostAlloc
__urdlllocal ur_result_t UR_APICALL urUSMHostAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t pool,
    /// [in] minimum size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM host memory object
    void **ppMem) {
  auto pfnHostAlloc = getContext()->urDdiTable.USM.pfnHostAlloc;

  if (nullptr == pfnHostAlloc)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_host_alloc_params_t params = {&hContext, &pUSMDesc, &pool, &size,
                                       &ppMem};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_HOST_ALLOC,
                                                 "urUSMHostAlloc", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMHostAlloc\n");

  ur_result_t result = pfnHostAlloc(hContext, pUSMDesc, pool, size, ppMem);

  getContext()->notify_end(UR_FUNCTION_USM_HOST_ALLOC, "urUSMHostAlloc",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_HOST_ALLOC,
                                    &params);
    logger.info("   <--- urUSMHostAlloc({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMDeviceAlloc
__urdlllocal ur_result_t UR_APICALL urUSMDeviceAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t pool,
    /// [in] minimum size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM device memory object
    void **ppMem) {
  auto pfnDeviceAlloc = getContext()->urDdiTable.USM.pfnDeviceAlloc;

  if (nullptr == pfnDeviceAlloc)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_device_alloc_params_t params = {&hContext, &hDevice, &pUSMDesc,
                                         &pool,     &size,    &ppMem};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_DEVICE_ALLOC,
                                                 "urUSMDeviceAlloc", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMDeviceAlloc\n");

  ur_result_t result =
      pfnDeviceAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);

  getContext()->notify_end(UR_FUNCTION_USM_DEVICE_ALLOC, "urUSMDeviceAlloc",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_DEVICE_ALLOC,
                                    &params);
    logger.info("   <--- urUSMDeviceAlloc({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMSharedAlloc
__urdlllocal ur_result_t UR_APICALL urUSMSharedAlloc(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] Pointer to USM memory allocation descriptor.
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t pool,
    /// [in] minimum size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM shared memory object
    void **ppMem) {
  auto pfnSharedAlloc = getContext()->urDdiTable.USM.pfnSharedAlloc;

  if (nullptr == pfnSharedAlloc)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_shared_alloc_params_t params = {&hContext, &hDevice, &pUSMDesc,
                                         &pool,     &size,    &ppMem};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_SHARED_ALLOC,
                                                 "urUSMSharedAlloc", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMSharedAlloc\n");

  ur_result_t result =
      pfnSharedAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);

  getContext()->notify_end(UR_FUNCTION_USM_SHARED_ALLOC, "urUSMSharedAlloc",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_SHARED_ALLOC,
                                    &params);
    logger.info("   <--- urUSMSharedAlloc({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to USM memory object
    void *pMem) {
  auto pfnFree = getContext()->urDdiTable.USM.pfnFree;

  if (nullptr == pfnFree)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_free_params_t params = {&hContext, &pMem};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_USM_FREE, "urUSMFree", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMFree\n");

  ur_result_t result = pfnFree(hContext, pMem);

  getContext()->notify_end(UR_FUNCTION_USM_FREE, "urUSMFree", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_FREE, &params);
    logger.info("   <--- urUSMFree({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMGetMemAllocInfo
__urdlllocal ur_result_t UR_APICALL urUSMGetMemAllocInfo(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to USM memory object
    const void *pMem,
    /// [in] the name of the USM allocation property to query
    ur_usm_alloc_info_t propName,
    /// [in] size in bytes of the USM allocation property value
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] value of the USM
    /// allocation property
    void *pPropValue,
    /// [out][optional] bytes returned in USM allocation property
    size_t *pPropSizeRet) {
  auto pfnGetMemAllocInfo = getContext()->urDdiTable.USM.pfnGetMemAllocInfo;

  if (nullptr == pfnGetMemAllocInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_get_mem_alloc_info_params_t params = {
      &hContext, &pMem, &propName, &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_USM_GET_MEM_ALLOC_INFO, "urUSMGetMemAllocInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMGetMemAllocInfo\n");

  ur_result_t result = pfnGetMemAllocInfo(hContext, pMem, propName, propSize,
                                          pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_USM_GET_MEM_ALLOC_INFO,
                           "urUSMGetMemAllocInfo", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_GET_MEM_ALLOC_INFO, &params);
    logger.info("   <--- urUSMGetMemAllocInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolCreate
__urdlllocal ur_result_t UR_APICALL urUSMPoolCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to USM pool descriptor. Can be chained with
    /// ::ur_usm_pool_limits_desc_t
    ur_usm_pool_desc_t *pPoolDesc,
    /// [out][alloc] pointer to USM memory pool
    ur_usm_pool_handle_t *ppPool) {
  auto pfnPoolCreate = getContext()->urDdiTable.USM.pfnPoolCreate;

  if (nullptr == pfnPoolCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_create_params_t params = {&hContext, &pPoolDesc, &ppPool};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_POOL_CREATE,
                                                 "urUSMPoolCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolCreate\n");

  ur_result_t result = pfnPoolCreate(hContext, pPoolDesc, ppPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_CREATE, "urUSMPoolCreate",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_CREATE,
                                    &params);
    logger.info("   <--- urUSMPoolCreate({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRetain
__urdlllocal ur_result_t UR_APICALL urUSMPoolRetain(
    /// [in][retain] pointer to USM memory pool
    ur_usm_pool_handle_t pPool) {
  auto pfnPoolRetain = getContext()->urDdiTable.USM.pfnPoolRetain;

  if (nullptr == pfnPoolRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_retain_params_t params = {&pPool};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_POOL_RETAIN,
                                                 "urUSMPoolRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolRetain\n");

  ur_result_t result = pfnPoolRetain(pPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_RETAIN, "urUSMPoolRetain",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_RETAIN,
                                    &params);
    logger.info("   <--- urUSMPoolRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRelease
__urdlllocal ur_result_t UR_APICALL urUSMPoolRelease(
    /// [in][release] pointer to USM memory pool
    ur_usm_pool_handle_t pPool) {
  auto pfnPoolRelease = getContext()->urDdiTable.USM.pfnPoolRelease;

  if (nullptr == pfnPoolRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_release_params_t params = {&pPool};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_POOL_RELEASE,
                                                 "urUSMPoolRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolRelease\n");

  ur_result_t result = pfnPoolRelease(pPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_RELEASE, "urUSMPoolRelease",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_RELEASE,
                                    &params);
    logger.info("   <--- urUSMPoolRelease({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolGetInfo
__urdlllocal ur_result_t UR_APICALL urUSMPoolGetInfo(
    /// [in] handle of the USM memory pool
    ur_usm_pool_handle_t hPool,
    /// [in] name of the pool property to query
    ur_usm_pool_info_t propName,
    /// [in] size in bytes of the pool property value provided
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] value of the pool
    /// property
    void *pPropValue,
    /// [out][optional] size in bytes returned in pool property value
    size_t *pPropSizeRet) {
  auto pfnPoolGetInfo = getContext()->urDdiTable.USM.pfnPoolGetInfo;

  if (nullptr == pfnPoolGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_get_info_params_t params = {&hPool, &propName, &propSize,
                                          &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_POOL_GET_INFO,
                                                 "urUSMPoolGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolGetInfo\n");

  ur_result_t result =
      pfnPoolGetInfo(hPool, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_GET_INFO, "urUSMPoolGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_GET_INFO,
                                    &params);
    logger.info("   <--- urUSMPoolGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemGranularityGetInfo
__urdlllocal ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    /// [in] handle of the context object.
    ur_context_handle_t hContext,
    /// [in][optional] is the device to get the granularity from, if the
    /// device is null then the granularity is suitable for all devices in
    /// context.
    ur_device_handle_t hDevice,
    /// [in] type of the info to query.
    ur_virtual_mem_granularity_info_t propName,
    /// [in] size in bytes of the memory pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info. If propSize is less than the real number of bytes needed to
    /// return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    /// returned and pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName."
    size_t *pPropSizeRet) {
  auto pfnGranularityGetInfo =
      getContext()->urDdiTable.VirtualMem.pfnGranularityGetInfo;

  if (nullptr == pfnGranularityGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_virtual_mem_granularity_get_info_params_t params = {
      &hContext, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_VIRTUAL_MEM_GRANULARITY_GET_INFO,
                                 "urVirtualMemGranularityGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urVirtualMemGranularityGetInfo\n");

  ur_result_t result = pfnGranularityGetInfo(
      hContext, hDevice, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_VIRTUAL_MEM_GRANULARITY_GET_INFO,
                           "urVirtualMemGranularityGetInfo", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_VIRTUAL_MEM_GRANULARITY_GET_INFO, &params);
    logger.info("   <--- urVirtualMemGranularityGetInfo({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemReserve
__urdlllocal ur_result_t UR_APICALL urVirtualMemReserve(
    /// [in] handle of the context object.
    ur_context_handle_t hContext,
    /// [in][optional] pointer to the start of the virtual memory region to
    /// reserve, specifying a null value causes the implementation to select a
    /// start address.
    const void *pStart,
    /// [in] size in bytes of the virtual address range to reserve.
    size_t size,
    /// [out] pointer to the returned address at the start of reserved virtual
    /// memory range.
    void **ppStart) {
  auto pfnReserve = getContext()->urDdiTable.VirtualMem.pfnReserve;

  if (nullptr == pfnReserve)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_virtual_mem_reserve_params_t params = {&hContext, &pStart, &size,
                                            &ppStart};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_VIRTUAL_MEM_RESERVE, "urVirtualMemReserve", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urVirtualMemReserve\n");

  ur_result_t result = pfnReserve(hContext, pStart, size, ppStart);

  getContext()->notify_end(UR_FUNCTION_VIRTUAL_MEM_RESERVE,
                           "urVirtualMemReserve", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_VIRTUAL_MEM_RESERVE,
                                    &params);
    logger.info("   <--- urVirtualMemReserve({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemFree
__urdlllocal ur_result_t UR_APICALL urVirtualMemFree(
    /// [in] handle of the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to the start of the virtual memory range to free.
    const void *pStart,
    /// [in] size in bytes of the virtual memory range to free.
    size_t size) {
  auto pfnFree = getContext()->urDdiTable.VirtualMem.pfnFree;

  if (nullptr == pfnFree)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_virtual_mem_free_params_t params = {&hContext, &pStart, &size};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_VIRTUAL_MEM_FREE,
                                                 "urVirtualMemFree", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urVirtualMemFree\n");

  ur_result_t result = pfnFree(hContext, pStart, size);

  getContext()->notify_end(UR_FUNCTION_VIRTUAL_MEM_FREE, "urVirtualMemFree",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_VIRTUAL_MEM_FREE,
                                    &params);
    logger.info("   <--- urVirtualMemFree({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemMap
__urdlllocal ur_result_t UR_APICALL urVirtualMemMap(
    /// [in] handle to the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to the start of the virtual memory range.
    const void *pStart,
    /// [in] size in bytes of the virtual memory range to map.
    size_t size,
    /// [in] handle of the physical memory to map pStart to.
    ur_physical_mem_handle_t hPhysicalMem,
    /// [in] offset in bytes into the physical memory to map pStart to.
    size_t offset,
    /// [in] access flags for the physical memory mapping.
    ur_virtual_mem_access_flags_t flags) {
  auto pfnMap = getContext()->urDdiTable.VirtualMem.pfnMap;

  if (nullptr == pfnMap)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_virtual_mem_map_params_t params = {&hContext,     &pStart, &size,
                                        &hPhysicalMem, &offset, &flags};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_VIRTUAL_MEM_MAP,
                                                 "urVirtualMemMap", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urVirtualMemMap\n");

  ur_result_t result =
      pfnMap(hContext, pStart, size, hPhysicalMem, offset, flags);

  getContext()->notify_end(UR_FUNCTION_VIRTUAL_MEM_MAP, "urVirtualMemMap",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_VIRTUAL_MEM_MAP,
                                    &params);
    logger.info("   <--- urVirtualMemMap({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemUnmap
__urdlllocal ur_result_t UR_APICALL urVirtualMemUnmap(
    /// [in] handle to the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to the start of the mapped virtual memory range
    const void *pStart,
    /// [in] size in bytes of the virtual memory range.
    size_t size) {
  auto pfnUnmap = getContext()->urDdiTable.VirtualMem.pfnUnmap;

  if (nullptr == pfnUnmap)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_virtual_mem_unmap_params_t params = {&hContext, &pStart, &size};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_VIRTUAL_MEM_UNMAP,
                                                 "urVirtualMemUnmap", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urVirtualMemUnmap\n");

  ur_result_t result = pfnUnmap(hContext, pStart, size);

  getContext()->notify_end(UR_FUNCTION_VIRTUAL_MEM_UNMAP, "urVirtualMemUnmap",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_VIRTUAL_MEM_UNMAP,
                                    &params);
    logger.info("   <--- urVirtualMemUnmap({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemSetAccess
__urdlllocal ur_result_t UR_APICALL urVirtualMemSetAccess(
    /// [in] handle to the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to the start of the virtual memory range.
    const void *pStart,
    /// [in] size in bytes of the virtual memory range.
    size_t size,
    /// [in] access flags to set for the mapped virtual memory range.
    ur_virtual_mem_access_flags_t flags) {
  auto pfnSetAccess = getContext()->urDdiTable.VirtualMem.pfnSetAccess;

  if (nullptr == pfnSetAccess)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_virtual_mem_set_access_params_t params = {&hContext, &pStart, &size,
                                               &flags};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_VIRTUAL_MEM_SET_ACCESS, "urVirtualMemSetAccess", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urVirtualMemSetAccess\n");

  ur_result_t result = pfnSetAccess(hContext, pStart, size, flags);

  getContext()->notify_end(UR_FUNCTION_VIRTUAL_MEM_SET_ACCESS,
                           "urVirtualMemSetAccess", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_VIRTUAL_MEM_SET_ACCESS, &params);
    logger.info("   <--- urVirtualMemSetAccess({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemGetInfo
__urdlllocal ur_result_t UR_APICALL urVirtualMemGetInfo(
    /// [in] handle to the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to the start of the virtual memory range.
    const void *pStart,
    /// [in] size in bytes of the virtual memory range.
    size_t size,
    /// [in] type of the info to query.
    ur_virtual_mem_info_t propName,
    /// [in] size in bytes of the memory pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info. If propSize is less than the real number of bytes needed to
    /// return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    /// returned and pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName."
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.VirtualMem.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_virtual_mem_get_info_params_t params = {
      &hContext, &pStart,     &size,        &propName,
      &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_VIRTUAL_MEM_GET_INFO, "urVirtualMemGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urVirtualMemGetInfo\n");

  ur_result_t result = pfnGetInfo(hContext, pStart, size, propName, propSize,
                                  pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_VIRTUAL_MEM_GET_INFO,
                           "urVirtualMemGetInfo", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_VIRTUAL_MEM_GET_INFO,
                                    &params);
    logger.info("   <--- urVirtualMemGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemCreate
__urdlllocal ur_result_t UR_APICALL urPhysicalMemCreate(
    /// [in] handle of the context object.
    ur_context_handle_t hContext,
    /// [in] handle of the device object.
    ur_device_handle_t hDevice,
    /// [in] size in bytes of physical memory to allocate, must be a multiple
    /// of ::UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM.
    size_t size,
    /// [in][optional] pointer to physical memory creation properties.
    const ur_physical_mem_properties_t *pProperties,
    /// [out][alloc] pointer to handle of physical memory object created.
    ur_physical_mem_handle_t *phPhysicalMem) {
  auto pfnCreate = getContext()->urDdiTable.PhysicalMem.pfnCreate;

  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_physical_mem_create_params_t params = {&hContext, &hDevice, &size,
                                            &pProperties, &phPhysicalMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PHYSICAL_MEM_CREATE, "urPhysicalMemCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPhysicalMemCreate\n");

  ur_result_t result =
      pfnCreate(hContext, hDevice, size, pProperties, phPhysicalMem);

  getContext()->notify_end(UR_FUNCTION_PHYSICAL_MEM_CREATE,
                           "urPhysicalMemCreate", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PHYSICAL_MEM_CREATE,
                                    &params);
    logger.info("   <--- urPhysicalMemCreate({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRetain
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRetain(
    /// [in][retain] handle of the physical memory object to retain.
    ur_physical_mem_handle_t hPhysicalMem) {
  auto pfnRetain = getContext()->urDdiTable.PhysicalMem.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_physical_mem_retain_params_t params = {&hPhysicalMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PHYSICAL_MEM_RETAIN, "urPhysicalMemRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPhysicalMemRetain\n");

  ur_result_t result = pfnRetain(hPhysicalMem);

  getContext()->notify_end(UR_FUNCTION_PHYSICAL_MEM_RETAIN,
                           "urPhysicalMemRetain", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PHYSICAL_MEM_RETAIN,
                                    &params);
    logger.info("   <--- urPhysicalMemRetain({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRelease
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRelease(
    /// [in][release] handle of the physical memory object to release.
    ur_physical_mem_handle_t hPhysicalMem) {
  auto pfnRelease = getContext()->urDdiTable.PhysicalMem.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_physical_mem_release_params_t params = {&hPhysicalMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PHYSICAL_MEM_RELEASE, "urPhysicalMemRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPhysicalMemRelease\n");

  ur_result_t result = pfnRelease(hPhysicalMem);

  getContext()->notify_end(UR_FUNCTION_PHYSICAL_MEM_RELEASE,
                           "urPhysicalMemRelease", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PHYSICAL_MEM_RELEASE,
                                    &params);
    logger.info("   <--- urPhysicalMemRelease({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemGetInfo
__urdlllocal ur_result_t UR_APICALL urPhysicalMemGetInfo(
    /// [in] handle of the physical memory object to query.
    ur_physical_mem_handle_t hPhysicalMem,
    /// [in] type of the info to query.
    ur_physical_mem_info_t propName,
    /// [in] size in bytes of the memory pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info. If propSize is less than the real number of bytes needed to
    /// return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    /// returned and pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName."
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.PhysicalMem.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_physical_mem_get_info_params_t params = {
      &hPhysicalMem, &propName, &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PHYSICAL_MEM_GET_INFO, "urPhysicalMemGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urPhysicalMemGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hPhysicalMem, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_PHYSICAL_MEM_GET_INFO,
                           "urPhysicalMemGetInfo", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PHYSICAL_MEM_GET_INFO,
                                    &params);
    logger.info("   <--- urPhysicalMemGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithIL
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithIL(
    /// [in] handle of the context instance
    ur_context_handle_t hContext,
    /// [in] pointer to IL binary.
    const void *pIL,
    /// [in] length of `pIL` in bytes.
    size_t length,
    /// [in][optional] pointer to program creation properties.
    const ur_program_properties_t *pProperties,
    /// [out][alloc] pointer to handle of program object created.
    ur_program_handle_t *phProgram) {
  auto pfnCreateWithIL = getContext()->urDdiTable.Program.pfnCreateWithIL;

  if (nullptr == pfnCreateWithIL)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_create_with_il_params_t params = {&hContext, &pIL, &length,
                                               &pProperties, &phProgram};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PROGRAM_CREATE_WITH_IL, "urProgramCreateWithIL", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramCreateWithIL\n");

  ur_result_t result =
      pfnCreateWithIL(hContext, pIL, length, pProperties, phProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_CREATE_WITH_IL,
                           "urProgramCreateWithIL", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_CREATE_WITH_IL, &params);
    logger.info("   <--- urProgramCreateWithIL({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithBinary
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithBinary(
    /// [in] handle of the context instance
    ur_context_handle_t hContext,
    /// [in] number of devices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] a pointer to a list of device handles. The
    /// binaries are loaded for devices specified in this list.
    ur_device_handle_t *phDevices,
    /// [in][range(0, numDevices)] array of sizes of program binaries
    /// specified by `pBinaries` (in bytes).
    size_t *pLengths,
    /// [in][range(0, numDevices)] pointer to program binaries to be loaded
    /// for devices specified by `phDevices`.
    const uint8_t **ppBinaries,
    /// [in][optional] pointer to program creation properties.
    const ur_program_properties_t *pProperties,
    /// [out][alloc] pointer to handle of Program object created.
    ur_program_handle_t *phProgram) {
  auto pfnCreateWithBinary =
      getContext()->urDdiTable.Program.pfnCreateWithBinary;

  if (nullptr == pfnCreateWithBinary)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_create_with_binary_params_t params = {
      &hContext,   &numDevices,  &phDevices, &pLengths,
      &ppBinaries, &pProperties, &phProgram};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_PROGRAM_CREATE_WITH_BINARY,
                                 "urProgramCreateWithBinary", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramCreateWithBinary\n");

  ur_result_t result =
      pfnCreateWithBinary(hContext, numDevices, phDevices, pLengths, ppBinaries,
                          pProperties, phProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_CREATE_WITH_BINARY,
                           "urProgramCreateWithBinary", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_CREATE_WITH_BINARY, &params);
    logger.info("   <--- urProgramCreateWithBinary({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuild
__urdlllocal ur_result_t UR_APICALL urProgramBuild(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in] Handle of the program to build.
    ur_program_handle_t hProgram,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) {
  auto pfnBuild = getContext()->urDdiTable.Program.pfnBuild;

  if (nullptr == pfnBuild)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_build_params_t params = {&hContext, &hProgram, &pOptions};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_BUILD,
                                                 "urProgramBuild", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramBuild\n");

  ur_result_t result = pfnBuild(hContext, hProgram, pOptions);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_BUILD, "urProgramBuild", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_BUILD,
                                    &params);
    logger.info("   <--- urProgramBuild({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCompile
__urdlllocal ur_result_t UR_APICALL urProgramCompile(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in][out] handle of the program to compile.
    ur_program_handle_t hProgram,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) {
  auto pfnCompile = getContext()->urDdiTable.Program.pfnCompile;

  if (nullptr == pfnCompile)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_compile_params_t params = {&hContext, &hProgram, &pOptions};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_COMPILE,
                                                 "urProgramCompile", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramCompile\n");

  ur_result_t result = pfnCompile(hContext, hProgram, pOptions);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_COMPILE, "urProgramCompile",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_COMPILE,
                                    &params);
    logger.info("   <--- urProgramCompile({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLink
__urdlllocal ur_result_t UR_APICALL urProgramLink(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in] number of program handles in `phPrograms`.
    uint32_t count,
    /// [in][range(0, count)] pointer to array of program handles.
    const ur_program_handle_t *phPrograms,
    /// [in][optional] pointer to linker options null-terminated string.
    const char *pOptions,
    /// [out][alloc] pointer to handle of program object created.
    ur_program_handle_t *phProgram) {
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }
  auto pfnLink = getContext()->urDdiTable.Program.pfnLink;

  if (nullptr == pfnLink)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_link_params_t params = {&hContext, &count, &phPrograms, &pOptions,
                                     &phProgram};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_LINK,
                                                 "urProgramLink", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramLink\n");

  ur_result_t result =
      pfnLink(hContext, count, phPrograms, pOptions, phProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_LINK, "urProgramLink", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_LINK,
                                    &params);
    logger.info("   <--- urProgramLink({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL urProgramRetain(
    /// [in][retain] handle for the Program to retain
    ur_program_handle_t hProgram) {
  auto pfnRetain = getContext()->urDdiTable.Program.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_retain_params_t params = {&hProgram};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_RETAIN,
                                                 "urProgramRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramRetain\n");

  ur_result_t result = pfnRetain(hProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_RETAIN, "urProgramRetain",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_RETAIN,
                                    &params);
    logger.info("   <--- urProgramRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
__urdlllocal ur_result_t UR_APICALL urProgramRelease(
    /// [in][release] handle for the Program to release
    ur_program_handle_t hProgram) {
  auto pfnRelease = getContext()->urDdiTable.Program.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_release_params_t params = {&hProgram};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_RELEASE,
                                                 "urProgramRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramRelease\n");

  ur_result_t result = pfnRelease(hProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_RELEASE, "urProgramRelease",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_RELEASE,
                                    &params);
    logger.info("   <--- urProgramRelease({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetFunctionPointer
__urdlllocal ur_result_t UR_APICALL urProgramGetFunctionPointer(
    /// [in] handle of the device to retrieve pointer for.
    ur_device_handle_t hDevice,
    /// [in] handle of the program to search for function in.
    /// The program must already be built to the specified device, or
    /// otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    ur_program_handle_t hProgram,
    /// [in] A null-terminates string denoting the mangled function name.
    const char *pFunctionName,
    /// [out] Returns the pointer to the function if it is found in the program.
    void **ppFunctionPointer) {
  auto pfnGetFunctionPointer =
      getContext()->urDdiTable.Program.pfnGetFunctionPointer;

  if (nullptr == pfnGetFunctionPointer)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_get_function_pointer_params_t params = {
      &hDevice, &hProgram, &pFunctionName, &ppFunctionPointer};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_PROGRAM_GET_FUNCTION_POINTER,
                                 "urProgramGetFunctionPointer", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramGetFunctionPointer\n");

  ur_result_t result = pfnGetFunctionPointer(hDevice, hProgram, pFunctionName,
                                             ppFunctionPointer);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_GET_FUNCTION_POINTER,
                           "urProgramGetFunctionPointer", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_GET_FUNCTION_POINTER, &params);
    logger.info("   <--- urProgramGetFunctionPointer({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetGlobalVariablePointer
__urdlllocal ur_result_t UR_APICALL urProgramGetGlobalVariablePointer(
    /// [in] handle of the device to retrieve the pointer for.
    ur_device_handle_t hDevice,
    /// [in] handle of the program where the global variable is.
    ur_program_handle_t hProgram,
    /// [in] mangled name of the global variable to retrieve the pointer for.
    const char *pGlobalVariableName,
    /// [out][optional] Returns the size of the global variable if it is found
    /// in the program.
    size_t *pGlobalVariableSizeRet,
    /// [out] Returns the pointer to the global variable if it is found in the
    /// program.
    void **ppGlobalVariablePointerRet) {
  auto pfnGetGlobalVariablePointer =
      getContext()->urDdiTable.Program.pfnGetGlobalVariablePointer;

  if (nullptr == pfnGetGlobalVariablePointer)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_get_global_variable_pointer_params_t params = {
      &hDevice, &hProgram, &pGlobalVariableName, &pGlobalVariableSizeRet,
      &ppGlobalVariablePointerRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PROGRAM_GET_GLOBAL_VARIABLE_POINTER,
      "urProgramGetGlobalVariablePointer", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramGetGlobalVariablePointer\n");

  ur_result_t result = pfnGetGlobalVariablePointer(
      hDevice, hProgram, pGlobalVariableName, pGlobalVariableSizeRet,
      ppGlobalVariablePointerRet);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_GET_GLOBAL_VARIABLE_POINTER,
                           "urProgramGetGlobalVariablePointer", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_GET_GLOBAL_VARIABLE_POINTER, &params);
    logger.info("   <--- urProgramGetGlobalVariablePointer({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetInfo
__urdlllocal ur_result_t UR_APICALL urProgramGetInfo(
    /// [in] handle of the Program object
    ur_program_handle_t hProgram,
    /// [in] name of the Program property to query
    ur_program_info_t propName,
    /// [in] the size of the Program property.
    size_t propSize,
    /// [in,out][optional][typename(propName, propSize)] array of bytes of
    /// holding the program info property.
    /// If propSize is not equal to or greater than the real number of bytes
    /// needed to return
    /// the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName.
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Program.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_get_info_params_t params = {&hProgram, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_GET_INFO,
                                                 "urProgramGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hProgram, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_GET_INFO, "urProgramGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_GET_INFO,
                                    &params);
    logger.info("   <--- urProgramGetInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetBuildInfo
__urdlllocal ur_result_t UR_APICALL urProgramGetBuildInfo(
    /// [in] handle of the Program object
    ur_program_handle_t hProgram,
    /// [in] handle of the Device object
    ur_device_handle_t hDevice,
    /// [in] name of the Program build info to query
    ur_program_build_info_t propName,
    /// [in] size of the Program build info property.
    size_t propSize,
    /// [in,out][optional][typename(propName, propSize)] value of the Program
    /// build property.
    /// If propSize is not equal to or greater than the real number of bytes
    /// needed to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE
    /// error is returned and pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of data being
    /// queried by propName.
    size_t *pPropSizeRet) {
  auto pfnGetBuildInfo = getContext()->urDdiTable.Program.pfnGetBuildInfo;

  if (nullptr == pfnGetBuildInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_get_build_info_params_t params = {
      &hProgram, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PROGRAM_GET_BUILD_INFO, "urProgramGetBuildInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramGetBuildInfo\n");

  ur_result_t result = pfnGetBuildInfo(hProgram, hDevice, propName, propSize,
                                       pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_GET_BUILD_INFO,
                           "urProgramGetBuildInfo", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_GET_BUILD_INFO, &params);
    logger.info("   <--- urProgramGetBuildInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    /// [in] handle of the Program object
    ur_program_handle_t hProgram,
    /// [in] the number of elements in the pSpecConstants array
    uint32_t count,
    /// [in][range(0, count)] array of specialization constant value
    /// descriptions
    const ur_specialization_constant_info_t *pSpecConstants) {
  auto pfnSetSpecializationConstants =
      getContext()->urDdiTable.Program.pfnSetSpecializationConstants;

  if (nullptr == pfnSetSpecializationConstants)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_set_specialization_constants_params_t params = {&hProgram, &count,
                                                             &pSpecConstants};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PROGRAM_SET_SPECIALIZATION_CONSTANTS,
      "urProgramSetSpecializationConstants", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramSetSpecializationConstants\n");

  ur_result_t result =
      pfnSetSpecializationConstants(hProgram, count, pSpecConstants);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_SET_SPECIALIZATION_CONSTANTS,
                           "urProgramSetSpecializationConstants", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_SET_SPECIALIZATION_CONSTANTS, &params);
    logger.info("   <--- urProgramSetSpecializationConstants({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramGetNativeHandle(
    /// [in] handle of the program.
    ur_program_handle_t hProgram,
    /// [out] a pointer to the native handle of the program.
    ur_native_handle_t *phNativeProgram) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Program.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_get_native_handle_params_t params = {&hProgram, &phNativeProgram};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_PROGRAM_GET_NATIVE_HANDLE,
                                 "urProgramGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hProgram, phNativeProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_GET_NATIVE_HANDLE,
                           "urProgramGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urProgramGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the program.
    ur_native_handle_t hNativeProgram,
    /// [in] handle of the context instance
    ur_context_handle_t hContext,
    /// [in][optional] pointer to native program properties struct.
    const ur_program_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the program object created.
    ur_program_handle_t *phProgram) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Program.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_create_with_native_handle_params_t params = {
      &hNativeProgram, &hContext, &pProperties, &phProgram};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_PROGRAM_CREATE_WITH_NATIVE_HANDLE,
                                 "urProgramCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramCreateWithNativeHandle\n");

  ur_result_t result = pfnCreateWithNativeHandle(hNativeProgram, hContext,
                                                 pProperties, phProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_CREATE_WITH_NATIVE_HANDLE,
                           "urProgramCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_PROGRAM_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urProgramCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
__urdlllocal ur_result_t UR_APICALL urKernelCreate(
    /// [in] handle of the program instance
    ur_program_handle_t hProgram,
    /// [in] pointer to null-terminated string.
    const char *pKernelName,
    /// [out][alloc] pointer to handle of kernel object created.
    ur_kernel_handle_t *phKernel) {
  auto pfnCreate = getContext()->urDdiTable.Kernel.pfnCreate;

  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_create_params_t params = {&hProgram, &pKernelName, &phKernel};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_KERNEL_CREATE,
                                                 "urKernelCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelCreate\n");

  ur_result_t result = pfnCreate(hProgram, pKernelName, phKernel);

  getContext()->notify_end(UR_FUNCTION_KERNEL_CREATE, "urKernelCreate", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_CREATE,
                                    &params);
    logger.info("   <--- urKernelCreate({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgValue
__urdlllocal ur_result_t UR_APICALL urKernelSetArgValue(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t argIndex,
    /// [in] size of argument type
    size_t argSize,
    /// [in][optional] pointer to value properties.
    const ur_kernel_arg_value_properties_t *pProperties,
    /// [in] argument value represented as matching arg type.
    /// The data pointed to will be copied and therefore can be reused on
    /// return.
    const void *pArgValue) {
  auto pfnSetArgValue = getContext()->urDdiTable.Kernel.pfnSetArgValue;

  if (nullptr == pfnSetArgValue)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_set_arg_value_params_t params = {&hKernel, &argIndex, &argSize,
                                             &pProperties, &pArgValue};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SET_ARG_VALUE, "urKernelSetArgValue", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSetArgValue\n");

  ur_result_t result =
      pfnSetArgValue(hKernel, argIndex, argSize, pProperties, pArgValue);

  getContext()->notify_end(UR_FUNCTION_KERNEL_SET_ARG_VALUE,
                           "urKernelSetArgValue", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_SET_ARG_VALUE,
                                    &params);
    logger.info("   <--- urKernelSetArgValue({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgLocal
__urdlllocal ur_result_t UR_APICALL urKernelSetArgLocal(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t argIndex,
    /// [in] size of the local buffer to be allocated by the runtime
    size_t argSize,
    /// [in][optional] pointer to local buffer properties.
    const ur_kernel_arg_local_properties_t *pProperties) {
  auto pfnSetArgLocal = getContext()->urDdiTable.Kernel.pfnSetArgLocal;

  if (nullptr == pfnSetArgLocal)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_set_arg_local_params_t params = {&hKernel, &argIndex, &argSize,
                                             &pProperties};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SET_ARG_LOCAL, "urKernelSetArgLocal", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSetArgLocal\n");

  ur_result_t result = pfnSetArgLocal(hKernel, argIndex, argSize, pProperties);

  getContext()->notify_end(UR_FUNCTION_KERNEL_SET_ARG_LOCAL,
                           "urKernelSetArgLocal", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_SET_ARG_LOCAL,
                                    &params);
    logger.info("   <--- urKernelSetArgLocal({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetInfo
__urdlllocal ur_result_t UR_APICALL urKernelGetInfo(
    /// [in] handle of the Kernel object
    ur_kernel_handle_t hKernel,
    /// [in] name of the Kernel property to query
    ur_kernel_info_t propName,
    /// [in] the size of the Kernel property value.
    size_t propSize,
    /// [in,out][optional][typename(propName, propSize)] array of bytes
    /// holding the kernel info property.
    /// If propSize is not equal to or greater than the real number of bytes
    /// needed to return
    /// the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of data being
    /// queried by propName.
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Kernel.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_get_info_params_t params = {&hKernel, &propName, &propSize,
                                        &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_KERNEL_GET_INFO,
                                                 "urKernelGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hKernel, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_KERNEL_GET_INFO, "urKernelGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_GET_INFO,
                                    &params);
    logger.info("   <--- urKernelGetInfo({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetGroupInfo
__urdlllocal ur_result_t UR_APICALL urKernelGetGroupInfo(
    /// [in] handle of the Kernel object
    ur_kernel_handle_t hKernel,
    /// [in] handle of the Device object
    ur_device_handle_t hDevice,
    /// [in] name of the work Group property to query
    ur_kernel_group_info_t propName,
    /// [in] size of the Kernel Work Group property value
    size_t propSize,
    /// [in,out][optional][typename(propName, propSize)] value of the Kernel
    /// Work Group property.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of data being
    /// queried by propName.
    size_t *pPropSizeRet) {
  auto pfnGetGroupInfo = getContext()->urDdiTable.Kernel.pfnGetGroupInfo;

  if (nullptr == pfnGetGroupInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_get_group_info_params_t params = {
      &hKernel, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_GET_GROUP_INFO, "urKernelGetGroupInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelGetGroupInfo\n");

  ur_result_t result = pfnGetGroupInfo(hKernel, hDevice, propName, propSize,
                                       pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_KERNEL_GET_GROUP_INFO,
                           "urKernelGetGroupInfo", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_GET_GROUP_INFO,
                                    &params);
    logger.info("   <--- urKernelGetGroupInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetSubGroupInfo
__urdlllocal ur_result_t UR_APICALL urKernelGetSubGroupInfo(
    /// [in] handle of the Kernel object
    ur_kernel_handle_t hKernel,
    /// [in] handle of the Device object
    ur_device_handle_t hDevice,
    /// [in] name of the SubGroup property to query
    ur_kernel_sub_group_info_t propName,
    /// [in] size of the Kernel SubGroup property value
    size_t propSize,
    /// [in,out][optional][typename(propName, propSize)] value of the Kernel
    /// SubGroup property.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of data being
    /// queried by propName.
    size_t *pPropSizeRet) {
  auto pfnGetSubGroupInfo = getContext()->urDdiTable.Kernel.pfnGetSubGroupInfo;

  if (nullptr == pfnGetSubGroupInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_get_sub_group_info_params_t params = {
      &hKernel, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_KERNEL_GET_SUB_GROUP_INFO,
                                 "urKernelGetSubGroupInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelGetSubGroupInfo\n");

  ur_result_t result = pfnGetSubGroupInfo(hKernel, hDevice, propName, propSize,
                                          pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_KERNEL_GET_SUB_GROUP_INFO,
                           "urKernelGetSubGroupInfo", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_GET_SUB_GROUP_INFO, &params);
    logger.info("   <--- urKernelGetSubGroupInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL urKernelRetain(
    /// [in][retain] handle for the Kernel to retain
    ur_kernel_handle_t hKernel) {
  auto pfnRetain = getContext()->urDdiTable.Kernel.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_retain_params_t params = {&hKernel};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_KERNEL_RETAIN,
                                                 "urKernelRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelRetain\n");

  ur_result_t result = pfnRetain(hKernel);

  getContext()->notify_end(UR_FUNCTION_KERNEL_RETAIN, "urKernelRetain", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_RETAIN,
                                    &params);
    logger.info("   <--- urKernelRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t UR_APICALL urKernelRelease(
    /// [in][release] handle for the Kernel to release
    ur_kernel_handle_t hKernel) {
  auto pfnRelease = getContext()->urDdiTable.Kernel.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_release_params_t params = {&hKernel};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_KERNEL_RELEASE,
                                                 "urKernelRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelRelease\n");

  ur_result_t result = pfnRelease(hKernel);

  getContext()->notify_end(UR_FUNCTION_KERNEL_RELEASE, "urKernelRelease",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_RELEASE,
                                    &params);
    logger.info("   <--- urKernelRelease({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgPointer
__urdlllocal ur_result_t UR_APICALL urKernelSetArgPointer(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t argIndex,
    /// [in][optional] pointer to USM pointer properties.
    const ur_kernel_arg_pointer_properties_t *pProperties,
    /// [in][optional] Pointer obtained by USM allocation or virtual memory
    /// mapping operation. If null then argument value is considered null.
    const void *pArgValue) {
  auto pfnSetArgPointer = getContext()->urDdiTable.Kernel.pfnSetArgPointer;

  if (nullptr == pfnSetArgPointer)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_set_arg_pointer_params_t params = {&hKernel, &argIndex,
                                               &pProperties, &pArgValue};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SET_ARG_POINTER, "urKernelSetArgPointer", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSetArgPointer\n");

  ur_result_t result =
      pfnSetArgPointer(hKernel, argIndex, pProperties, pArgValue);

  getContext()->notify_end(UR_FUNCTION_KERNEL_SET_ARG_POINTER,
                           "urKernelSetArgPointer", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_SET_ARG_POINTER, &params);
    logger.info("   <--- urKernelSetArgPointer({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetExecInfo
__urdlllocal ur_result_t UR_APICALL urKernelSetExecInfo(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] name of the execution attribute
    ur_kernel_exec_info_t propName,
    /// [in] size in byte the attribute value
    size_t propSize,
    /// [in][optional] pointer to execution info properties.
    const ur_kernel_exec_info_properties_t *pProperties,
    /// [in][typename(propName, propSize)] pointer to memory location holding
    /// the property value.
    const void *pPropValue) {
  auto pfnSetExecInfo = getContext()->urDdiTable.Kernel.pfnSetExecInfo;

  if (nullptr == pfnSetExecInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_set_exec_info_params_t params = {&hKernel, &propName, &propSize,
                                             &pProperties, &pPropValue};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SET_EXEC_INFO, "urKernelSetExecInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSetExecInfo\n");

  ur_result_t result =
      pfnSetExecInfo(hKernel, propName, propSize, pProperties, pPropValue);

  getContext()->notify_end(UR_FUNCTION_KERNEL_SET_EXEC_INFO,
                           "urKernelSetExecInfo", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_KERNEL_SET_EXEC_INFO,
                                    &params);
    logger.info("   <--- urKernelSetExecInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgSampler
__urdlllocal ur_result_t UR_APICALL urKernelSetArgSampler(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t argIndex,
    /// [in][optional] pointer to sampler properties.
    const ur_kernel_arg_sampler_properties_t *pProperties,
    /// [in] handle of Sampler object.
    ur_sampler_handle_t hArgValue) {
  auto pfnSetArgSampler = getContext()->urDdiTable.Kernel.pfnSetArgSampler;

  if (nullptr == pfnSetArgSampler)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_set_arg_sampler_params_t params = {&hKernel, &argIndex,
                                               &pProperties, &hArgValue};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SET_ARG_SAMPLER, "urKernelSetArgSampler", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSetArgSampler\n");

  ur_result_t result =
      pfnSetArgSampler(hKernel, argIndex, pProperties, hArgValue);

  getContext()->notify_end(UR_FUNCTION_KERNEL_SET_ARG_SAMPLER,
                           "urKernelSetArgSampler", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_SET_ARG_SAMPLER, &params);
    logger.info("   <--- urKernelSetArgSampler({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgMemObj
__urdlllocal ur_result_t UR_APICALL urKernelSetArgMemObj(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t argIndex,
    /// [in][optional] pointer to Memory object properties.
    const ur_kernel_arg_mem_obj_properties_t *pProperties,
    /// [in][optional] handle of Memory object.
    ur_mem_handle_t hArgValue) {
  auto pfnSetArgMemObj = getContext()->urDdiTable.Kernel.pfnSetArgMemObj;

  if (nullptr == pfnSetArgMemObj)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_set_arg_mem_obj_params_t params = {&hKernel, &argIndex,
                                               &pProperties, &hArgValue};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SET_ARG_MEM_OBJ, "urKernelSetArgMemObj", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSetArgMemObj\n");

  ur_result_t result =
      pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue);

  getContext()->notify_end(UR_FUNCTION_KERNEL_SET_ARG_MEM_OBJ,
                           "urKernelSetArgMemObj", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_SET_ARG_MEM_OBJ, &params);
    logger.info("   <--- urKernelSetArgMemObj({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] the number of elements in the pSpecConstants array
    uint32_t count,
    /// [in] array of specialization constant value descriptions
    const ur_specialization_constant_info_t *pSpecConstants) {
  auto pfnSetSpecializationConstants =
      getContext()->urDdiTable.Kernel.pfnSetSpecializationConstants;

  if (nullptr == pfnSetSpecializationConstants)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_set_specialization_constants_params_t params = {&hKernel, &count,
                                                            &pSpecConstants};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SET_SPECIALIZATION_CONSTANTS,
      "urKernelSetSpecializationConstants", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSetSpecializationConstants\n");

  ur_result_t result =
      pfnSetSpecializationConstants(hKernel, count, pSpecConstants);

  getContext()->notify_end(UR_FUNCTION_KERNEL_SET_SPECIALIZATION_CONSTANTS,
                           "urKernelSetSpecializationConstants", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_SET_SPECIALIZATION_CONSTANTS, &params);
    logger.info("   <--- urKernelSetSpecializationConstants({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelGetNativeHandle(
    /// [in] handle of the kernel.
    ur_kernel_handle_t hKernel,
    /// [out] a pointer to the native handle of the kernel.
    ur_native_handle_t *phNativeKernel) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Kernel.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_get_native_handle_params_t params = {&hKernel, &phNativeKernel};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_GET_NATIVE_HANDLE, "urKernelGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hKernel, phNativeKernel);

  getContext()->notify_end(UR_FUNCTION_KERNEL_GET_NATIVE_HANDLE,
                           "urKernelGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urKernelGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the kernel.
    ur_native_handle_t hNativeKernel,
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in][optional] handle of the program associated with the kernel
    ur_program_handle_t hProgram,
    /// [in][optional] pointer to native kernel properties struct
    const ur_kernel_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the kernel object created.
    ur_kernel_handle_t *phKernel) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Kernel.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_create_with_native_handle_params_t params = {
      &hNativeKernel, &hContext, &hProgram, &pProperties, &phKernel};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_KERNEL_CREATE_WITH_NATIVE_HANDLE,
                                 "urKernelCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelCreateWithNativeHandle\n");

  ur_result_t result = pfnCreateWithNativeHandle(
      hNativeKernel, hContext, hProgram, pProperties, phKernel);

  getContext()->notify_end(UR_FUNCTION_KERNEL_CREATE_WITH_NATIVE_HANDLE,
                           "urKernelCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urKernelCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetSuggestedLocalWorkSize
__urdlllocal ur_result_t UR_APICALL urKernelGetSuggestedLocalWorkSize(
    /// [in] handle of the kernel
    ur_kernel_handle_t hKernel,
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] number of dimensions, from 1 to 3, to specify the global
    /// and work-group work-items
    uint32_t numWorkDim,
    /// [in] pointer to an array of numWorkDim unsigned values that specify
    /// the offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkOffset,
    /// [in] pointer to an array of numWorkDim unsigned values that specify
    /// the number of global work-items in workDim that will execute the
    /// kernel function
    const size_t *pGlobalWorkSize,
    /// [out] pointer to an array of numWorkDim unsigned values that specify
    /// suggested local work size that will contain the result of the query
    size_t *pSuggestedLocalWorkSize) {
  auto pfnGetSuggestedLocalWorkSize =
      getContext()->urDdiTable.Kernel.pfnGetSuggestedLocalWorkSize;

  if (nullptr == pfnGetSuggestedLocalWorkSize)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_get_suggested_local_work_size_params_t params = {
      &hKernel,           &hQueue,          &numWorkDim,
      &pGlobalWorkOffset, &pGlobalWorkSize, &pSuggestedLocalWorkSize};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_GET_SUGGESTED_LOCAL_WORK_SIZE,
      "urKernelGetSuggestedLocalWorkSize", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelGetSuggestedLocalWorkSize\n");

  ur_result_t result = pfnGetSuggestedLocalWorkSize(
      hKernel, hQueue, numWorkDim, pGlobalWorkOffset, pGlobalWorkSize,
      pSuggestedLocalWorkSize);

  getContext()->notify_end(UR_FUNCTION_KERNEL_GET_SUGGESTED_LOCAL_WORK_SIZE,
                           "urKernelGetSuggestedLocalWorkSize", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_GET_SUGGESTED_LOCAL_WORK_SIZE, &params);
    logger.info("   <--- urKernelGetSuggestedLocalWorkSize({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetInfo
__urdlllocal ur_result_t UR_APICALL urQueueGetInfo(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] name of the queue property to query
    ur_queue_info_t propName,
    /// [in] size in bytes of the queue property value provided
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] value of the queue
    /// property
    void *pPropValue,
    /// [out][optional] size in bytes returned in queue property value
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Queue.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_get_info_params_t params = {&hQueue, &propName, &propSize,
                                       &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_QUEUE_GET_INFO,
                                                 "urQueueGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hQueue, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_QUEUE_GET_INFO, "urQueueGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_QUEUE_GET_INFO,
                                    &params);
    logger.info("   <--- urQueueGetInfo({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueCreate
__urdlllocal ur_result_t UR_APICALL urQueueCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] pointer to queue creation properties.
    const ur_queue_properties_t *pProperties,
    /// [out][alloc] pointer to handle of queue object created
    ur_queue_handle_t *phQueue) {
  auto pfnCreate = getContext()->urDdiTable.Queue.pfnCreate;

  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_create_params_t params = {&hContext, &hDevice, &pProperties,
                                     &phQueue};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_QUEUE_CREATE,
                                                 "urQueueCreate", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueCreate\n");

  ur_result_t result = pfnCreate(hContext, hDevice, pProperties, phQueue);

  getContext()->notify_end(UR_FUNCTION_QUEUE_CREATE, "urQueueCreate", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_QUEUE_CREATE,
                                    &params);
    logger.info("   <--- urQueueCreate({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRetain
__urdlllocal ur_result_t UR_APICALL urQueueRetain(
    /// [in][retain] handle of the queue object to get access
    ur_queue_handle_t hQueue) {
  auto pfnRetain = getContext()->urDdiTable.Queue.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_retain_params_t params = {&hQueue};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_QUEUE_RETAIN,
                                                 "urQueueRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueRetain\n");

  ur_result_t result = pfnRetain(hQueue);

  getContext()->notify_end(UR_FUNCTION_QUEUE_RETAIN, "urQueueRetain", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_QUEUE_RETAIN,
                                    &params);
    logger.info("   <--- urQueueRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRelease
__urdlllocal ur_result_t UR_APICALL urQueueRelease(
    /// [in][release] handle of the queue object to release
    ur_queue_handle_t hQueue) {
  auto pfnRelease = getContext()->urDdiTable.Queue.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_release_params_t params = {&hQueue};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_QUEUE_RELEASE,
                                                 "urQueueRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueRelease\n");

  ur_result_t result = pfnRelease(hQueue);

  getContext()->notify_end(UR_FUNCTION_QUEUE_RELEASE, "urQueueRelease", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_QUEUE_RELEASE,
                                    &params);
    logger.info("   <--- urQueueRelease({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urQueueGetNativeHandle(
    /// [in] handle of the queue.
    ur_queue_handle_t hQueue,
    /// [in][optional] pointer to native descriptor
    ur_queue_native_desc_t *pDesc,
    /// [out] a pointer to the native handle of the queue.
    ur_native_handle_t *phNativeQueue) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Queue.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_get_native_handle_params_t params = {&hQueue, &pDesc,
                                                &phNativeQueue};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_QUEUE_GET_NATIVE_HANDLE, "urQueueGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hQueue, pDesc, phNativeQueue);

  getContext()->notify_end(UR_FUNCTION_QUEUE_GET_NATIVE_HANDLE,
                           "urQueueGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_QUEUE_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urQueueGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the queue.
    ur_native_handle_t hNativeQueue,
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in][optional] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] pointer to native queue properties struct
    const ur_queue_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the queue object created.
    ur_queue_handle_t *phQueue) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Queue.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_create_with_native_handle_params_t params = {
      &hNativeQueue, &hContext, &hDevice, &pProperties, &phQueue};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_QUEUE_CREATE_WITH_NATIVE_HANDLE,
                                 "urQueueCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueCreateWithNativeHandle\n");

  ur_result_t result = pfnCreateWithNativeHandle(hNativeQueue, hContext,
                                                 hDevice, pProperties, phQueue);

  getContext()->notify_end(UR_FUNCTION_QUEUE_CREATE_WITH_NATIVE_HANDLE,
                           "urQueueCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_QUEUE_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urQueueCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFinish
__urdlllocal ur_result_t UR_APICALL urQueueFinish(
    /// [in] handle of the queue to be finished.
    ur_queue_handle_t hQueue) {
  auto pfnFinish = getContext()->urDdiTable.Queue.pfnFinish;

  if (nullptr == pfnFinish)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_finish_params_t params = {&hQueue};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_QUEUE_FINISH,
                                                 "urQueueFinish", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueFinish\n");

  ur_result_t result = pfnFinish(hQueue);

  getContext()->notify_end(UR_FUNCTION_QUEUE_FINISH, "urQueueFinish", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_QUEUE_FINISH,
                                    &params);
    logger.info("   <--- urQueueFinish({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFlush
__urdlllocal ur_result_t UR_APICALL urQueueFlush(
    /// [in] handle of the queue to be flushed.
    ur_queue_handle_t hQueue) {
  auto pfnFlush = getContext()->urDdiTable.Queue.pfnFlush;

  if (nullptr == pfnFlush)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_queue_flush_params_t params = {&hQueue};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_QUEUE_FLUSH,
                                                 "urQueueFlush", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urQueueFlush\n");

  ur_result_t result = pfnFlush(hQueue);

  getContext()->notify_end(UR_FUNCTION_QUEUE_FLUSH, "urQueueFlush", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_QUEUE_FLUSH, &params);
    logger.info("   <--- urQueueFlush({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetInfo
__urdlllocal ur_result_t UR_APICALL urEventGetInfo(
    /// [in] handle of the event object
    ur_event_handle_t hEvent,
    /// [in] the name of the event property to query
    ur_event_info_t propName,
    /// [in] size in bytes of the event property value
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] value of the event
    /// property
    void *pPropValue,
    /// [out][optional] bytes returned in event property
    size_t *pPropSizeRet) {
  auto pfnGetInfo = getContext()->urDdiTable.Event.pfnGetInfo;

  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_get_info_params_t params = {&hEvent, &propName, &propSize,
                                       &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_EVENT_GET_INFO,
                                                 "urEventGetInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventGetInfo\n");

  ur_result_t result =
      pfnGetInfo(hEvent, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_EVENT_GET_INFO, "urEventGetInfo",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_EVENT_GET_INFO,
                                    &params);
    logger.info("   <--- urEventGetInfo({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetProfilingInfo
__urdlllocal ur_result_t UR_APICALL urEventGetProfilingInfo(
    /// [in] handle of the event object
    ur_event_handle_t hEvent,
    /// [in] the name of the profiling property to query
    ur_profiling_info_t propName,
    /// [in] size in bytes of the profiling property value
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] value of the profiling
    /// property
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes returned in
    /// propValue
    size_t *pPropSizeRet) {
  auto pfnGetProfilingInfo = getContext()->urDdiTable.Event.pfnGetProfilingInfo;

  if (nullptr == pfnGetProfilingInfo)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_get_profiling_info_params_t params = {&hEvent, &propName, &propSize,
                                                 &pPropValue, &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_EVENT_GET_PROFILING_INFO, "urEventGetProfilingInfo", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventGetProfilingInfo\n");

  ur_result_t result =
      pfnGetProfilingInfo(hEvent, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_EVENT_GET_PROFILING_INFO,
                           "urEventGetProfilingInfo", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_EVENT_GET_PROFILING_INFO, &params);
    logger.info("   <--- urEventGetProfilingInfo({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventWait
__urdlllocal ur_result_t UR_APICALL urEventWait(
    /// [in] number of events in the event list
    uint32_t numEvents,
    /// [in][range(0, numEvents)] pointer to a list of events to wait for
    /// completion
    const ur_event_handle_t *phEventWaitList) {
  auto pfnWait = getContext()->urDdiTable.Event.pfnWait;

  if (nullptr == pfnWait)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_wait_params_t params = {&numEvents, &phEventWaitList};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_EVENT_WAIT,
                                                 "urEventWait", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventWait\n");

  ur_result_t result = pfnWait(numEvents, phEventWaitList);

  getContext()->notify_end(UR_FUNCTION_EVENT_WAIT, "urEventWait", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_EVENT_WAIT, &params);
    logger.info("   <--- urEventWait({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRetain
__urdlllocal ur_result_t UR_APICALL urEventRetain(
    /// [in][retain] handle of the event object
    ur_event_handle_t hEvent) {
  auto pfnRetain = getContext()->urDdiTable.Event.pfnRetain;

  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_retain_params_t params = {&hEvent};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_EVENT_RETAIN,
                                                 "urEventRetain", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventRetain\n");

  ur_result_t result = pfnRetain(hEvent);

  getContext()->notify_end(UR_FUNCTION_EVENT_RETAIN, "urEventRetain", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_EVENT_RETAIN,
                                    &params);
    logger.info("   <--- urEventRetain({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRelease
__urdlllocal ur_result_t UR_APICALL urEventRelease(
    /// [in][release] handle of the event object
    ur_event_handle_t hEvent) {
  auto pfnRelease = getContext()->urDdiTable.Event.pfnRelease;

  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_release_params_t params = {&hEvent};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_EVENT_RELEASE,
                                                 "urEventRelease", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventRelease\n");

  ur_result_t result = pfnRelease(hEvent);

  getContext()->notify_end(UR_FUNCTION_EVENT_RELEASE, "urEventRelease", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_EVENT_RELEASE,
                                    &params);
    logger.info("   <--- urEventRelease({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventGetNativeHandle(
    /// [in] handle of the event.
    ur_event_handle_t hEvent,
    /// [out] a pointer to the native handle of the event.
    ur_native_handle_t *phNativeEvent) {
  auto pfnGetNativeHandle = getContext()->urDdiTable.Event.pfnGetNativeHandle;

  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_get_native_handle_params_t params = {&hEvent, &phNativeEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_EVENT_GET_NATIVE_HANDLE, "urEventGetNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventGetNativeHandle\n");

  ur_result_t result = pfnGetNativeHandle(hEvent, phNativeEvent);

  getContext()->notify_end(UR_FUNCTION_EVENT_GET_NATIVE_HANDLE,
                           "urEventGetNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_EVENT_GET_NATIVE_HANDLE, &params);
    logger.info("   <--- urEventGetNativeHandle({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the event.
    ur_native_handle_t hNativeEvent,
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in][optional] pointer to native event properties struct
    const ur_event_native_properties_t *pProperties,
    /// [out][alloc] pointer to the handle of the event object created.
    ur_event_handle_t *phEvent) {
  auto pfnCreateWithNativeHandle =
      getContext()->urDdiTable.Event.pfnCreateWithNativeHandle;

  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_create_with_native_handle_params_t params = {
      &hNativeEvent, &hContext, &pProperties, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_EVENT_CREATE_WITH_NATIVE_HANDLE,
                                 "urEventCreateWithNativeHandle", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventCreateWithNativeHandle\n");

  ur_result_t result =
      pfnCreateWithNativeHandle(hNativeEvent, hContext, pProperties, phEvent);

  getContext()->notify_end(UR_FUNCTION_EVENT_CREATE_WITH_NATIVE_HANDLE,
                           "urEventCreateWithNativeHandle", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_EVENT_CREATE_WITH_NATIVE_HANDLE, &params);
    logger.info("   <--- urEventCreateWithNativeHandle({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventSetCallback
__urdlllocal ur_result_t UR_APICALL urEventSetCallback(
    /// [in] handle of the event object
    ur_event_handle_t hEvent,
    /// [in] execution status of the event
    ur_execution_info_t execStatus,
    /// [in] execution status of the event
    ur_event_callback_t pfnNotify,
    /// [in][out][optional] pointer to data to be passed to callback.
    void *pUserData) {
  auto pfnSetCallback = getContext()->urDdiTable.Event.pfnSetCallback;

  if (nullptr == pfnSetCallback)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_event_set_callback_params_t params = {&hEvent, &execStatus, &pfnNotify,
                                           &pUserData};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_EVENT_SET_CALLBACK,
                                                 "urEventSetCallback", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEventSetCallback\n");

  ur_result_t result = pfnSetCallback(hEvent, execStatus, pfnNotify, pUserData);

  getContext()->notify_end(UR_FUNCTION_EVENT_SET_CALLBACK, "urEventSetCallback",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_EVENT_SET_CALLBACK,
                                    &params);
    logger.info("   <--- urEventSetCallback({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunch
__urdlllocal ur_result_t UR_APICALL urEnqueueKernelLaunch(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] number of dimensions, from 1 to 3, to specify the global and
    /// work-group work-items
    uint32_t workDim,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkOffset,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// number of global work-items in workDim that will execute the kernel
    /// function
    const size_t *pGlobalWorkSize,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the number of local work-items forming a work-group that will
    /// execute the kernel function.
    /// If nullptr, the runtime implementation will choose the work-group size.
    const size_t *pLocalWorkSize,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular kernel execution instance. If phEventWaitList and phEvent
    /// are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnKernelLaunch = getContext()->urDdiTable.Enqueue.pfnKernelLaunch;

  if (nullptr == pfnKernelLaunch)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_kernel_launch_params_t params = {&hQueue,
                                              &hKernel,
                                              &workDim,
                                              &pGlobalWorkOffset,
                                              &pGlobalWorkSize,
                                              &pLocalWorkSize,
                                              &numEventsInWaitList,
                                              &phEventWaitList,
                                              &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH, "urEnqueueKernelLaunch", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueKernelLaunch\n");

  ur_result_t result = pfnKernelLaunch(
      hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH,
                           "urEnqueueKernelLaunch", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH,
                                    &params);
    logger.info("   <--- urEnqueueKernelLaunch({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueEventsWait
__urdlllocal ur_result_t UR_APICALL urEnqueueEventsWait(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands
    /// must be complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnEventsWait = getContext()->urDdiTable.Enqueue.pfnEventsWait;

  if (nullptr == pfnEventsWait)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_events_wait_params_t params = {&hQueue, &numEventsInWaitList,
                                            &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_EVENTS_WAIT, "urEnqueueEventsWait", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueEventsWait\n");

  ur_result_t result =
      pfnEventsWait(hQueue, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_EVENTS_WAIT,
                           "urEnqueueEventsWait", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_EVENTS_WAIT,
                                    &params);
    logger.info("   <--- urEnqueueEventsWait({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueEventsWaitWithBarrier
__urdlllocal ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands
    /// must be complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnEventsWaitWithBarrier =
      getContext()->urDdiTable.Enqueue.pfnEventsWaitWithBarrier;

  if (nullptr == pfnEventsWaitWithBarrier)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_events_wait_with_barrier_params_t params = {
      &hQueue, &numEventsInWaitList, &phEventWaitList, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER,
                                 "urEnqueueEventsWaitWithBarrier", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueEventsWaitWithBarrier\n");

  ur_result_t result = pfnEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                                phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER,
                           "urEnqueueEventsWaitWithBarrier", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER, &params);
    logger.info("   <--- urEnqueueEventsWaitWithBarrier({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferRead
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferRead(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(offset, size)] handle of the buffer object
    ur_mem_handle_t hBuffer,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingRead,
    /// [in] offset in bytes in the buffer object
    size_t offset,
    /// [in] size in bytes of data being read
    size_t size,
    /// [in] pointer to host memory where data is to be read into
    void *pDst,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemBufferRead = getContext()->urDdiTable.Enqueue.pfnMemBufferRead;

  if (nullptr == pfnMemBufferRead)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_read_params_t params = {
      &hQueue, &hBuffer, &blockingRead,        &offset,
      &size,   &pDst,    &numEventsInWaitList, &phEventWaitList,
      &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ, "urEnqueueMemBufferRead", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferRead\n");

  ur_result_t result =
      pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size, pDst,
                       numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ,
                           "urEnqueueMemBufferRead", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ, &params);
    logger.info("   <--- urEnqueueMemBufferRead({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWrite
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(offset, size)] handle of the buffer object
    ur_mem_handle_t hBuffer,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingWrite,
    /// [in] offset in bytes in the buffer object
    size_t offset,
    /// [in] size in bytes of data being written
    size_t size,
    /// [in] pointer to host memory where data is to be written from
    const void *pSrc,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemBufferWrite = getContext()->urDdiTable.Enqueue.pfnMemBufferWrite;

  if (nullptr == pfnMemBufferWrite)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_write_params_t params = {
      &hQueue, &hBuffer, &blockingWrite,       &offset,
      &size,   &pSrc,    &numEventsInWaitList, &phEventWaitList,
      &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE, "urEnqueueMemBufferWrite", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferWrite\n");

  ur_result_t result =
      pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size, pSrc,
                        numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE,
                           "urEnqueueMemBufferWrite", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE, &params);
    logger.info("   <--- urEnqueueMemBufferWrite({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferReadRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(bufferOrigin, region)] handle of the buffer object
    ur_mem_handle_t hBuffer,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingRead,
    /// [in] 3D offset in the buffer
    ur_rect_offset_t bufferOrigin,
    /// [in] 3D offset in the host region
    ur_rect_offset_t hostOrigin,
    /// [in] 3D rectangular region descriptor: width, height, depth
    ur_rect_region_t region,
    /// [in] length of each row in bytes in the buffer object
    size_t bufferRowPitch,
    /// [in] length of each 2D slice in bytes in the buffer object being read
    size_t bufferSlicePitch,
    /// [in] length of each row in bytes in the host memory region pointed by
    /// dst
    size_t hostRowPitch,
    /// [in] length of each 2D slice in bytes in the host memory region
    /// pointed by dst
    size_t hostSlicePitch,
    /// [in] pointer to host memory where data is to be read into
    void *pDst,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemBufferReadRect =
      getContext()->urDdiTable.Enqueue.pfnMemBufferReadRect;

  if (nullptr == pfnMemBufferReadRect)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_read_rect_params_t params = {
      &hQueue,          &hBuffer,        &blockingRead,   &bufferOrigin,
      &hostOrigin,      &region,         &bufferRowPitch, &bufferSlicePitch,
      &hostRowPitch,    &hostSlicePitch, &pDst,           &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ_RECT,
                                 "urEnqueueMemBufferReadRect", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferReadRect\n");

  ur_result_t result = pfnMemBufferReadRect(
      hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ_RECT,
                           "urEnqueueMemBufferReadRect", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ_RECT, &params);
    logger.info("   <--- urEnqueueMemBufferReadRect({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWriteRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(bufferOrigin, region)] handle of the buffer object
    ur_mem_handle_t hBuffer,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingWrite,
    /// [in] 3D offset in the buffer
    ur_rect_offset_t bufferOrigin,
    /// [in] 3D offset in the host region
    ur_rect_offset_t hostOrigin,
    /// [in] 3D rectangular region descriptor: width, height, depth
    ur_rect_region_t region,
    /// [in] length of each row in bytes in the buffer object
    size_t bufferRowPitch,
    /// [in] length of each 2D slice in bytes in the buffer object being
    /// written
    size_t bufferSlicePitch,
    /// [in] length of each row in bytes in the host memory region pointed by
    /// src
    size_t hostRowPitch,
    /// [in] length of each 2D slice in bytes in the host memory region
    /// pointed by src
    size_t hostSlicePitch,
    /// [in] pointer to host memory where data is to be written from
    void *pSrc,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] points to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemBufferWriteRect =
      getContext()->urDdiTable.Enqueue.pfnMemBufferWriteRect;

  if (nullptr == pfnMemBufferWriteRect)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_write_rect_params_t params = {
      &hQueue,          &hBuffer,        &blockingWrite,  &bufferOrigin,
      &hostOrigin,      &region,         &bufferRowPitch, &bufferSlicePitch,
      &hostRowPitch,    &hostSlicePitch, &pSrc,           &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE_RECT,
                                 "urEnqueueMemBufferWriteRect", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferWriteRect\n");

  ur_result_t result = pfnMemBufferWriteRect(
      hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE_RECT,
                           "urEnqueueMemBufferWriteRect", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE_RECT, &params);
    logger.info("   <--- urEnqueueMemBufferWriteRect({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopy
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(srcOffset, size)] handle of the src buffer object
    ur_mem_handle_t hBufferSrc,
    /// [in][bounds(dstOffset, size)] handle of the dest buffer object
    ur_mem_handle_t hBufferDst,
    /// [in] offset into hBufferSrc to begin copying from
    size_t srcOffset,
    /// [in] offset info hBufferDst to begin copying into
    size_t dstOffset,
    /// [in] size in bytes of data being copied
    size_t size,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemBufferCopy = getContext()->urDdiTable.Enqueue.pfnMemBufferCopy;

  if (nullptr == pfnMemBufferCopy)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_copy_params_t params = {
      &hQueue, &hBufferSrc,          &hBufferDst,      &srcOffset, &dstOffset,
      &size,   &numEventsInWaitList, &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY, "urEnqueueMemBufferCopy", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferCopy\n");

  ur_result_t result =
      pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset,
                       size, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY,
                           "urEnqueueMemBufferCopy", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY, &params);
    logger.info("   <--- urEnqueueMemBufferCopy({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopyRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(srcOrigin, region)] handle of the source buffer object
    ur_mem_handle_t hBufferSrc,
    /// [in][bounds(dstOrigin, region)] handle of the dest buffer object
    ur_mem_handle_t hBufferDst,
    /// [in] 3D offset in the source buffer
    ur_rect_offset_t srcOrigin,
    /// [in] 3D offset in the destination buffer
    ur_rect_offset_t dstOrigin,
    /// [in] source 3D rectangular region descriptor: width, height, depth
    ur_rect_region_t region,
    /// [in] length of each row in bytes in the source buffer object
    size_t srcRowPitch,
    /// [in] length of each 2D slice in bytes in the source buffer object
    size_t srcSlicePitch,
    /// [in] length of each row in bytes in the destination buffer object
    size_t dstRowPitch,
    /// [in] length of each 2D slice in bytes in the destination buffer object
    size_t dstSlicePitch,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemBufferCopyRect =
      getContext()->urDdiTable.Enqueue.pfnMemBufferCopyRect;

  if (nullptr == pfnMemBufferCopyRect)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_copy_rect_params_t params = {
      &hQueue,      &hBufferSrc,    &hBufferDst,          &srcOrigin,
      &dstOrigin,   &region,        &srcRowPitch,         &srcSlicePitch,
      &dstRowPitch, &dstSlicePitch, &numEventsInWaitList, &phEventWaitList,
      &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY_RECT,
                                 "urEnqueueMemBufferCopyRect", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferCopyRect\n");

  ur_result_t result = pfnMemBufferCopyRect(
      hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region, srcRowPitch,
      srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList,
      phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY_RECT,
                           "urEnqueueMemBufferCopyRect", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY_RECT, &params);
    logger.info("   <--- urEnqueueMemBufferCopyRect({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferFill
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferFill(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(offset, size)] handle of the buffer object
    ur_mem_handle_t hBuffer,
    /// [in] pointer to the fill pattern
    const void *pPattern,
    /// [in] size in bytes of the pattern
    size_t patternSize,
    /// [in] offset into the buffer
    size_t offset,
    /// [in] fill size in bytes, must be a multiple of patternSize
    size_t size,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemBufferFill = getContext()->urDdiTable.Enqueue.pfnMemBufferFill;

  if (nullptr == pfnMemBufferFill)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_fill_params_t params = {&hQueue,
                                                &hBuffer,
                                                &pPattern,
                                                &patternSize,
                                                &offset,
                                                &size,
                                                &numEventsInWaitList,
                                                &phEventWaitList,
                                                &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_BUFFER_FILL, "urEnqueueMemBufferFill", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferFill\n");

  ur_result_t result =
      pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize, offset, size,
                       numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_FILL,
                           "urEnqueueMemBufferFill", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_FILL, &params);
    logger.info("   <--- urEnqueueMemBufferFill({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageRead
__urdlllocal ur_result_t UR_APICALL urEnqueueMemImageRead(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(origin, region)] handle of the image object
    ur_mem_handle_t hImage,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingRead,
    /// [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_offset_t origin,
    /// [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
    /// image
    ur_rect_region_t region,
    /// [in] length of each row in bytes
    size_t rowPitch,
    /// [in] length of each 2D slice of the 3D image
    size_t slicePitch,
    /// [in] pointer to host memory where image is to be read into
    void *pDst,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemImageRead = getContext()->urDdiTable.Enqueue.pfnMemImageRead;

  if (nullptr == pfnMemImageRead)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_image_read_params_t params = {
      &hQueue,          &hImage, &blockingRead,
      &origin,          &region, &rowPitch,
      &slicePitch,      &pDst,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_IMAGE_READ, "urEnqueueMemImageRead", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemImageRead\n");

  ur_result_t result = pfnMemImageRead(
      hQueue, hImage, blockingRead, origin, region, rowPitch, slicePitch, pDst,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_IMAGE_READ,
                           "urEnqueueMemImageRead", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_IMAGE_READ, &params);
    logger.info("   <--- urEnqueueMemImageRead({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageWrite
__urdlllocal ur_result_t UR_APICALL urEnqueueMemImageWrite(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(origin, region)] handle of the image object
    ur_mem_handle_t hImage,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingWrite,
    /// [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_offset_t origin,
    /// [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
    /// image
    ur_rect_region_t region,
    /// [in] length of each row in bytes
    size_t rowPitch,
    /// [in] length of each 2D slice of the 3D image
    size_t slicePitch,
    /// [in] pointer to host memory where image is to be read into
    void *pSrc,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemImageWrite = getContext()->urDdiTable.Enqueue.pfnMemImageWrite;

  if (nullptr == pfnMemImageWrite)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_image_write_params_t params = {
      &hQueue,          &hImage, &blockingWrite,
      &origin,          &region, &rowPitch,
      &slicePitch,      &pSrc,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_IMAGE_WRITE, "urEnqueueMemImageWrite", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemImageWrite\n");

  ur_result_t result = pfnMemImageWrite(
      hQueue, hImage, blockingWrite, origin, region, rowPitch, slicePitch, pSrc,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_IMAGE_WRITE,
                           "urEnqueueMemImageWrite", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_IMAGE_WRITE, &params);
    logger.info("   <--- urEnqueueMemImageWrite({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemImageCopy
__urdlllocal ur_result_t UR_APICALL urEnqueueMemImageCopy(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(srcOrigin, region)] handle of the src image object
    ur_mem_handle_t hImageSrc,
    /// [in][bounds(dstOrigin, region)] handle of the dest image object
    ur_mem_handle_t hImageDst,
    /// [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
    /// image
    ur_rect_offset_t srcOrigin,
    /// [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
    /// or 3D image
    ur_rect_offset_t dstOrigin,
    /// [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
    /// image
    ur_rect_region_t region,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemImageCopy = getContext()->urDdiTable.Enqueue.pfnMemImageCopy;

  if (nullptr == pfnMemImageCopy)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_image_copy_params_t params = {
      &hQueue, &hImageSrc,           &hImageDst,       &srcOrigin, &dstOrigin,
      &region, &numEventsInWaitList, &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_IMAGE_COPY, "urEnqueueMemImageCopy", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemImageCopy\n");

  ur_result_t result =
      pfnMemImageCopy(hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin,
                      region, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_IMAGE_COPY,
                           "urEnqueueMemImageCopy", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_IMAGE_COPY, &params);
    logger.info("   <--- urEnqueueMemImageCopy({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferMap
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferMap(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(offset, size)] handle of the buffer object
    ur_mem_handle_t hBuffer,
    /// [in] indicates blocking (true), non-blocking (false)
    bool blockingMap,
    /// [in] flags for read, write, readwrite mapping
    ur_map_flags_t mapFlags,
    /// [in] offset in bytes of the buffer region being mapped
    size_t offset,
    /// [in] size in bytes of the buffer region being mapped
    size_t size,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent,
    /// [out] return mapped pointer.  TODO: move it before
    /// numEventsInWaitList?
    void **ppRetMap) {
  auto pfnMemBufferMap = getContext()->urDdiTable.Enqueue.pfnMemBufferMap;

  if (nullptr == pfnMemBufferMap)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_buffer_map_params_t params = {
      &hQueue,  &hBuffer, &blockingMap,         &mapFlags,
      &offset,  &size,    &numEventsInWaitList, &phEventWaitList,
      &phEvent, &ppRetMap};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_MEM_BUFFER_MAP, "urEnqueueMemBufferMap", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemBufferMap\n");

  ur_result_t result =
      pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags, offset, size,
                      numEventsInWaitList, phEventWaitList, phEvent, ppRetMap);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_BUFFER_MAP,
                           "urEnqueueMemBufferMap", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_MEM_BUFFER_MAP, &params);
    logger.info("   <--- urEnqueueMemBufferMap({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemUnmap
__urdlllocal ur_result_t UR_APICALL urEnqueueMemUnmap(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] handle of the memory (buffer or image) object
    ur_mem_handle_t hMem,
    /// [in] mapped host address
    void *pMappedPtr,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnMemUnmap = getContext()->urDdiTable.Enqueue.pfnMemUnmap;

  if (nullptr == pfnMemUnmap)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_mem_unmap_params_t params = {
      &hQueue,          &hMem,   &pMappedPtr, &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ENQUEUE_MEM_UNMAP,
                                                 "urEnqueueMemUnmap", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueMemUnmap\n");

  ur_result_t result = pfnMemUnmap(
      hQueue, hMem, pMappedPtr, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_MEM_UNMAP, "urEnqueueMemUnmap",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_MEM_UNMAP,
                                    &params);
    logger.info("   <--- urEnqueueMemUnmap({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMFill
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMFill(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(0, size)] pointer to USM memory object
    void *pMem,
    /// [in] the size in bytes of the pattern. Must be a power of 2 and less
    /// than or equal to width.
    size_t patternSize,
    /// [in] pointer with the bytes of the pattern to set.
    const void *pPattern,
    /// [in] size in bytes to be set. Must be a multiple of patternSize.
    size_t size,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnUSMFill = getContext()->urDdiTable.Enqueue.pfnUSMFill;

  if (nullptr == pfnUSMFill)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_fill_params_t params = {
      &hQueue,          &pMem,   &patternSize,
      &pPattern,        &size,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ENQUEUE_USM_FILL,
                                                 "urEnqueueUSMFill", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMFill\n");

  ur_result_t result =
      pfnUSMFill(hQueue, pMem, patternSize, pPattern, size, numEventsInWaitList,
                 phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_FILL, "urEnqueueUSMFill",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_USM_FILL,
                                    &params);
    logger.info("   <--- urEnqueueUSMFill({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemcpy
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] blocking or non-blocking copy
    bool blocking,
    /// [in][bounds(0, size)] pointer to the destination USM memory object
    void *pDst,
    /// [in][bounds(0, size)] pointer to the source USM memory object
    const void *pSrc,
    /// [in] size in bytes to be copied
    size_t size,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnUSMMemcpy = getContext()->urDdiTable.Enqueue.pfnUSMMemcpy;

  if (nullptr == pfnUSMMemcpy)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_memcpy_params_t params = {
      &hQueue,          &blocking, &pDst, &pSrc, &size, &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ENQUEUE_USM_MEMCPY,
                                                 "urEnqueueUSMMemcpy", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMMemcpy\n");

  ur_result_t result =
      pfnUSMMemcpy(hQueue, blocking, pDst, pSrc, size, numEventsInWaitList,
                   phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_MEMCPY, "urEnqueueUSMMemcpy",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_USM_MEMCPY,
                                    &params);
    logger.info("   <--- urEnqueueUSMMemcpy({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMPrefetch
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(0, size)] pointer to the USM memory object
    const void *pMem,
    /// [in] size in bytes to be fetched
    size_t size,
    /// [in] USM prefetch flags
    ur_usm_migration_flags_t flags,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnUSMPrefetch = getContext()->urDdiTable.Enqueue.pfnUSMPrefetch;

  if (nullptr == pfnUSMPrefetch)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_prefetch_params_t params = {
      &hQueue,          &pMem,   &size, &flags, &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_USM_PREFETCH, "urEnqueueUSMPrefetch", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMPrefetch\n");

  ur_result_t result = pfnUSMPrefetch(
      hQueue, pMem, size, flags, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_PREFETCH,
                           "urEnqueueUSMPrefetch", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_USM_PREFETCH,
                                    &params);
    logger.info("   <--- urEnqueueUSMPrefetch({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMAdvise
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMAdvise(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][bounds(0, size)] pointer to the USM memory object
    const void *pMem,
    /// [in] size in bytes to be advised
    size_t size,
    /// [in] USM memory advice
    ur_usm_advice_flags_t advice,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance.
    ur_event_handle_t *phEvent) {
  auto pfnUSMAdvise = getContext()->urDdiTable.Enqueue.pfnUSMAdvise;

  if (nullptr == pfnUSMAdvise)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_advise_params_t params = {&hQueue, &pMem, &size, &advice,
                                           &phEvent};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_ENQUEUE_USM_ADVISE,
                                                 "urEnqueueUSMAdvise", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMAdvise\n");

  ur_result_t result = pfnUSMAdvise(hQueue, pMem, size, advice, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_ADVISE, "urEnqueueUSMAdvise",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_USM_ADVISE,
                                    &params);
    logger.info("   <--- urEnqueueUSMAdvise({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMFill2D
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMFill2D(
    /// [in] handle of the queue to submit to.
    ur_queue_handle_t hQueue,
    /// [in][bounds(0, pitch * height)] pointer to memory to be filled.
    void *pMem,
    /// [in] the total width of the destination memory including padding.
    size_t pitch,
    /// [in] the size in bytes of the pattern. Must be a power of 2 and less
    /// than or equal to width.
    size_t patternSize,
    /// [in] pointer with the bytes of the pattern to set.
    const void *pPattern,
    /// [in] the width in bytes of each row to fill. Must be a multiple of
    /// patternSize.
    size_t width,
    /// [in] the height of the columns to fill.
    size_t height,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular kernel execution instance. If phEventWaitList and phEvent
    /// are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnUSMFill2D = getContext()->urDdiTable.Enqueue.pfnUSMFill2D;

  if (nullptr == pfnUSMFill2D)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_fill_2d_params_t params = {
      &hQueue,          &pMem,   &pitch,  &patternSize,
      &pPattern,        &width,  &height, &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_USM_FILL_2D, "urEnqueueUSMFill2D", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMFill2D\n");

  ur_result_t result =
      pfnUSMFill2D(hQueue, pMem, pitch, patternSize, pPattern, width, height,
                   numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_FILL_2D,
                           "urEnqueueUSMFill2D", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_USM_FILL_2D,
                                    &params);
    logger.info("   <--- urEnqueueUSMFill2D({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMMemcpy2D
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    /// [in] handle of the queue to submit to.
    ur_queue_handle_t hQueue,
    /// [in] indicates if this operation should block the host.
    bool blocking,
    /// [in][bounds(0, dstPitch * height)] pointer to memory where data will
    /// be copied.
    void *pDst,
    /// [in] the total width of the source memory including padding.
    size_t dstPitch,
    /// [in][bounds(0, srcPitch * height)] pointer to memory to be copied.
    const void *pSrc,
    /// [in] the total width of the source memory including padding.
    size_t srcPitch,
    /// [in] the width in bytes of each row to be copied.
    size_t width,
    /// [in] the height of columns to be copied.
    size_t height,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular kernel execution instance. If phEventWaitList and phEvent
    /// are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnUSMMemcpy2D = getContext()->urDdiTable.Enqueue.pfnUSMMemcpy2D;

  if (nullptr == pfnUSMMemcpy2D)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_memcpy_2d_params_t params = {
      &hQueue,          &blocking, &pDst,
      &dstPitch,        &pSrc,     &srcPitch,
      &width,           &height,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_USM_MEMCPY_2D, "urEnqueueUSMMemcpy2D", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMMemcpy2D\n");

  ur_result_t result =
      pfnUSMMemcpy2D(hQueue, blocking, pDst, dstPitch, pSrc, srcPitch, width,
                     height, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_MEMCPY_2D,
                           "urEnqueueUSMMemcpy2D", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_USM_MEMCPY_2D,
                                    &params);
    logger.info("   <--- urEnqueueUSMMemcpy2D({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueDeviceGlobalVariableWrite
__urdlllocal ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    /// [in] handle of the queue to submit to.
    ur_queue_handle_t hQueue,
    /// [in] handle of the program containing the device global variable.
    ur_program_handle_t hProgram,
    /// [in] the unique identifier for the device global variable.
    const char *name,
    /// [in] indicates if this operation should block.
    bool blockingWrite,
    /// [in] the number of bytes to copy.
    size_t count,
    /// [in] the byte offset into the device global variable to start copying.
    size_t offset,
    /// [in] pointer to where the data must be copied from.
    const void *pSrc,
    /// [in] size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular kernel execution instance. If phEventWaitList and phEvent
    /// are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnDeviceGlobalVariableWrite =
      getContext()->urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite;

  if (nullptr == pfnDeviceGlobalVariableWrite)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_device_global_variable_write_params_t params = {
      &hQueue,          &hProgram, &name, &blockingWrite,
      &count,           &offset,   &pSrc, &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_WRITE,
      "urEnqueueDeviceGlobalVariableWrite", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueDeviceGlobalVariableWrite\n");

  ur_result_t result = pfnDeviceGlobalVariableWrite(
      hQueue, hProgram, name, blockingWrite, count, offset, pSrc,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_WRITE,
                           "urEnqueueDeviceGlobalVariableWrite", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_WRITE, &params);
    logger.info("   <--- urEnqueueDeviceGlobalVariableWrite({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueDeviceGlobalVariableRead
__urdlllocal ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    /// [in] handle of the queue to submit to.
    ur_queue_handle_t hQueue,
    /// [in] handle of the program containing the device global variable.
    ur_program_handle_t hProgram,
    /// [in] the unique identifier for the device global variable.
    const char *name,
    /// [in] indicates if this operation should block.
    bool blockingRead,
    /// [in] the number of bytes to copy.
    size_t count,
    /// [in] the byte offset into the device global variable to start copying.
    size_t offset,
    /// [in] pointer to where the data must be copied to.
    void *pDst,
    /// [in] size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular kernel execution instance. If phEventWaitList and phEvent
    /// are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnDeviceGlobalVariableRead =
      getContext()->urDdiTable.Enqueue.pfnDeviceGlobalVariableRead;

  if (nullptr == pfnDeviceGlobalVariableRead)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_device_global_variable_read_params_t params = {
      &hQueue,          &hProgram, &name, &blockingRead,
      &count,           &offset,   &pDst, &numEventsInWaitList,
      &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_READ,
      "urEnqueueDeviceGlobalVariableRead", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueDeviceGlobalVariableRead\n");

  ur_result_t result = pfnDeviceGlobalVariableRead(
      hQueue, hProgram, name, blockingRead, count, offset, pDst,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_READ,
                           "urEnqueueDeviceGlobalVariableRead", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_READ, &params);
    logger.info("   <--- urEnqueueDeviceGlobalVariableRead({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueReadHostPipe
__urdlllocal ur_result_t UR_APICALL urEnqueueReadHostPipe(
    /// [in] a valid host command-queue in which the read command
    /// will be queued. hQueue and hProgram must be created with the same
    /// UR context.
    ur_queue_handle_t hQueue,
    /// [in] a program object with a successfully built executable.
    ur_program_handle_t hProgram,
    /// [in] the name of the program scope pipe global variable.
    const char *pipe_symbol,
    /// [in] indicate if the read operation is blocking or non-blocking.
    bool blocking,
    /// [in] a pointer to buffer in host memory that will hold resulting data
    /// from pipe.
    void *pDst,
    /// [in] size of the memory region to read, in bytes.
    size_t size,
    /// [in] number of events in the wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the host pipe read.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] returns an event object that identifies this
    /// read command
    /// and can be used to query or queue a wait for this command to complete.
    /// If phEventWaitList and phEvent are not NULL, phEvent must not refer to
    /// an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnReadHostPipe = getContext()->urDdiTable.Enqueue.pfnReadHostPipe;

  if (nullptr == pfnReadHostPipe)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_read_host_pipe_params_t params = {
      &hQueue, &hProgram, &pipe_symbol,         &blocking,
      &pDst,   &size,     &numEventsInWaitList, &phEventWaitList,
      &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_READ_HOST_PIPE, "urEnqueueReadHostPipe", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueReadHostPipe\n");

  ur_result_t result =
      pfnReadHostPipe(hQueue, hProgram, pipe_symbol, blocking, pDst, size,
                      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_READ_HOST_PIPE,
                           "urEnqueueReadHostPipe", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_READ_HOST_PIPE, &params);
    logger.info("   <--- urEnqueueReadHostPipe({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueWriteHostPipe
__urdlllocal ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    /// [in] a valid host command-queue in which the write command
    /// will be queued. hQueue and hProgram must be created with the same
    /// UR context.
    ur_queue_handle_t hQueue,
    /// [in] a program object with a successfully built executable.
    ur_program_handle_t hProgram,
    /// [in] the name of the program scope pipe global variable.
    const char *pipe_symbol,
    /// [in] indicate if the read and write operations are blocking or
    /// non-blocking.
    bool blocking,
    /// [in] a pointer to buffer in host memory that holds data to be written
    /// to the host pipe.
    void *pSrc,
    /// [in] size of the memory region to read or write, in bytes.
    size_t size,
    /// [in] number of events in the wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the host pipe write.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] returns an event object that identifies this
    /// write command
    /// and can be used to query or queue a wait for this command to complete.
    /// If phEventWaitList and phEvent are not NULL, phEvent must not refer to
    /// an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnWriteHostPipe = getContext()->urDdiTable.Enqueue.pfnWriteHostPipe;

  if (nullptr == pfnWriteHostPipe)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_write_host_pipe_params_t params = {
      &hQueue, &hProgram, &pipe_symbol,         &blocking,
      &pSrc,   &size,     &numEventsInWaitList, &phEventWaitList,
      &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_WRITE_HOST_PIPE, "urEnqueueWriteHostPipe", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueWriteHostPipe\n");

  ur_result_t result =
      pfnWriteHostPipe(hQueue, hProgram, pipe_symbol, blocking, pSrc, size,
                       numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_WRITE_HOST_PIPE,
                           "urEnqueueWriteHostPipe", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_WRITE_HOST_PIPE, &params);
    logger.info("   <--- urEnqueueWriteHostPipe({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMDeviceAllocExp
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMDeviceAllocExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][optional] USM pool descriptor
    ur_usm_pool_handle_t pPool,
    /// [in] minimum size in bytes of the USM memory object to be allocated
    const size_t size,
    /// [in][optional] pointer to the enqueue async alloc properties
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating no wait
    /// events.
    const ur_event_handle_t *phEventWaitList,
    /// [out] pointer to USM memory object
    void **ppMem,
    /// [out][optional] return an event object that identifies the async alloc
    ur_event_handle_t *phEvent) {
  auto pfnUSMDeviceAllocExp =
      getContext()->urDdiTable.EnqueueExp.pfnUSMDeviceAllocExp;

  if (nullptr == pfnUSMDeviceAllocExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_device_alloc_exp_params_t params = {
      &hQueue,          &pPool, &size,   &pProperties, &numEventsInWaitList,
      &phEventWaitList, &ppMem, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_USM_DEVICE_ALLOC_EXP,
                                 "urEnqueueUSMDeviceAllocExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMDeviceAllocExp\n");

  ur_result_t result = pfnUSMDeviceAllocExp(hQueue, pPool, size, pProperties,
                                            numEventsInWaitList,
                                            phEventWaitList, ppMem, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_DEVICE_ALLOC_EXP,
                           "urEnqueueUSMDeviceAllocExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_USM_DEVICE_ALLOC_EXP, &params);
    logger.info("   <--- urEnqueueUSMDeviceAllocExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMSharedAllocExp
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMSharedAllocExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][optional] USM pool descriptor
    ur_usm_pool_handle_t pPool,
    /// [in] minimum size in bytes of the USM memory object to be allocated
    const size_t size,
    /// [in][optional] pointer to the enqueue async alloc properties
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating no wait
    /// events.
    const ur_event_handle_t *phEventWaitList,
    /// [out] pointer to USM memory object
    void **ppMem,
    /// [out][optional] return an event object that identifies the async alloc
    ur_event_handle_t *phEvent) {
  auto pfnUSMSharedAllocExp =
      getContext()->urDdiTable.EnqueueExp.pfnUSMSharedAllocExp;

  if (nullptr == pfnUSMSharedAllocExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_shared_alloc_exp_params_t params = {
      &hQueue,          &pPool, &size,   &pProperties, &numEventsInWaitList,
      &phEventWaitList, &ppMem, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_USM_SHARED_ALLOC_EXP,
                                 "urEnqueueUSMSharedAllocExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMSharedAllocExp\n");

  ur_result_t result = pfnUSMSharedAllocExp(hQueue, pPool, size, pProperties,
                                            numEventsInWaitList,
                                            phEventWaitList, ppMem, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_SHARED_ALLOC_EXP,
                           "urEnqueueUSMSharedAllocExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_USM_SHARED_ALLOC_EXP, &params);
    logger.info("   <--- urEnqueueUSMSharedAllocExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMHostAllocExp
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMHostAllocExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][optional] USM pool descriptor
    ur_usm_pool_handle_t pPool,
    /// [in] minimum size in bytes of the USM memory object to be allocated
    const size_t size,
    /// [in][optional] pointer to the enqueue async alloc properties
    const ur_exp_async_usm_alloc_properties_t *pProperties,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating no wait
    /// events.
    const ur_event_handle_t *phEventWaitList,
    /// [out] pointer to USM memory object
    void **ppMem,
    /// [out][optional] return an event object that identifies the async alloc
    ur_event_handle_t *phEvent) {
  auto pfnUSMHostAllocExp =
      getContext()->urDdiTable.EnqueueExp.pfnUSMHostAllocExp;

  if (nullptr == pfnUSMHostAllocExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_host_alloc_exp_params_t params = {
      &hQueue,          &pPool, &size,   &pProperties, &numEventsInWaitList,
      &phEventWaitList, &ppMem, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_USM_HOST_ALLOC_EXP,
                                 "urEnqueueUSMHostAllocExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMHostAllocExp\n");

  ur_result_t result =
      pfnUSMHostAllocExp(hQueue, pPool, size, pProperties, numEventsInWaitList,
                         phEventWaitList, ppMem, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_HOST_ALLOC_EXP,
                           "urEnqueueUSMHostAllocExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_USM_HOST_ALLOC_EXP, &params);
    logger.info("   <--- urEnqueueUSMHostAllocExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueUSMFreeExp
__urdlllocal ur_result_t UR_APICALL urEnqueueUSMFreeExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][optional] USM pool descriptor
    ur_usm_pool_handle_t pPool,
    /// [in] pointer to USM memory object
    void *pMem,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating no wait
    /// events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies the async alloc
    ur_event_handle_t *phEvent) {
  auto pfnUSMFreeExp = getContext()->urDdiTable.EnqueueExp.pfnUSMFreeExp;

  if (nullptr == pfnUSMFreeExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_usm_free_exp_params_t params = {
      &hQueue, &pPool, &pMem, &numEventsInWaitList, &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_USM_FREE_EXP, "urEnqueueUSMFreeExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueUSMFreeExp\n");

  ur_result_t result = pfnUSMFreeExp(hQueue, pPool, pMem, numEventsInWaitList,
                                     phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_USM_FREE_EXP,
                           "urEnqueueUSMFreeExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_ENQUEUE_USM_FREE_EXP,
                                    &params);
    logger.info("   <--- urEnqueueUSMFreeExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolCreateExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolCreateExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] pointer to USM pool descriptor. Can be chained with
    /// ::ur_usm_pool_limits_desc_t
    ur_usm_pool_desc_t *pPoolDesc,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *pPool) {
  auto pfnPoolCreateExp = getContext()->urDdiTable.USMExp.pfnPoolCreateExp;

  if (nullptr == pfnPoolCreateExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_create_exp_params_t params = {&hContext, &hDevice, &pPoolDesc,
                                            &pPool};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_USM_POOL_CREATE_EXP, "urUSMPoolCreateExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolCreateExp\n");

  ur_result_t result = pfnPoolCreateExp(hContext, hDevice, pPoolDesc, pPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_CREATE_EXP,
                           "urUSMPoolCreateExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_CREATE_EXP,
                                    &params);
    logger.info("   <--- urUSMPoolCreateExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolDestroyExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolDestroyExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to USM memory pool to be destroyed
    ur_usm_pool_handle_t hPool) {
  auto pfnPoolDestroyExp = getContext()->urDdiTable.USMExp.pfnPoolDestroyExp;

  if (nullptr == pfnPoolDestroyExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_destroy_exp_params_t params = {&hContext, &hDevice, &hPool};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_USM_POOL_DESTROY_EXP, "urUSMPoolDestroyExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolDestroyExp\n");

  ur_result_t result = pfnPoolDestroyExp(hContext, hDevice, hPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_DESTROY_EXP,
                           "urUSMPoolDestroyExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_DESTROY_EXP,
                                    &params);
    logger.info("   <--- urUSMPoolDestroyExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolSetThresholdExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolSetThresholdExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to USM memory pool for the threshold to be set
    ur_usm_pool_handle_t hPool,
    /// [in] release threshold to be set
    size_t newThreshold) {
  auto pfnPoolSetThresholdExp =
      getContext()->urDdiTable.USMExp.pfnPoolSetThresholdExp;

  if (nullptr == pfnPoolSetThresholdExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_set_threshold_exp_params_t params = {&hContext, &hDevice, &hPool,
                                                   &newThreshold};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_USM_POOL_SET_THRESHOLD_EXP,
                                 "urUSMPoolSetThresholdExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolSetThresholdExp\n");

  ur_result_t result =
      pfnPoolSetThresholdExp(hContext, hDevice, hPool, newThreshold);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_SET_THRESHOLD_EXP,
                           "urUSMPoolSetThresholdExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_POOL_SET_THRESHOLD_EXP, &params);
    logger.info("   <--- urUSMPoolSetThresholdExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolGetDefaultDevicePoolExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *pPool) {
  auto pfnPoolGetDefaultDevicePoolExp =
      getContext()->urDdiTable.USMExp.pfnPoolGetDefaultDevicePoolExp;

  if (nullptr == pfnPoolGetDefaultDevicePoolExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_get_default_device_pool_exp_params_t params = {&hContext,
                                                             &hDevice, &pPool};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_USM_POOL_GET_DEFAULT_DEVICE_POOL_EXP,
      "urUSMPoolGetDefaultDevicePoolExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolGetDefaultDevicePoolExp\n");

  ur_result_t result = pfnPoolGetDefaultDevicePoolExp(hContext, hDevice, pPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_GET_DEFAULT_DEVICE_POOL_EXP,
                           "urUSMPoolGetDefaultDevicePoolExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_POOL_GET_DEFAULT_DEVICE_POOL_EXP, &params);
    logger.info("   <--- urUSMPoolGetDefaultDevicePoolExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolGetInfoExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolGetInfoExp(
    /// [in] handle to USM memory pool for property retrieval
    ur_usm_pool_handle_t hPool,
    /// [in] queried property name
    ur_usm_pool_info_t propName,
    /// [out][optional] returned query value
    void *pPropValue,
    /// [out][optional] returned query value size
    size_t *pPropSizeRet) {
  auto pfnPoolGetInfoExp = getContext()->urDdiTable.USMExp.pfnPoolGetInfoExp;

  if (nullptr == pfnPoolGetInfoExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_get_info_exp_params_t params = {&hPool, &propName, &pPropValue,
                                              &pPropSizeRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_USM_POOL_GET_INFO_EXP, "urUSMPoolGetInfoExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolGetInfoExp\n");

  ur_result_t result =
      pfnPoolGetInfoExp(hPool, propName, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_GET_INFO_EXP,
                           "urUSMPoolGetInfoExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_GET_INFO_EXP,
                                    &params);
    logger.info("   <--- urUSMPoolGetInfoExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolSetDevicePoolExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to USM memory pool to set for a device
    ur_usm_pool_handle_t hPool) {
  auto pfnPoolSetDevicePoolExp =
      getContext()->urDdiTable.USMExp.pfnPoolSetDevicePoolExp;

  if (nullptr == pfnPoolSetDevicePoolExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_set_device_pool_exp_params_t params = {&hContext, &hDevice,
                                                     &hPool};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_USM_POOL_SET_DEVICE_POOL_EXP,
                                 "urUSMPoolSetDevicePoolExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolSetDevicePoolExp\n");

  ur_result_t result = pfnPoolSetDevicePoolExp(hContext, hDevice, hPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_SET_DEVICE_POOL_EXP,
                           "urUSMPoolSetDevicePoolExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_POOL_SET_DEVICE_POOL_EXP, &params);
    logger.info("   <--- urUSMPoolSetDevicePoolExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolGetDevicePoolExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *pPool) {
  auto pfnPoolGetDevicePoolExp =
      getContext()->urDdiTable.USMExp.pfnPoolGetDevicePoolExp;

  if (nullptr == pfnPoolGetDevicePoolExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_get_device_pool_exp_params_t params = {&hContext, &hDevice,
                                                     &pPool};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_USM_POOL_GET_DEVICE_POOL_EXP,
                                 "urUSMPoolGetDevicePoolExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolGetDevicePoolExp\n");

  ur_result_t result = pfnPoolGetDevicePoolExp(hContext, hDevice, pPool);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_GET_DEVICE_POOL_EXP,
                           "urUSMPoolGetDevicePoolExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_POOL_GET_DEVICE_POOL_EXP, &params);
    logger.info("   <--- urUSMPoolGetDevicePoolExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolTrimToExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolTrimToExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to USM memory pool for trimming
    ur_usm_pool_handle_t hPool,
    /// [in] minimum number of bytes to keep in the pool
    size_t minBytesToKeep) {
  auto pfnPoolTrimToExp = getContext()->urDdiTable.USMExp.pfnPoolTrimToExp;

  if (nullptr == pfnPoolTrimToExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pool_trim_to_exp_params_t params = {&hContext, &hDevice, &hPool,
                                             &minBytesToKeep};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_USM_POOL_TRIM_TO_EXP, "urUSMPoolTrimToExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPoolTrimToExp\n");

  ur_result_t result =
      pfnPoolTrimToExp(hContext, hDevice, hPool, minBytesToKeep);

  getContext()->notify_end(UR_FUNCTION_USM_POOL_TRIM_TO_EXP,
                           "urUSMPoolTrimToExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_POOL_TRIM_TO_EXP,
                                    &params);
    logger.info("   <--- urUSMPoolTrimToExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPitchedAllocExp
__urdlllocal ur_result_t UR_APICALL urUSMPitchedAllocExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][optional] Pointer to USM memory allocation descriptor.
    const ur_usm_desc_t *pUSMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t pool,
    /// [in] width in bytes of the USM memory object to be allocated
    size_t widthInBytes,
    /// [in] height of the USM memory object to be allocated
    size_t height,
    /// [in] size in bytes of an element in the allocation
    size_t elementSizeBytes,
    /// [out] pointer to USM shared memory object
    void **ppMem,
    /// [out] pitch of the allocation
    size_t *pResultPitch) {
  auto pfnPitchedAllocExp = getContext()->urDdiTable.USMExp.pfnPitchedAllocExp;

  if (nullptr == pfnPitchedAllocExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_pitched_alloc_exp_params_t params = {
      &hContext, &hDevice,          &pUSMDesc, &pool,        &widthInBytes,
      &height,   &elementSizeBytes, &ppMem,    &pResultPitch};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_USM_PITCHED_ALLOC_EXP, "urUSMPitchedAllocExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMPitchedAllocExp\n");

  ur_result_t result =
      pfnPitchedAllocExp(hContext, hDevice, pUSMDesc, pool, widthInBytes,
                         height, elementSizeBytes, ppMem, pResultPitch);

  getContext()->notify_end(UR_FUNCTION_USM_PITCHED_ALLOC_EXP,
                           "urUSMPitchedAllocExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_PITCHED_ALLOC_EXP,
                                    &params);
    logger.info("   <--- urUSMPitchedAllocExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesUnsampledImageHandleDestroyExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] pointer to handle of image object to destroy
    ur_exp_image_native_handle_t hImage) {
  auto pfnUnsampledImageHandleDestroyExp =
      getContext()
          ->urDdiTable.BindlessImagesExp.pfnUnsampledImageHandleDestroyExp;

  if (nullptr == pfnUnsampledImageHandleDestroyExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_unsampled_image_handle_destroy_exp_params_t params = {
      &hContext, &hDevice, &hImage};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_HANDLE_DESTROY_EXP,
      "urBindlessImagesUnsampledImageHandleDestroyExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesUnsampledImageHandleDestroyExp\n");

  ur_result_t result =
      pfnUnsampledImageHandleDestroyExp(hContext, hDevice, hImage);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_HANDLE_DESTROY_EXP,
      "urBindlessImagesUnsampledImageHandleDestroyExp", &params, &result,
      instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str,
        UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_HANDLE_DESTROY_EXP,
        &params);
    logger.info(
        "   <--- urBindlessImagesUnsampledImageHandleDestroyExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSampledImageHandleDestroyExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] pointer to handle of image object to destroy
    ur_exp_image_native_handle_t hImage) {
  auto pfnSampledImageHandleDestroyExp =
      getContext()
          ->urDdiTable.BindlessImagesExp.pfnSampledImageHandleDestroyExp;

  if (nullptr == pfnSampledImageHandleDestroyExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_sampled_image_handle_destroy_exp_params_t params = {
      &hContext, &hDevice, &hImage};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_HANDLE_DESTROY_EXP,
      "urBindlessImagesSampledImageHandleDestroyExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesSampledImageHandleDestroyExp\n");

  ur_result_t result =
      pfnSampledImageHandleDestroyExp(hContext, hDevice, hImage);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_HANDLE_DESTROY_EXP,
      "urBindlessImagesSampledImageHandleDestroyExp", &params, &result,
      instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_HANDLE_DESTROY_EXP,
        &params);
    logger.info(
        "   <--- urBindlessImagesSampledImageHandleDestroyExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageAllocateExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [out][alloc] pointer to handle of image memory allocated
    ur_exp_image_mem_native_handle_t *phImageMem) {
  auto pfnImageAllocateExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnImageAllocateExp;

  if (nullptr == pfnImageAllocateExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_image_allocate_exp_params_t params = {
      &hContext, &hDevice, &pImageFormat, &pImageDesc, &phImageMem};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_ALLOCATE_EXP,
                                 "urBindlessImagesImageAllocateExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesImageAllocateExp\n");

  ur_result_t result = pfnImageAllocateExp(hContext, hDevice, pImageFormat,
                                           pImageDesc, phImageMem);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_ALLOCATE_EXP,
                           "urBindlessImagesImageAllocateExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_IMAGE_ALLOCATE_EXP, &params);
    logger.info("   <--- urBindlessImagesImageAllocateExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of image memory to be freed
    ur_exp_image_mem_native_handle_t hImageMem) {
  auto pfnImageFreeExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnImageFreeExp;

  if (nullptr == pfnImageFreeExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_image_free_exp_params_t params = {&hContext, &hDevice,
                                                       &hImageMem};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_FREE_EXP,
                                 "urBindlessImagesImageFreeExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesImageFreeExp\n");

  ur_result_t result = pfnImageFreeExp(hContext, hDevice, hImageMem);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_FREE_EXP,
                           "urBindlessImagesImageFreeExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_IMAGE_FREE_EXP, &params);
    logger.info("   <--- urBindlessImagesImageFreeExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesUnsampledImageCreateExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to memory from which to create the image
    ur_exp_image_mem_native_handle_t hImageMem,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [out][alloc] pointer to handle of image object created
    ur_exp_image_native_handle_t *phImage) {
  auto pfnUnsampledImageCreateExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnUnsampledImageCreateExp;

  if (nullptr == pfnUnsampledImageCreateExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_unsampled_image_create_exp_params_t params = {
      &hContext, &hDevice, &hImageMem, &pImageFormat, &pImageDesc, &phImage};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_CREATE_EXP,
      "urBindlessImagesUnsampledImageCreateExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesUnsampledImageCreateExp\n");

  ur_result_t result = pfnUnsampledImageCreateExp(
      hContext, hDevice, hImageMem, pImageFormat, pImageDesc, phImage);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_CREATE_EXP,
      "urBindlessImagesUnsampledImageCreateExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_CREATE_EXP,
        &params);
    logger.info("   <--- urBindlessImagesUnsampledImageCreateExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSampledImageCreateExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to memory from which to create the image
    ur_exp_image_mem_native_handle_t hImageMem,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [in] sampler to be used
    ur_sampler_handle_t hSampler,
    /// [out][alloc] pointer to handle of image object created
    ur_exp_image_native_handle_t *phImage) {
  auto pfnSampledImageCreateExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnSampledImageCreateExp;

  if (nullptr == pfnSampledImageCreateExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_sampled_image_create_exp_params_t params = {
      &hContext,   &hDevice,  &hImageMem, &pImageFormat,
      &pImageDesc, &hSampler, &phImage};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_CREATE_EXP,
      "urBindlessImagesSampledImageCreateExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesSampledImageCreateExp\n");

  ur_result_t result =
      pfnSampledImageCreateExp(hContext, hDevice, hImageMem, pImageFormat,
                               pImageDesc, hSampler, phImage);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_CREATE_EXP,
                           "urBindlessImagesSampledImageCreateExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_CREATE_EXP,
        &params);
    logger.info("   <--- urBindlessImagesSampledImageCreateExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageCopyExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] location the data will be copied from
    const void *pSrc,
    /// [in] location the data will be copied to
    void *pDst,
    /// [in] pointer to image description
    const ur_image_desc_t *pSrcImageDesc,
    /// [in] pointer to image description
    const ur_image_desc_t *pDstImageDesc,
    /// [in] pointer to image format specification
    const ur_image_format_t *pSrcImageFormat,
    /// [in] pointer to image format specification
    const ur_image_format_t *pDstImageFormat,
    /// [in] Pointer to structure describing the (sub-)regions of source and
    /// destination images
    ur_exp_image_copy_region_t *pCopyRegion,
    /// [in] flags describing copy direction e.g. H2D or D2H
    ur_exp_image_copy_flags_t imageCopyFlags,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands
    /// must be complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnImageCopyExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnImageCopyExp;

  if (nullptr == pfnImageCopyExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_image_copy_exp_params_t params = {&hQueue,
                                                       &pSrc,
                                                       &pDst,
                                                       &pSrcImageDesc,
                                                       &pDstImageDesc,
                                                       &pSrcImageFormat,
                                                       &pDstImageFormat,
                                                       &pCopyRegion,
                                                       &imageCopyFlags,
                                                       &numEventsInWaitList,
                                                       &phEventWaitList,
                                                       &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_COPY_EXP,
                                 "urBindlessImagesImageCopyExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesImageCopyExp\n");

  ur_result_t result = pfnImageCopyExp(
      hQueue, pSrc, pDst, pSrcImageDesc, pDstImageDesc, pSrcImageFormat,
      pDstImageFormat, pCopyRegion, imageCopyFlags, numEventsInWaitList,
      phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_COPY_EXP,
                           "urBindlessImagesImageCopyExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_IMAGE_COPY_EXP, &params);
    logger.info("   <--- urBindlessImagesImageCopyExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageGetInfoExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle to the image memory
    ur_exp_image_mem_native_handle_t hImageMem,
    /// [in] queried info name
    ur_image_info_t propName,
    /// [out][optional] returned query value
    void *pPropValue,
    /// [out][optional] returned query value size
    size_t *pPropSizeRet) {
  auto pfnImageGetInfoExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnImageGetInfoExp;

  if (nullptr == pfnImageGetInfoExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_image_get_info_exp_params_t params = {
      &hContext, &hImageMem, &propName, &pPropValue, &pPropSizeRet};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_GET_INFO_EXP,
                                 "urBindlessImagesImageGetInfoExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesImageGetInfoExp\n");

  ur_result_t result = pfnImageGetInfoExp(hContext, hImageMem, propName,
                                          pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_IMAGE_GET_INFO_EXP,
                           "urBindlessImagesImageGetInfoExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_IMAGE_GET_INFO_EXP, &params);
    logger.info("   <--- urBindlessImagesImageGetInfoExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMipmapGetLevelExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] memory handle to the mipmap image
    ur_exp_image_mem_native_handle_t hImageMem,
    /// [in] requested level of the mipmap
    uint32_t mipmapLevel,
    /// [out] returning memory handle to the individual image
    ur_exp_image_mem_native_handle_t *phImageMem) {
  auto pfnMipmapGetLevelExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnMipmapGetLevelExp;

  if (nullptr == pfnMipmapGetLevelExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_mipmap_get_level_exp_params_t params = {
      &hContext, &hDevice, &hImageMem, &mipmapLevel, &phImageMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_GET_LEVEL_EXP,
      "urBindlessImagesMipmapGetLevelExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesMipmapGetLevelExp\n");

  ur_result_t result = pfnMipmapGetLevelExp(hContext, hDevice, hImageMem,
                                            mipmapLevel, phImageMem);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_GET_LEVEL_EXP,
                           "urBindlessImagesMipmapGetLevelExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_GET_LEVEL_EXP, &params);
    logger.info("   <--- urBindlessImagesMipmapGetLevelExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMipmapFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of image memory to be freed
    ur_exp_image_mem_native_handle_t hMem) {
  auto pfnMipmapFreeExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnMipmapFreeExp;

  if (nullptr == pfnMipmapFreeExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_mipmap_free_exp_params_t params = {&hContext, &hDevice,
                                                        &hMem};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_FREE_EXP,
                                 "urBindlessImagesMipmapFreeExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesMipmapFreeExp\n");

  ur_result_t result = pfnMipmapFreeExp(hContext, hDevice, hMem);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_FREE_EXP,
                           "urBindlessImagesMipmapFreeExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_FREE_EXP, &params);
    logger.info("   <--- urBindlessImagesMipmapFreeExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImportExternalMemoryExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImportExternalMemoryExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] size of the external memory
    size_t size,
    /// [in] type of external memory handle
    ur_exp_external_mem_type_t memHandleType,
    /// [in] the external memory descriptor
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    /// [out][alloc] external memory handle to the external memory
    ur_exp_external_mem_handle_t *phExternalMem) {
  auto pfnImportExternalMemoryExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnImportExternalMemoryExp;

  if (nullptr == pfnImportExternalMemoryExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_import_external_memory_exp_params_t params = {
      &hContext,      &hDevice,          &size,
      &memHandleType, &pExternalMemDesc, &phExternalMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_MEMORY_EXP,
      "urBindlessImagesImportExternalMemoryExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesImportExternalMemoryExp\n");

  ur_result_t result = pfnImportExternalMemoryExp(
      hContext, hDevice, size, memHandleType, pExternalMemDesc, phExternalMem);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_MEMORY_EXP,
      "urBindlessImagesImportExternalMemoryExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_MEMORY_EXP,
        &params);
    logger.info("   <--- urBindlessImagesImportExternalMemoryExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMapExternalArrayExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [in] external memory handle to the external memory
    ur_exp_external_mem_handle_t hExternalMem,
    /// [out] image memory handle to the externally allocated memory
    ur_exp_image_mem_native_handle_t *phImageMem) {
  auto pfnMapExternalArrayExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnMapExternalArrayExp;

  if (nullptr == pfnMapExternalArrayExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_map_external_array_exp_params_t params = {
      &hContext,   &hDevice,      &pImageFormat,
      &pImageDesc, &hExternalMem, &phImageMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_ARRAY_EXP,
      "urBindlessImagesMapExternalArrayExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesMapExternalArrayExp\n");

  ur_result_t result = pfnMapExternalArrayExp(
      hContext, hDevice, pImageFormat, pImageDesc, hExternalMem, phImageMem);

  getContext()->notify_end(UR_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_ARRAY_EXP,
                           "urBindlessImagesMapExternalArrayExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_ARRAY_EXP, &params);
    logger.info("   <--- urBindlessImagesMapExternalArrayExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMapExternalLinearMemoryExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMapExternalLinearMemoryExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] offset into memory region to map
    uint64_t offset,
    /// [in] size of memory region to map
    uint64_t size,
    /// [in] external memory handle to the external memory
    ur_exp_external_mem_handle_t hExternalMem,
    /// [out] pointer of the externally allocated memory
    void **ppRetMem) {
  auto pfnMapExternalLinearMemoryExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnMapExternalLinearMemoryExp;

  if (nullptr == pfnMapExternalLinearMemoryExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_map_external_linear_memory_exp_params_t params = {
      &hContext, &hDevice, &offset, &size, &hExternalMem, &ppRetMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_LINEAR_MEMORY_EXP,
      "urBindlessImagesMapExternalLinearMemoryExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesMapExternalLinearMemoryExp\n");

  ur_result_t result = pfnMapExternalLinearMemoryExp(
      hContext, hDevice, offset, size, hExternalMem, ppRetMem);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_LINEAR_MEMORY_EXP,
      "urBindlessImagesMapExternalLinearMemoryExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_LINEAR_MEMORY_EXP,
        &params);
    logger.info(
        "   <--- urBindlessImagesMapExternalLinearMemoryExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesReleaseExternalMemoryExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesReleaseExternalMemoryExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of external memory to be destroyed
    ur_exp_external_mem_handle_t hExternalMem) {
  auto pfnReleaseExternalMemoryExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnReleaseExternalMemoryExp;

  if (nullptr == pfnReleaseExternalMemoryExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_release_external_memory_exp_params_t params = {
      &hContext, &hDevice, &hExternalMem};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_RELEASE_EXTERNAL_MEMORY_EXP,
      "urBindlessImagesReleaseExternalMemoryExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesReleaseExternalMemoryExp\n");

  ur_result_t result =
      pfnReleaseExternalMemoryExp(hContext, hDevice, hExternalMem);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_RELEASE_EXTERNAL_MEMORY_EXP,
      "urBindlessImagesReleaseExternalMemoryExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_RELEASE_EXTERNAL_MEMORY_EXP,
        &params);
    logger.info("   <--- urBindlessImagesReleaseExternalMemoryExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImportExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImportExternalSemaphoreExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] type of external memory handle
    ur_exp_external_semaphore_type_t semHandleType,
    /// [in] the external semaphore descriptor
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    /// [out][alloc] external semaphore handle to the external semaphore
    ur_exp_external_semaphore_handle_t *phExternalSemaphore) {
  auto pfnImportExternalSemaphoreExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnImportExternalSemaphoreExp;

  if (nullptr == pfnImportExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_import_external_semaphore_exp_params_t params = {
      &hContext, &hDevice, &semHandleType, &pExternalSemaphoreDesc,
      &phExternalSemaphore};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesImportExternalSemaphoreExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesImportExternalSemaphoreExp\n");

  ur_result_t result = pfnImportExternalSemaphoreExp(
      hContext, hDevice, semHandleType, pExternalSemaphoreDesc,
      phExternalSemaphore);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesImportExternalSemaphoreExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_SEMAPHORE_EXP,
        &params);
    logger.info(
        "   <--- urBindlessImagesImportExternalSemaphoreExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesReleaseExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesReleaseExternalSemaphoreExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of external semaphore to be destroyed
    ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  auto pfnReleaseExternalSemaphoreExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnReleaseExternalSemaphoreExp;

  if (nullptr == pfnReleaseExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_release_external_semaphore_exp_params_t params = {
      &hContext, &hDevice, &hExternalSemaphore};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_RELEASE_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesReleaseExternalSemaphoreExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesReleaseExternalSemaphoreExp\n");

  ur_result_t result =
      pfnReleaseExternalSemaphoreExp(hContext, hDevice, hExternalSemaphore);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_RELEASE_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesReleaseExternalSemaphoreExp", &params, &result,
      instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_RELEASE_EXTERNAL_SEMAPHORE_EXP,
        &params);
    logger.info(
        "   <--- urBindlessImagesReleaseExternalSemaphoreExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesWaitExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] external semaphore handle
    ur_exp_external_semaphore_handle_t hSemaphore,
    /// [in] indicates whether the samephore is capable and should wait on a
    /// certain value.
    /// Otherwise the semaphore is treated like a binary state, and
    /// `waitValue` is ignored.
    bool hasWaitValue,
    /// [in] the value to be waited on
    uint64_t waitValue,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands
    /// must be complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnWaitExternalSemaphoreExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnWaitExternalSemaphoreExp;

  if (nullptr == pfnWaitExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_wait_external_semaphore_exp_params_t params = {
      &hQueue,    &hSemaphore,          &hasWaitValue,
      &waitValue, &numEventsInWaitList, &phEventWaitList,
      &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_WAIT_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesWaitExternalSemaphoreExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesWaitExternalSemaphoreExp\n");

  ur_result_t result = pfnWaitExternalSemaphoreExp(
      hQueue, hSemaphore, hasWaitValue, waitValue, numEventsInWaitList,
      phEventWaitList, phEvent);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_WAIT_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesWaitExternalSemaphoreExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_WAIT_EXTERNAL_SEMAPHORE_EXP,
        &params);
    logger.info("   <--- urBindlessImagesWaitExternalSemaphoreExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSignalExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] external semaphore handle
    ur_exp_external_semaphore_handle_t hSemaphore,
    /// [in] indicates whether the samephore is capable and should signal on a
    /// certain value.
    /// Otherwise the semaphore is treated like a binary state, and
    /// `signalValue` is ignored.
    bool hasSignalValue,
    /// [in] the value to be signalled
    uint64_t signalValue,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands
    /// must be complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnSignalExternalSemaphoreExp =
      getContext()->urDdiTable.BindlessImagesExp.pfnSignalExternalSemaphoreExp;

  if (nullptr == pfnSignalExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_bindless_images_signal_external_semaphore_exp_params_t params = {
      &hQueue,      &hSemaphore,          &hasSignalValue,
      &signalValue, &numEventsInWaitList, &phEventWaitList,
      &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_BINDLESS_IMAGES_SIGNAL_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesSignalExternalSemaphoreExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urBindlessImagesSignalExternalSemaphoreExp\n");

  ur_result_t result = pfnSignalExternalSemaphoreExp(
      hQueue, hSemaphore, hasSignalValue, signalValue, numEventsInWaitList,
      phEventWaitList, phEvent);

  getContext()->notify_end(
      UR_FUNCTION_BINDLESS_IMAGES_SIGNAL_EXTERNAL_SEMAPHORE_EXP,
      "urBindlessImagesSignalExternalSemaphoreExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_BINDLESS_IMAGES_SIGNAL_EXTERNAL_SEMAPHORE_EXP,
        &params);
    logger.info(
        "   <--- urBindlessImagesSignalExternalSemaphoreExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferCreateExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferCreateExp(
    /// [in] Handle of the context object.
    ur_context_handle_t hContext,
    /// [in] Handle of the device object.
    ur_device_handle_t hDevice,
    /// [in] Command-buffer descriptor.
    const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    /// [out][alloc] Pointer to command-Buffer handle.
    ur_exp_command_buffer_handle_t *phCommandBuffer) {
  auto pfnCreateExp = getContext()->urDdiTable.CommandBufferExp.pfnCreateExp;

  if (nullptr == pfnCreateExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_create_exp_params_t params = {
      &hContext, &hDevice, &pCommandBufferDesc, &phCommandBuffer};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_COMMAND_BUFFER_CREATE_EXP,
                                 "urCommandBufferCreateExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferCreateExp\n");

  ur_result_t result =
      pfnCreateExp(hContext, hDevice, pCommandBufferDesc, phCommandBuffer);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_CREATE_EXP,
                           "urCommandBufferCreateExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_CREATE_EXP, &params);
    logger.info("   <--- urCommandBufferCreateExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainExp(
    /// [in][retain] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) {
  auto pfnRetainExp = getContext()->urDdiTable.CommandBufferExp.pfnRetainExp;

  if (nullptr == pfnRetainExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_retain_exp_params_t params = {&hCommandBuffer};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_COMMAND_BUFFER_RETAIN_EXP,
                                 "urCommandBufferRetainExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferRetainExp\n");

  ur_result_t result = pfnRetainExp(hCommandBuffer);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_RETAIN_EXP,
                           "urCommandBufferRetainExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_RETAIN_EXP, &params);
    logger.info("   <--- urCommandBufferRetainExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseExp(
    /// [in][release] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) {
  auto pfnReleaseExp = getContext()->urDdiTable.CommandBufferExp.pfnReleaseExp;

  if (nullptr == pfnReleaseExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_release_exp_params_t params = {&hCommandBuffer};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_COMMAND_BUFFER_RELEASE_EXP,
                                 "urCommandBufferReleaseExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferReleaseExp\n");

  ur_result_t result = pfnReleaseExp(hCommandBuffer);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_RELEASE_EXP,
                           "urCommandBufferReleaseExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_RELEASE_EXP, &params);
    logger.info("   <--- urCommandBufferReleaseExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferFinalizeExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) {
  auto pfnFinalizeExp =
      getContext()->urDdiTable.CommandBufferExp.pfnFinalizeExp;

  if (nullptr == pfnFinalizeExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_finalize_exp_params_t params = {&hCommandBuffer};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_COMMAND_BUFFER_FINALIZE_EXP,
                                 "urCommandBufferFinalizeExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferFinalizeExp\n");

  ur_result_t result = pfnFinalizeExp(hCommandBuffer);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_FINALIZE_EXP,
                           "urCommandBufferFinalizeExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_FINALIZE_EXP, &params);
    logger.info("   <--- urCommandBufferFinalizeExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Kernel to append.
    ur_kernel_handle_t hKernel,
    /// [in] Dimension of the kernel execution.
    uint32_t workDim,
    /// [in] Offset to use when executing kernel.
    const size_t *pGlobalWorkOffset,
    /// [in] Global work size to use when executing kernel.
    const size_t *pGlobalWorkSize,
    /// [in][optional] Local work size to use when executing kernel. If this
    /// parameter is nullptr, then a local work size will be generated by the
    /// implementation.
    const size_t *pLocalWorkSize,
    /// [in] The number of kernel alternatives provided in
    /// phKernelAlternatives.
    uint32_t numKernelAlternatives,
    /// [in][optional][range(0, numKernelAlternatives)] List of kernel handles
    /// that might be used to update the kernel in this
    /// command after the command-buffer is finalized. The default kernel
    /// `hKernel` is implicitly marked as an alternative. It's
    /// invalid to specify it as part of this list.
    ur_kernel_handle_t *phKernelAlternatives,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command. Only available if the
    /// command-buffer is updatable.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendKernelLaunchExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendKernelLaunchExp;

  if (nullptr == pfnAppendKernelLaunchExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_kernel_launch_exp_params_t params = {
      &hCommandBuffer,
      &hKernel,
      &workDim,
      &pGlobalWorkOffset,
      &pGlobalWorkSize,
      &pLocalWorkSize,
      &numKernelAlternatives,
      &phKernelAlternatives,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP,
      "urCommandBufferAppendKernelLaunchExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendKernelLaunchExp\n");

  ur_result_t result = pfnAppendKernelLaunchExp(
      hCommandBuffer, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numKernelAlternatives, phKernelAlternatives,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP,
                           "urCommandBufferAppendKernelLaunchExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP, &params);
    logger.info("   <--- urCommandBufferAppendKernelLaunchExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMMemcpyExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Location the data will be copied to.
    void *pDst,
    /// [in] The data to be copied.
    const void *pSrc,
    /// [in] The number of bytes to copy.
    size_t size,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendUSMMemcpyExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendUSMMemcpyExp;

  if (nullptr == pfnAppendUSMMemcpyExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_usm_memcpy_exp_params_t params = {
      &hCommandBuffer,
      &pDst,
      &pSrc,
      &size,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_MEMCPY_EXP,
      "urCommandBufferAppendUSMMemcpyExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendUSMMemcpyExp\n");

  ur_result_t result = pfnAppendUSMMemcpyExp(
      hCommandBuffer, pDst, pSrc, size, numSyncPointsInWaitList,
      pSyncPointWaitList, numEventsInWaitList, phEventWaitList, pSyncPoint,
      phEvent, phCommand);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_MEMCPY_EXP,
                           "urCommandBufferAppendUSMMemcpyExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_MEMCPY_EXP, &params);
    logger.info("   <--- urCommandBufferAppendUSMMemcpyExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMFillExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    /// [in] handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] pointer to USM allocated memory to fill.
    void *pMemory,
    /// [in] pointer to the fill pattern.
    const void *pPattern,
    /// [in] size in bytes of the pattern.
    size_t patternSize,
    /// [in] fill size in bytes, must be a multiple of patternSize.
    size_t size,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendUSMFillExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendUSMFillExp;

  if (nullptr == pfnAppendUSMFillExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_usm_fill_exp_params_t params = {
      &hCommandBuffer,
      &pMemory,
      &pPattern,
      &patternSize,
      &size,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_FILL_EXP,
                                 "urCommandBufferAppendUSMFillExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendUSMFillExp\n");

  ur_result_t result = pfnAppendUSMFillExp(
      hCommandBuffer, pMemory, pPattern, patternSize, size,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_FILL_EXP,
                           "urCommandBufferAppendUSMFillExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_FILL_EXP, &params);
    logger.info("   <--- urCommandBufferAppendUSMFillExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferCopyExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] The data to be copied.
    ur_mem_handle_t hSrcMem,
    /// [in] The location the data will be copied to.
    ur_mem_handle_t hDstMem,
    /// [in] Offset into the source memory.
    size_t srcOffset,
    /// [in] Offset into the destination memory
    size_t dstOffset,
    /// [in] The number of bytes to be copied.
    size_t size,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendMemBufferCopyExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendMemBufferCopyExp;

  if (nullptr == pfnAppendMemBufferCopyExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_mem_buffer_copy_exp_params_t params = {
      &hCommandBuffer,
      &hSrcMem,
      &hDstMem,
      &srcOffset,
      &dstOffset,
      &size,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_EXP,
      "urCommandBufferAppendMemBufferCopyExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendMemBufferCopyExp\n");

  ur_result_t result = pfnAppendMemBufferCopyExp(
      hCommandBuffer, hSrcMem, hDstMem, srcOffset, dstOffset, size,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);

  getContext()->notify_end(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_EXP,
      "urCommandBufferAppendMemBufferCopyExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_EXP,
        &params);
    logger.info("   <--- urCommandBufferAppendMemBufferCopyExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferWriteExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Handle of the buffer object.
    ur_mem_handle_t hBuffer,
    /// [in] Offset in bytes in the buffer object.
    size_t offset,
    /// [in] Size in bytes of data being written.
    size_t size,
    /// [in] Pointer to host memory where data is to be written from.
    const void *pSrc,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendMemBufferWriteExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendMemBufferWriteExp;

  if (nullptr == pfnAppendMemBufferWriteExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_mem_buffer_write_exp_params_t params = {
      &hCommandBuffer,
      &hBuffer,
      &offset,
      &size,
      &pSrc,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_EXP,
      "urCommandBufferAppendMemBufferWriteExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendMemBufferWriteExp\n");

  ur_result_t result = pfnAppendMemBufferWriteExp(
      hCommandBuffer, hBuffer, offset, size, pSrc, numSyncPointsInWaitList,
      pSyncPointWaitList, numEventsInWaitList, phEventWaitList, pSyncPoint,
      phEvent, phCommand);

  getContext()->notify_end(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_EXP,
      "urCommandBufferAppendMemBufferWriteExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_EXP,
        &params);
    logger.info("   <--- urCommandBufferAppendMemBufferWriteExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferReadExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Handle of the buffer object.
    ur_mem_handle_t hBuffer,
    /// [in] Offset in bytes in the buffer object.
    size_t offset,
    /// [in] Size in bytes of data being written.
    size_t size,
    /// [in] Pointer to host memory where data is to be written to.
    void *pDst,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendMemBufferReadExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendMemBufferReadExp;

  if (nullptr == pfnAppendMemBufferReadExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_mem_buffer_read_exp_params_t params = {
      &hCommandBuffer,
      &hBuffer,
      &offset,
      &size,
      &pDst,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_EXP,
      "urCommandBufferAppendMemBufferReadExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendMemBufferReadExp\n");

  ur_result_t result = pfnAppendMemBufferReadExp(
      hCommandBuffer, hBuffer, offset, size, pDst, numSyncPointsInWaitList,
      pSyncPointWaitList, numEventsInWaitList, phEventWaitList, pSyncPoint,
      phEvent, phCommand);

  getContext()->notify_end(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_EXP,
      "urCommandBufferAppendMemBufferReadExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_EXP,
        &params);
    logger.info("   <--- urCommandBufferAppendMemBufferReadExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferCopyRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] The data to be copied.
    ur_mem_handle_t hSrcMem,
    /// [in] The location the data will be copied to.
    ur_mem_handle_t hDstMem,
    /// [in] Origin for the region of data to be copied from the source.
    ur_rect_offset_t srcOrigin,
    /// [in] Origin for the region of data to be copied to in the destination.
    ur_rect_offset_t dstOrigin,
    /// [in] The extents describing the region to be copied.
    ur_rect_region_t region,
    /// [in] Row pitch of the source memory.
    size_t srcRowPitch,
    /// [in] Slice pitch of the source memory.
    size_t srcSlicePitch,
    /// [in] Row pitch of the destination memory.
    size_t dstRowPitch,
    /// [in] Slice pitch of the destination memory.
    size_t dstSlicePitch,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendMemBufferCopyRectExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendMemBufferCopyRectExp;

  if (nullptr == pfnAppendMemBufferCopyRectExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_mem_buffer_copy_rect_exp_params_t params = {
      &hCommandBuffer,
      &hSrcMem,
      &hDstMem,
      &srcOrigin,
      &dstOrigin,
      &region,
      &srcRowPitch,
      &srcSlicePitch,
      &dstRowPitch,
      &dstSlicePitch,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_RECT_EXP,
      "urCommandBufferAppendMemBufferCopyRectExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendMemBufferCopyRectExp\n");

  ur_result_t result = pfnAppendMemBufferCopyRectExp(
      hCommandBuffer, hSrcMem, hDstMem, srcOrigin, dstOrigin, region,
      srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);

  getContext()->notify_end(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_RECT_EXP,
      "urCommandBufferAppendMemBufferCopyRectExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_RECT_EXP,
        &params);
    logger.info(
        "   <--- urCommandBufferAppendMemBufferCopyRectExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferWriteRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Handle of the buffer object.
    ur_mem_handle_t hBuffer,
    /// [in] 3D offset in the buffer.
    ur_rect_offset_t bufferOffset,
    /// [in] 3D offset in the host region.
    ur_rect_offset_t hostOffset,
    /// [in] 3D rectangular region descriptor: width, height, depth.
    ur_rect_region_t region,
    /// [in] Length of each row in bytes in the buffer object.
    size_t bufferRowPitch,
    /// [in] Length of each 2D slice in bytes in the buffer object being
    /// written.
    size_t bufferSlicePitch,
    /// [in] Length of each row in bytes in the host memory region pointed to
    /// by pSrc.
    size_t hostRowPitch,
    /// [in] Length of each 2D slice in bytes in the host memory region
    /// pointed to by pSrc.
    size_t hostSlicePitch,
    /// [in] Pointer to host memory where data is to be written from.
    void *pSrc,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendMemBufferWriteRectExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendMemBufferWriteRectExp;

  if (nullptr == pfnAppendMemBufferWriteRectExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_mem_buffer_write_rect_exp_params_t params = {
      &hCommandBuffer,
      &hBuffer,
      &bufferOffset,
      &hostOffset,
      &region,
      &bufferRowPitch,
      &bufferSlicePitch,
      &hostRowPitch,
      &hostSlicePitch,
      &pSrc,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_RECT_EXP,
      "urCommandBufferAppendMemBufferWriteRectExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendMemBufferWriteRectExp\n");

  ur_result_t result = pfnAppendMemBufferWriteRectExp(
      hCommandBuffer, hBuffer, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);

  getContext()->notify_end(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_RECT_EXP,
      "urCommandBufferAppendMemBufferWriteRectExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_RECT_EXP,
        &params);
    logger.info(
        "   <--- urCommandBufferAppendMemBufferWriteRectExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferReadRectExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Handle of the buffer object.
    ur_mem_handle_t hBuffer,
    /// [in] 3D offset in the buffer.
    ur_rect_offset_t bufferOffset,
    /// [in] 3D offset in the host region.
    ur_rect_offset_t hostOffset,
    /// [in] 3D rectangular region descriptor: width, height, depth.
    ur_rect_region_t region,
    /// [in] Length of each row in bytes in the buffer object.
    size_t bufferRowPitch,
    /// [in] Length of each 2D slice in bytes in the buffer object being read.
    size_t bufferSlicePitch,
    /// [in] Length of each row in bytes in the host memory region pointed to
    /// by pDst.
    size_t hostRowPitch,
    /// [in] Length of each 2D slice in bytes in the host memory region
    /// pointed to by pDst.
    size_t hostSlicePitch,
    /// [in] Pointer to host memory where data is to be read into.
    void *pDst,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional] return an event object that will be signaled by the
    /// completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendMemBufferReadRectExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendMemBufferReadRectExp;

  if (nullptr == pfnAppendMemBufferReadRectExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_mem_buffer_read_rect_exp_params_t params = {
      &hCommandBuffer,
      &hBuffer,
      &bufferOffset,
      &hostOffset,
      &region,
      &bufferRowPitch,
      &bufferSlicePitch,
      &hostRowPitch,
      &hostSlicePitch,
      &pDst,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_RECT_EXP,
      "urCommandBufferAppendMemBufferReadRectExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendMemBufferReadRectExp\n");

  ur_result_t result = pfnAppendMemBufferReadRectExp(
      hCommandBuffer, hBuffer, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);

  getContext()->notify_end(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_RECT_EXP,
      "urCommandBufferAppendMemBufferReadRectExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_RECT_EXP,
        &params);
    logger.info(
        "   <--- urCommandBufferAppendMemBufferReadRectExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendMemBufferFillExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    /// [in] handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] handle of the buffer object.
    ur_mem_handle_t hBuffer,
    /// [in] pointer to the fill pattern.
    const void *pPattern,
    /// [in] size in bytes of the pattern.
    size_t patternSize,
    /// [in] offset into the buffer.
    size_t offset,
    /// [in] fill size in bytes, must be a multiple of patternSize.
    size_t size,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendMemBufferFillExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendMemBufferFillExp;

  if (nullptr == pfnAppendMemBufferFillExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_mem_buffer_fill_exp_params_t params = {
      &hCommandBuffer,
      &hBuffer,
      &pPattern,
      &patternSize,
      &offset,
      &size,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_FILL_EXP,
      "urCommandBufferAppendMemBufferFillExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendMemBufferFillExp\n");

  ur_result_t result = pfnAppendMemBufferFillExp(
      hCommandBuffer, hBuffer, pPattern, patternSize, offset, size,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);

  getContext()->notify_end(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_FILL_EXP,
      "urCommandBufferAppendMemBufferFillExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_FILL_EXP,
        &params);
    logger.info("   <--- urCommandBufferAppendMemBufferFillExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMPrefetchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    /// [in] handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] pointer to USM allocated memory to prefetch.
    const void *pMemory,
    /// [in] size in bytes to be fetched.
    size_t size,
    /// [in] USM prefetch flags
    ur_usm_migration_flags_t flags,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendUSMPrefetchExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendUSMPrefetchExp;

  if (nullptr == pfnAppendUSMPrefetchExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_usm_prefetch_exp_params_t params = {
      &hCommandBuffer,
      &pMemory,
      &size,
      &flags,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_PREFETCH_EXP,
      "urCommandBufferAppendUSMPrefetchExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendUSMPrefetchExp\n");

  ur_result_t result = pfnAppendUSMPrefetchExp(
      hCommandBuffer, pMemory, size, flags, numSyncPointsInWaitList,
      pSyncPointWaitList, numEventsInWaitList, phEventWaitList, pSyncPoint,
      phEvent, phCommand);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_PREFETCH_EXP,
                           "urCommandBufferAppendUSMPrefetchExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_PREFETCH_EXP, &params);
    logger.info("   <--- urCommandBufferAppendUSMPrefetchExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendUSMAdviseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    /// [in] handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] pointer to the USM memory object.
    const void *pMemory,
    /// [in] size in bytes to be advised.
    size_t size,
    /// [in] USM memory advice
    ur_usm_advice_flags_t advice,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on. May
    /// be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,
    /// [out][optional][alloc] return an event object that will be signaled by
    /// the completion of this command in the next execution of the
    /// command-buffer.
    ur_event_handle_t *phEvent,
    /// [out][optional][alloc] Handle to this command.
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto pfnAppendUSMAdviseExp =
      getContext()->urDdiTable.CommandBufferExp.pfnAppendUSMAdviseExp;

  if (nullptr == pfnAppendUSMAdviseExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_append_usm_advise_exp_params_t params = {
      &hCommandBuffer,
      &pMemory,
      &size,
      &advice,
      &numSyncPointsInWaitList,
      &pSyncPointWaitList,
      &numEventsInWaitList,
      &phEventWaitList,
      &pSyncPoint,
      &phEvent,
      &phCommand};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_ADVISE_EXP,
      "urCommandBufferAppendUSMAdviseExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferAppendUSMAdviseExp\n");

  ur_result_t result = pfnAppendUSMAdviseExp(
      hCommandBuffer, pMemory, size, advice, numSyncPointsInWaitList,
      pSyncPointWaitList, numEventsInWaitList, phEventWaitList, pSyncPoint,
      phEvent, phCommand);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_ADVISE_EXP,
                           "urCommandBufferAppendUSMAdviseExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_ADVISE_EXP, &params);
    logger.info("   <--- urCommandBufferAppendUSMAdviseExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferEnqueueExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] The queue to submit this command-buffer for execution.
    ur_queue_handle_t hQueue,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command-buffer execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating no wait
    /// events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command-buffer execution instance. If phEventWaitList and
    /// phEvent are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnEnqueueExp = getContext()->urDdiTable.CommandBufferExp.pfnEnqueueExp;

  if (nullptr == pfnEnqueueExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_enqueue_exp_params_t params = {&hCommandBuffer, &hQueue,
                                                   &numEventsInWaitList,
                                                   &phEventWaitList, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_COMMAND_BUFFER_ENQUEUE_EXP,
                                 "urCommandBufferEnqueueExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferEnqueueExp\n");

  ur_result_t result = pfnEnqueueExp(
      hCommandBuffer, hQueue, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_ENQUEUE_EXP,
                           "urCommandBufferEnqueueExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_ENQUEUE_EXP, &params);
    logger.info("   <--- urCommandBufferEnqueueExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferUpdateKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Length of pUpdateKernelLaunch.
    uint32_t numKernelUpdates,
    /// [in][range(0, numKernelUpdates)]  List of structs defining how a
    /// kernel commands are to be updated.
    const ur_exp_command_buffer_update_kernel_launch_desc_t
        *pUpdateKernelLaunch) {
  auto pfnUpdateKernelLaunchExp =
      getContext()->urDdiTable.CommandBufferExp.pfnUpdateKernelLaunchExp;

  if (nullptr == pfnUpdateKernelLaunchExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_update_kernel_launch_exp_params_t params = {
      &hCommandBuffer, &numKernelUpdates, &pUpdateKernelLaunch};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_EXP,
      "urCommandBufferUpdateKernelLaunchExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferUpdateKernelLaunchExp\n");

  ur_result_t result = pfnUpdateKernelLaunchExp(
      hCommandBuffer, numKernelUpdates, pUpdateKernelLaunch);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_EXP,
                           "urCommandBufferUpdateKernelLaunchExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_EXP, &params);
    logger.info("   <--- urCommandBufferUpdateKernelLaunchExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferUpdateSignalEventExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferUpdateSignalEventExp(
    /// [in] Handle of the command-buffer command to update.
    ur_exp_command_buffer_command_handle_t hCommand,
    /// [out][alloc] Event to be signaled.
    ur_event_handle_t *phSignalEvent) {
  auto pfnUpdateSignalEventExp =
      getContext()->urDdiTable.CommandBufferExp.pfnUpdateSignalEventExp;

  if (nullptr == pfnUpdateSignalEventExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_update_signal_event_exp_params_t params = {&hCommand,
                                                               &phSignalEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_UPDATE_SIGNAL_EVENT_EXP,
      "urCommandBufferUpdateSignalEventExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferUpdateSignalEventExp\n");

  ur_result_t result = pfnUpdateSignalEventExp(hCommand, phSignalEvent);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_UPDATE_SIGNAL_EVENT_EXP,
                           "urCommandBufferUpdateSignalEventExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_UPDATE_SIGNAL_EVENT_EXP, &params);
    logger.info("   <--- urCommandBufferUpdateSignalEventExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferUpdateWaitEventsExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferUpdateWaitEventsExp(
    /// [in] Handle of the command-buffer command to update.
    ur_exp_command_buffer_command_handle_t hCommand,
    /// [in] Size of the event wait list.
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the command execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating no wait events.
    const ur_event_handle_t *phEventWaitList) {
  auto pfnUpdateWaitEventsExp =
      getContext()->urDdiTable.CommandBufferExp.pfnUpdateWaitEventsExp;

  if (nullptr == pfnUpdateWaitEventsExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_update_wait_events_exp_params_t params = {
      &hCommand, &numEventsInWaitList, &phEventWaitList};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_COMMAND_BUFFER_UPDATE_WAIT_EVENTS_EXP,
      "urCommandBufferUpdateWaitEventsExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferUpdateWaitEventsExp\n");

  ur_result_t result =
      pfnUpdateWaitEventsExp(hCommand, numEventsInWaitList, phEventWaitList);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_UPDATE_WAIT_EVENTS_EXP,
                           "urCommandBufferUpdateWaitEventsExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_UPDATE_WAIT_EVENTS_EXP, &params);
    logger.info("   <--- urCommandBufferUpdateWaitEventsExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferGetInfoExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferGetInfoExp(
    /// [in] handle of the command-buffer object
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] the name of the command-buffer property to query
    ur_exp_command_buffer_info_t propName,
    /// [in] size in bytes of the command-buffer property value
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] value of the
    /// command-buffer property
    void *pPropValue,
    /// [out][optional] bytes returned in command-buffer property
    size_t *pPropSizeRet) {
  auto pfnGetInfoExp = getContext()->urDdiTable.CommandBufferExp.pfnGetInfoExp;

  if (nullptr == pfnGetInfoExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_command_buffer_get_info_exp_params_t params = {
      &hCommandBuffer, &propName, &propSize, &pPropValue, &pPropSizeRet};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_COMMAND_BUFFER_GET_INFO_EXP,
                                 "urCommandBufferGetInfoExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urCommandBufferGetInfoExp\n");

  ur_result_t result = pfnGetInfoExp(hCommandBuffer, propName, propSize,
                                     pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_COMMAND_BUFFER_GET_INFO_EXP,
                           "urCommandBufferGetInfoExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_COMMAND_BUFFER_GET_INFO_EXP, &params);
    logger.info("   <--- urCommandBufferGetInfoExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueCooperativeKernelLaunchExp
__urdlllocal ur_result_t UR_APICALL urEnqueueCooperativeKernelLaunchExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] number of dimensions, from 1 to 3, to specify the global and
    /// work-group work-items
    uint32_t workDim,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkOffset,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// number of global work-items in workDim that will execute the kernel
    /// function
    const size_t *pGlobalWorkSize,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the number of local work-items forming a work-group that will
    /// execute the kernel function.
    /// If nullptr, the runtime implementation will choose the work-group size.
    const size_t *pLocalWorkSize,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular kernel execution instance. If phEventWaitList and phEvent
    /// are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnCooperativeKernelLaunchExp =
      getContext()->urDdiTable.EnqueueExp.pfnCooperativeKernelLaunchExp;

  if (nullptr == pfnCooperativeKernelLaunchExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_cooperative_kernel_launch_exp_params_t params = {
      &hQueue,
      &hKernel,
      &workDim,
      &pGlobalWorkOffset,
      &pGlobalWorkSize,
      &pLocalWorkSize,
      &numEventsInWaitList,
      &phEventWaitList,
      &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_COOPERATIVE_KERNEL_LAUNCH_EXP,
      "urEnqueueCooperativeKernelLaunchExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueCooperativeKernelLaunchExp\n");

  ur_result_t result = pfnCooperativeKernelLaunchExp(
      hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_COOPERATIVE_KERNEL_LAUNCH_EXP,
                           "urEnqueueCooperativeKernelLaunchExp", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_COOPERATIVE_KERNEL_LAUNCH_EXP, &params);
    logger.info("   <--- urEnqueueCooperativeKernelLaunchExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSuggestMaxCooperativeGroupCountExp
__urdlllocal ur_result_t UR_APICALL urKernelSuggestMaxCooperativeGroupCountExp(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] number of dimensions, from 1 to 3, to specify the work-group
    /// work-items
    uint32_t workDim,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// number of local work-items forming a work-group that will execute the
    /// kernel function.
    const size_t *pLocalWorkSize,
    /// [in] size of dynamic shared memory, for each work-group, in bytes,
    /// that will be used when the kernel is launched
    size_t dynamicSharedMemorySize,
    /// [out] pointer to maximum number of groups
    uint32_t *pGroupCountRet) {
  auto pfnSuggestMaxCooperativeGroupCountExp =
      getContext()->urDdiTable.KernelExp.pfnSuggestMaxCooperativeGroupCountExp;

  if (nullptr == pfnSuggestMaxCooperativeGroupCountExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_kernel_suggest_max_cooperative_group_count_exp_params_t params = {
      &hKernel,
      &hDevice,
      &workDim,
      &pLocalWorkSize,
      &dynamicSharedMemorySize,
      &pGroupCountRet};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_KERNEL_SUGGEST_MAX_COOPERATIVE_GROUP_COUNT_EXP,
      "urKernelSuggestMaxCooperativeGroupCountExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urKernelSuggestMaxCooperativeGroupCountExp\n");

  ur_result_t result = pfnSuggestMaxCooperativeGroupCountExp(
      hKernel, hDevice, workDim, pLocalWorkSize, dynamicSharedMemorySize,
      pGroupCountRet);

  getContext()->notify_end(
      UR_FUNCTION_KERNEL_SUGGEST_MAX_COOPERATIVE_GROUP_COUNT_EXP,
      "urKernelSuggestMaxCooperativeGroupCountExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_KERNEL_SUGGEST_MAX_COOPERATIVE_GROUP_COUNT_EXP,
        &params);
    logger.info(
        "   <--- urKernelSuggestMaxCooperativeGroupCountExp({}) -> {};\n",
        args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueTimestampRecordingExp
__urdlllocal ur_result_t UR_APICALL urEnqueueTimestampRecordingExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] indicates whether the call to this function should block until
    /// until the device timestamp recording command has executed on the
    /// device.
    bool blocking,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating no wait
    /// events.
    const ur_event_handle_t *phEventWaitList,
    /// [in,out] return an event object that identifies this particular kernel
    /// execution instance. Profiling information can be queried
    /// from this event as if `hQueue` had profiling enabled. Querying
    /// `UR_PROFILING_INFO_COMMAND_QUEUED` or `UR_PROFILING_INFO_COMMAND_SUBMIT`
    /// reports the timestamp at the time of the call to this function.
    /// Querying `UR_PROFILING_INFO_COMMAND_START` or
    /// `UR_PROFILING_INFO_COMMAND_END` reports the timestamp recorded when the
    /// command is executed on the device. If phEventWaitList and phEvent are
    /// not NULL, phEvent must not refer to an element of the phEventWaitList
    /// array.
    ur_event_handle_t *phEvent) {
  auto pfnTimestampRecordingExp =
      getContext()->urDdiTable.EnqueueExp.pfnTimestampRecordingExp;

  if (nullptr == pfnTimestampRecordingExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_timestamp_recording_exp_params_t params = {
      &hQueue, &blocking, &numEventsInWaitList, &phEventWaitList, &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_TIMESTAMP_RECORDING_EXP,
                                 "urEnqueueTimestampRecordingExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueTimestampRecordingExp\n");

  ur_result_t result = pfnTimestampRecordingExp(
      hQueue, blocking, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_TIMESTAMP_RECORDING_EXP,
                           "urEnqueueTimestampRecordingExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_TIMESTAMP_RECORDING_EXP, &params);
    logger.info("   <--- urEnqueueTimestampRecordingExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunchCustomExp
__urdlllocal ur_result_t UR_APICALL urEnqueueKernelLaunchCustomExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] number of dimensions, from 1 to 3, to specify the global and
    /// work-group work-items
    uint32_t workDim,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkOffset,
    /// [in] pointer to an array of workDim unsigned values that specify the
    /// number of global work-items in workDim that will execute the kernel
    /// function
    const size_t *pGlobalWorkSize,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the number of local work-items forming a work-group that will
    /// execute the kernel function. If nullptr, the runtime implementation
    /// will choose the work-group size.
    const size_t *pLocalWorkSize,
    /// [in] size of the launch prop list
    uint32_t numPropsInLaunchPropList,
    /// [in][range(0, numPropsInLaunchPropList)] pointer to a list of launch
    /// properties
    const ur_exp_launch_property_t *launchPropList,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If nullptr,
    /// the numEventsInWaitList must be 0, indicating that no wait event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular kernel execution instance. If phEventWaitList and phEvent
    /// are not NULL, phEvent must not refer to an element of the
    /// phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnKernelLaunchCustomExp =
      getContext()->urDdiTable.EnqueueExp.pfnKernelLaunchCustomExp;

  if (nullptr == pfnKernelLaunchCustomExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_kernel_launch_custom_exp_params_t params = {
      &hQueue,
      &hKernel,
      &workDim,
      &pGlobalWorkOffset,
      &pGlobalWorkSize,
      &pLocalWorkSize,
      &numPropsInLaunchPropList,
      &launchPropList,
      &numEventsInWaitList,
      &phEventWaitList,
      &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH_CUSTOM_EXP,
                                 "urEnqueueKernelLaunchCustomExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueKernelLaunchCustomExp\n");

  ur_result_t result = pfnKernelLaunchCustomExp(
      hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numPropsInLaunchPropList, launchPropList,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH_CUSTOM_EXP,
                           "urEnqueueKernelLaunchCustomExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH_CUSTOM_EXP, &params);
    logger.info("   <--- urEnqueueKernelLaunchCustomExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuildExp
__urdlllocal ur_result_t UR_APICALL urProgramBuildExp(
    /// [in] Handle of the program to build.
    ur_program_handle_t hProgram,
    /// [in] number of devices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] pointer to array of device handles
    ur_device_handle_t *phDevices,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) {
  auto pfnBuildExp = getContext()->urDdiTable.ProgramExp.pfnBuildExp;

  if (nullptr == pfnBuildExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_build_exp_params_t params = {&hProgram, &numDevices, &phDevices,
                                          &pOptions};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_BUILD_EXP,
                                                 "urProgramBuildExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramBuildExp\n");

  ur_result_t result = pfnBuildExp(hProgram, numDevices, phDevices, pOptions);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_BUILD_EXP, "urProgramBuildExp",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_BUILD_EXP,
                                    &params);
    logger.info("   <--- urProgramBuildExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCompileExp
__urdlllocal ur_result_t UR_APICALL urProgramCompileExp(
    /// [in][out] handle of the program to compile.
    ur_program_handle_t hProgram,
    /// [in] number of devices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] pointer to array of device handles
    ur_device_handle_t *phDevices,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) {
  auto pfnCompileExp = getContext()->urDdiTable.ProgramExp.pfnCompileExp;

  if (nullptr == pfnCompileExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_compile_exp_params_t params = {&hProgram, &numDevices, &phDevices,
                                            &pOptions};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_PROGRAM_COMPILE_EXP, "urProgramCompileExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramCompileExp\n");

  ur_result_t result = pfnCompileExp(hProgram, numDevices, phDevices, pOptions);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_COMPILE_EXP,
                           "urProgramCompileExp", &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_COMPILE_EXP,
                                    &params);
    logger.info("   <--- urProgramCompileExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLinkExp
__urdlllocal ur_result_t UR_APICALL urProgramLinkExp(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in] number of devices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] pointer to array of device handles
    ur_device_handle_t *phDevices,
    /// [in] number of program handles in `phPrograms`.
    uint32_t count,
    /// [in][range(0, count)] pointer to array of program handles.
    const ur_program_handle_t *phPrograms,
    /// [in][optional] pointer to linker options null-terminated string.
    const char *pOptions,
    /// [out][alloc] pointer to handle of program object created.
    ur_program_handle_t *phProgram) {
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }
  auto pfnLinkExp = getContext()->urDdiTable.ProgramExp.pfnLinkExp;

  if (nullptr == pfnLinkExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_program_link_exp_params_t params = {&hContext, &numDevices, &phDevices,
                                         &count,    &phPrograms, &pOptions,
                                         &phProgram};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_PROGRAM_LINK_EXP,
                                                 "urProgramLinkExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urProgramLinkExp\n");

  ur_result_t result = pfnLinkExp(hContext, numDevices, phDevices, count,
                                  phPrograms, pOptions, phProgram);

  getContext()->notify_end(UR_FUNCTION_PROGRAM_LINK_EXP, "urProgramLinkExp",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_PROGRAM_LINK_EXP,
                                    &params);
    logger.info("   <--- urProgramLinkExp({}) -> {};\n", args_str.str(),
                result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMImportExp
__urdlllocal ur_result_t UR_APICALL urUSMImportExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to host memory object
    void *pMem,
    /// [in] size in bytes of the host memory object to be imported
    size_t size) {
  auto pfnImportExp = getContext()->urDdiTable.USMExp.pfnImportExp;

  if (nullptr == pfnImportExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_import_exp_params_t params = {&hContext, &pMem, &size};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_IMPORT_EXP,
                                                 "urUSMImportExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMImportExp\n");

  ur_result_t result = pfnImportExp(hContext, pMem, size);

  getContext()->notify_end(UR_FUNCTION_USM_IMPORT_EXP, "urUSMImportExp",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_IMPORT_EXP,
                                    &params);
    logger.info("   <--- urUSMImportExp({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMReleaseExp
__urdlllocal ur_result_t UR_APICALL urUSMReleaseExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to host memory object
    void *pMem) {
  auto pfnReleaseExp = getContext()->urDdiTable.USMExp.pfnReleaseExp;

  if (nullptr == pfnReleaseExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_release_exp_params_t params = {&hContext, &pMem};
  uint64_t instance = getContext()->notify_begin(UR_FUNCTION_USM_RELEASE_EXP,
                                                 "urUSMReleaseExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUSMReleaseExp\n");

  ur_result_t result = pfnReleaseExp(hContext, pMem);

  getContext()->notify_end(UR_FUNCTION_USM_RELEASE_EXP, "urUSMReleaseExp",
                           &params, &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(args_str, UR_FUNCTION_USM_RELEASE_EXP,
                                    &params);
    logger.info("   <--- urUSMReleaseExp({}) -> {};\n", args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PEnablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    /// [in] handle of the command device object
    ur_device_handle_t commandDevice,
    /// [in] handle of the peer device object
    ur_device_handle_t peerDevice) {
  auto pfnEnablePeerAccessExp =
      getContext()->urDdiTable.UsmP2PExp.pfnEnablePeerAccessExp;

  if (nullptr == pfnEnablePeerAccessExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_p2p_enable_peer_access_exp_params_t params = {&commandDevice,
                                                       &peerDevice};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_USM_P2P_ENABLE_PEER_ACCESS_EXP,
                                 "urUsmP2PEnablePeerAccessExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUsmP2PEnablePeerAccessExp\n");

  ur_result_t result = pfnEnablePeerAccessExp(commandDevice, peerDevice);

  getContext()->notify_end(UR_FUNCTION_USM_P2P_ENABLE_PEER_ACCESS_EXP,
                           "urUsmP2PEnablePeerAccessExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_P2P_ENABLE_PEER_ACCESS_EXP, &params);
    logger.info("   <--- urUsmP2PEnablePeerAccessExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PDisablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    /// [in] handle of the command device object
    ur_device_handle_t commandDevice,
    /// [in] handle of the peer device object
    ur_device_handle_t peerDevice) {
  auto pfnDisablePeerAccessExp =
      getContext()->urDdiTable.UsmP2PExp.pfnDisablePeerAccessExp;

  if (nullptr == pfnDisablePeerAccessExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_p2p_disable_peer_access_exp_params_t params = {&commandDevice,
                                                        &peerDevice};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_USM_P2P_DISABLE_PEER_ACCESS_EXP,
                                 "urUsmP2PDisablePeerAccessExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUsmP2PDisablePeerAccessExp\n");

  ur_result_t result = pfnDisablePeerAccessExp(commandDevice, peerDevice);

  getContext()->notify_end(UR_FUNCTION_USM_P2P_DISABLE_PEER_ACCESS_EXP,
                           "urUsmP2PDisablePeerAccessExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_P2P_DISABLE_PEER_ACCESS_EXP, &params);
    logger.info("   <--- urUsmP2PDisablePeerAccessExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PPeerAccessGetInfoExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    /// [in] handle of the command device object
    ur_device_handle_t commandDevice,
    /// [in] handle of the peer device object
    ur_device_handle_t peerDevice,
    /// [in] type of the info to retrieve
    ur_exp_peer_info_t propName,
    /// [in] the number of bytes pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding
    /// the info.
    /// If propSize is not equal to or greater than the real number of bytes
    /// needed to return the info
    /// then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName.
    size_t *pPropSizeRet) {
  auto pfnPeerAccessGetInfoExp =
      getContext()->urDdiTable.UsmP2PExp.pfnPeerAccessGetInfoExp;

  if (nullptr == pfnPeerAccessGetInfoExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_usm_p2p_peer_access_get_info_exp_params_t params = {
      &commandDevice, &peerDevice, &propName,
      &propSize,      &pPropValue, &pPropSizeRet};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_USM_P2P_PEER_ACCESS_GET_INFO_EXP,
                                 "urUsmP2PPeerAccessGetInfoExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urUsmP2PPeerAccessGetInfoExp\n");

  ur_result_t result = pfnPeerAccessGetInfoExp(
      commandDevice, peerDevice, propName, propSize, pPropValue, pPropSizeRet);

  getContext()->notify_end(UR_FUNCTION_USM_P2P_PEER_ACCESS_GET_INFO_EXP,
                           "urUsmP2PPeerAccessGetInfoExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_USM_P2P_PEER_ACCESS_GET_INFO_EXP, &params);
    logger.info("   <--- urUsmP2PPeerAccessGetInfoExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueEventsWaitWithBarrierExt
__urdlllocal ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrierExt(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in][optional] pointer to the extended enqueue properties
    const ur_exp_enqueue_ext_properties_t *pProperties,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed.
    /// If nullptr, the numEventsInWaitList must be 0, indicating that all
    /// previously enqueued commands
    /// must be complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies this
    /// particular command instance. If phEventWaitList and phEvent are not
    /// NULL, phEvent must not refer to an element of the phEventWaitList array.
    ur_event_handle_t *phEvent) {
  auto pfnEventsWaitWithBarrierExt =
      getContext()->urDdiTable.Enqueue.pfnEventsWaitWithBarrierExt;

  if (nullptr == pfnEventsWaitWithBarrierExt)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_events_wait_with_barrier_ext_params_t params = {
      &hQueue, &pProperties, &numEventsInWaitList, &phEventWaitList, &phEvent};
  uint64_t instance = getContext()->notify_begin(
      UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER_EXT,
      "urEnqueueEventsWaitWithBarrierExt", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueEventsWaitWithBarrierExt\n");

  ur_result_t result = pfnEventsWaitWithBarrierExt(
      hQueue, pProperties, numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER_EXT,
                           "urEnqueueEventsWaitWithBarrierExt", &params,
                           &result, instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER_EXT, &params);
    logger.info("   <--- urEnqueueEventsWaitWithBarrierExt({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueNativeCommandExp
__urdlllocal ur_result_t UR_APICALL urEnqueueNativeCommandExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] function calling the native underlying API, to be executed
    /// immediately.
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue,
    /// [in][optional] data used by pfnNativeEnqueue
    void *data,
    /// [in] size of the mem list
    uint32_t numMemsInMemList,
    /// [in][optional][range(0, numMemsInMemList)] mems that are used within
    /// pfnNativeEnqueue using ::urMemGetNativeHandle.
    /// If nullptr, the numMemsInMemList must be 0, indicating that no mems
    /// are accessed with ::urMemGetNativeHandle within pfnNativeEnqueue.
    const ur_mem_handle_t *phMemList,
    /// [in][optional] pointer to the native enqueue properties
    const ur_exp_enqueue_native_command_properties_t *pProperties,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution.
    /// If nullptr, the numEventsInWaitList must be 0, indicating no wait
    /// events.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional][alloc] return an event object that identifies the work
    /// that has
    /// been enqueued in nativeEnqueueFunc. If phEventWaitList and phEvent are
    /// not NULL, phEvent must not refer to an element of the phEventWaitList
    /// array.
    ur_event_handle_t *phEvent) {
  auto pfnNativeCommandExp =
      getContext()->urDdiTable.EnqueueExp.pfnNativeCommandExp;

  if (nullptr == pfnNativeCommandExp)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  ur_enqueue_native_command_exp_params_t params = {&hQueue,
                                                   &pfnNativeEnqueue,
                                                   &data,
                                                   &numMemsInMemList,
                                                   &phMemList,
                                                   &pProperties,
                                                   &numEventsInWaitList,
                                                   &phEventWaitList,
                                                   &phEvent};
  uint64_t instance =
      getContext()->notify_begin(UR_FUNCTION_ENQUEUE_NATIVE_COMMAND_EXP,
                                 "urEnqueueNativeCommandExp", &params);

  auto &logger = getContext()->logger;
  logger.info("   ---> urEnqueueNativeCommandExp\n");

  ur_result_t result = pfnNativeCommandExp(
      hQueue, pfnNativeEnqueue, data, numMemsInMemList, phMemList, pProperties,
      numEventsInWaitList, phEventWaitList, phEvent);

  getContext()->notify_end(UR_FUNCTION_ENQUEUE_NATIVE_COMMAND_EXP,
                           "urEnqueueNativeCommandExp", &params, &result,
                           instance);

  if (logger.getLevel() <= logger::Level::INFO) {
    std::ostringstream args_str;
    ur::extras::printFunctionParams(
        args_str, UR_FUNCTION_ENQUEUE_NATIVE_COMMAND_EXP, &params);
    logger.info("   <--- urEnqueueNativeCommandExp({}) -> {};\n",
                args_str.str(), result);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_global_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Global;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnAdapterGet = pDdiTable->pfnAdapterGet;
  pDdiTable->pfnAdapterGet = ur_tracing_layer::urAdapterGet;

  dditable.pfnAdapterRelease = pDdiTable->pfnAdapterRelease;
  pDdiTable->pfnAdapterRelease = ur_tracing_layer::urAdapterRelease;

  dditable.pfnAdapterRetain = pDdiTable->pfnAdapterRetain;
  pDdiTable->pfnAdapterRetain = ur_tracing_layer::urAdapterRetain;

  dditable.pfnAdapterGetLastError = pDdiTable->pfnAdapterGetLastError;
  pDdiTable->pfnAdapterGetLastError = ur_tracing_layer::urAdapterGetLastError;

  dditable.pfnAdapterGetInfo = pDdiTable->pfnAdapterGetInfo;
  pDdiTable->pfnAdapterGetInfo = ur_tracing_layer::urAdapterGetInfo;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's BindlessImagesExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_bindless_images_exp_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.BindlessImagesExp;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnUnsampledImageHandleDestroyExp =
      pDdiTable->pfnUnsampledImageHandleDestroyExp;
  pDdiTable->pfnUnsampledImageHandleDestroyExp =
      ur_tracing_layer::urBindlessImagesUnsampledImageHandleDestroyExp;

  dditable.pfnSampledImageHandleDestroyExp =
      pDdiTable->pfnSampledImageHandleDestroyExp;
  pDdiTable->pfnSampledImageHandleDestroyExp =
      ur_tracing_layer::urBindlessImagesSampledImageHandleDestroyExp;

  dditable.pfnImageAllocateExp = pDdiTable->pfnImageAllocateExp;
  pDdiTable->pfnImageAllocateExp =
      ur_tracing_layer::urBindlessImagesImageAllocateExp;

  dditable.pfnImageFreeExp = pDdiTable->pfnImageFreeExp;
  pDdiTable->pfnImageFreeExp = ur_tracing_layer::urBindlessImagesImageFreeExp;

  dditable.pfnUnsampledImageCreateExp = pDdiTable->pfnUnsampledImageCreateExp;
  pDdiTable->pfnUnsampledImageCreateExp =
      ur_tracing_layer::urBindlessImagesUnsampledImageCreateExp;

  dditable.pfnSampledImageCreateExp = pDdiTable->pfnSampledImageCreateExp;
  pDdiTable->pfnSampledImageCreateExp =
      ur_tracing_layer::urBindlessImagesSampledImageCreateExp;

  dditable.pfnImageCopyExp = pDdiTable->pfnImageCopyExp;
  pDdiTable->pfnImageCopyExp = ur_tracing_layer::urBindlessImagesImageCopyExp;

  dditable.pfnImageGetInfoExp = pDdiTable->pfnImageGetInfoExp;
  pDdiTable->pfnImageGetInfoExp =
      ur_tracing_layer::urBindlessImagesImageGetInfoExp;

  dditable.pfnMipmapGetLevelExp = pDdiTable->pfnMipmapGetLevelExp;
  pDdiTable->pfnMipmapGetLevelExp =
      ur_tracing_layer::urBindlessImagesMipmapGetLevelExp;

  dditable.pfnMipmapFreeExp = pDdiTable->pfnMipmapFreeExp;
  pDdiTable->pfnMipmapFreeExp = ur_tracing_layer::urBindlessImagesMipmapFreeExp;

  dditable.pfnImportExternalMemoryExp = pDdiTable->pfnImportExternalMemoryExp;
  pDdiTable->pfnImportExternalMemoryExp =
      ur_tracing_layer::urBindlessImagesImportExternalMemoryExp;

  dditable.pfnMapExternalArrayExp = pDdiTable->pfnMapExternalArrayExp;
  pDdiTable->pfnMapExternalArrayExp =
      ur_tracing_layer::urBindlessImagesMapExternalArrayExp;

  dditable.pfnMapExternalLinearMemoryExp =
      pDdiTable->pfnMapExternalLinearMemoryExp;
  pDdiTable->pfnMapExternalLinearMemoryExp =
      ur_tracing_layer::urBindlessImagesMapExternalLinearMemoryExp;

  dditable.pfnReleaseExternalMemoryExp = pDdiTable->pfnReleaseExternalMemoryExp;
  pDdiTable->pfnReleaseExternalMemoryExp =
      ur_tracing_layer::urBindlessImagesReleaseExternalMemoryExp;

  dditable.pfnImportExternalSemaphoreExp =
      pDdiTable->pfnImportExternalSemaphoreExp;
  pDdiTable->pfnImportExternalSemaphoreExp =
      ur_tracing_layer::urBindlessImagesImportExternalSemaphoreExp;

  dditable.pfnReleaseExternalSemaphoreExp =
      pDdiTable->pfnReleaseExternalSemaphoreExp;
  pDdiTable->pfnReleaseExternalSemaphoreExp =
      ur_tracing_layer::urBindlessImagesReleaseExternalSemaphoreExp;

  dditable.pfnWaitExternalSemaphoreExp = pDdiTable->pfnWaitExternalSemaphoreExp;
  pDdiTable->pfnWaitExternalSemaphoreExp =
      ur_tracing_layer::urBindlessImagesWaitExternalSemaphoreExp;

  dditable.pfnSignalExternalSemaphoreExp =
      pDdiTable->pfnSignalExternalSemaphoreExp;
  pDdiTable->pfnSignalExternalSemaphoreExp =
      ur_tracing_layer::urBindlessImagesSignalExternalSemaphoreExp;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's CommandBufferExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_command_buffer_exp_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.CommandBufferExp;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnCreateExp = pDdiTable->pfnCreateExp;
  pDdiTable->pfnCreateExp = ur_tracing_layer::urCommandBufferCreateExp;

  dditable.pfnRetainExp = pDdiTable->pfnRetainExp;
  pDdiTable->pfnRetainExp = ur_tracing_layer::urCommandBufferRetainExp;

  dditable.pfnReleaseExp = pDdiTable->pfnReleaseExp;
  pDdiTable->pfnReleaseExp = ur_tracing_layer::urCommandBufferReleaseExp;

  dditable.pfnFinalizeExp = pDdiTable->pfnFinalizeExp;
  pDdiTable->pfnFinalizeExp = ur_tracing_layer::urCommandBufferFinalizeExp;

  dditable.pfnAppendKernelLaunchExp = pDdiTable->pfnAppendKernelLaunchExp;
  pDdiTable->pfnAppendKernelLaunchExp =
      ur_tracing_layer::urCommandBufferAppendKernelLaunchExp;

  dditable.pfnAppendUSMMemcpyExp = pDdiTable->pfnAppendUSMMemcpyExp;
  pDdiTable->pfnAppendUSMMemcpyExp =
      ur_tracing_layer::urCommandBufferAppendUSMMemcpyExp;

  dditable.pfnAppendUSMFillExp = pDdiTable->pfnAppendUSMFillExp;
  pDdiTable->pfnAppendUSMFillExp =
      ur_tracing_layer::urCommandBufferAppendUSMFillExp;

  dditable.pfnAppendMemBufferCopyExp = pDdiTable->pfnAppendMemBufferCopyExp;
  pDdiTable->pfnAppendMemBufferCopyExp =
      ur_tracing_layer::urCommandBufferAppendMemBufferCopyExp;

  dditable.pfnAppendMemBufferWriteExp = pDdiTable->pfnAppendMemBufferWriteExp;
  pDdiTable->pfnAppendMemBufferWriteExp =
      ur_tracing_layer::urCommandBufferAppendMemBufferWriteExp;

  dditable.pfnAppendMemBufferReadExp = pDdiTable->pfnAppendMemBufferReadExp;
  pDdiTable->pfnAppendMemBufferReadExp =
      ur_tracing_layer::urCommandBufferAppendMemBufferReadExp;

  dditable.pfnAppendMemBufferCopyRectExp =
      pDdiTable->pfnAppendMemBufferCopyRectExp;
  pDdiTable->pfnAppendMemBufferCopyRectExp =
      ur_tracing_layer::urCommandBufferAppendMemBufferCopyRectExp;

  dditable.pfnAppendMemBufferWriteRectExp =
      pDdiTable->pfnAppendMemBufferWriteRectExp;
  pDdiTable->pfnAppendMemBufferWriteRectExp =
      ur_tracing_layer::urCommandBufferAppendMemBufferWriteRectExp;

  dditable.pfnAppendMemBufferReadRectExp =
      pDdiTable->pfnAppendMemBufferReadRectExp;
  pDdiTable->pfnAppendMemBufferReadRectExp =
      ur_tracing_layer::urCommandBufferAppendMemBufferReadRectExp;

  dditable.pfnAppendMemBufferFillExp = pDdiTable->pfnAppendMemBufferFillExp;
  pDdiTable->pfnAppendMemBufferFillExp =
      ur_tracing_layer::urCommandBufferAppendMemBufferFillExp;

  dditable.pfnAppendUSMPrefetchExp = pDdiTable->pfnAppendUSMPrefetchExp;
  pDdiTable->pfnAppendUSMPrefetchExp =
      ur_tracing_layer::urCommandBufferAppendUSMPrefetchExp;

  dditable.pfnAppendUSMAdviseExp = pDdiTable->pfnAppendUSMAdviseExp;
  pDdiTable->pfnAppendUSMAdviseExp =
      ur_tracing_layer::urCommandBufferAppendUSMAdviseExp;

  dditable.pfnEnqueueExp = pDdiTable->pfnEnqueueExp;
  pDdiTable->pfnEnqueueExp = ur_tracing_layer::urCommandBufferEnqueueExp;

  dditable.pfnUpdateKernelLaunchExp = pDdiTable->pfnUpdateKernelLaunchExp;
  pDdiTable->pfnUpdateKernelLaunchExp =
      ur_tracing_layer::urCommandBufferUpdateKernelLaunchExp;

  dditable.pfnUpdateSignalEventExp = pDdiTable->pfnUpdateSignalEventExp;
  pDdiTable->pfnUpdateSignalEventExp =
      ur_tracing_layer::urCommandBufferUpdateSignalEventExp;

  dditable.pfnUpdateWaitEventsExp = pDdiTable->pfnUpdateWaitEventsExp;
  pDdiTable->pfnUpdateWaitEventsExp =
      ur_tracing_layer::urCommandBufferUpdateWaitEventsExp;

  dditable.pfnGetInfoExp = pDdiTable->pfnGetInfoExp;
  pDdiTable->pfnGetInfoExp = ur_tracing_layer::urCommandBufferGetInfoExp;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetContextProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_context_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Context;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnCreate = pDdiTable->pfnCreate;
  pDdiTable->pfnCreate = ur_tracing_layer::urContextCreate;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urContextRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urContextRelease;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urContextGetInfo;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urContextGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urContextCreateWithNativeHandle;

  dditable.pfnSetExtendedDeleter = pDdiTable->pfnSetExtendedDeleter;
  pDdiTable->pfnSetExtendedDeleter =
      ur_tracing_layer::urContextSetExtendedDeleter;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Enqueue;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnKernelLaunch = pDdiTable->pfnKernelLaunch;
  pDdiTable->pfnKernelLaunch = ur_tracing_layer::urEnqueueKernelLaunch;

  dditable.pfnEventsWait = pDdiTable->pfnEventsWait;
  pDdiTable->pfnEventsWait = ur_tracing_layer::urEnqueueEventsWait;

  dditable.pfnEventsWaitWithBarrier = pDdiTable->pfnEventsWaitWithBarrier;
  pDdiTable->pfnEventsWaitWithBarrier =
      ur_tracing_layer::urEnqueueEventsWaitWithBarrier;

  dditable.pfnMemBufferRead = pDdiTable->pfnMemBufferRead;
  pDdiTable->pfnMemBufferRead = ur_tracing_layer::urEnqueueMemBufferRead;

  dditable.pfnMemBufferWrite = pDdiTable->pfnMemBufferWrite;
  pDdiTable->pfnMemBufferWrite = ur_tracing_layer::urEnqueueMemBufferWrite;

  dditable.pfnMemBufferReadRect = pDdiTable->pfnMemBufferReadRect;
  pDdiTable->pfnMemBufferReadRect =
      ur_tracing_layer::urEnqueueMemBufferReadRect;

  dditable.pfnMemBufferWriteRect = pDdiTable->pfnMemBufferWriteRect;
  pDdiTable->pfnMemBufferWriteRect =
      ur_tracing_layer::urEnqueueMemBufferWriteRect;

  dditable.pfnMemBufferCopy = pDdiTable->pfnMemBufferCopy;
  pDdiTable->pfnMemBufferCopy = ur_tracing_layer::urEnqueueMemBufferCopy;

  dditable.pfnMemBufferCopyRect = pDdiTable->pfnMemBufferCopyRect;
  pDdiTable->pfnMemBufferCopyRect =
      ur_tracing_layer::urEnqueueMemBufferCopyRect;

  dditable.pfnMemBufferFill = pDdiTable->pfnMemBufferFill;
  pDdiTable->pfnMemBufferFill = ur_tracing_layer::urEnqueueMemBufferFill;

  dditable.pfnMemImageRead = pDdiTable->pfnMemImageRead;
  pDdiTable->pfnMemImageRead = ur_tracing_layer::urEnqueueMemImageRead;

  dditable.pfnMemImageWrite = pDdiTable->pfnMemImageWrite;
  pDdiTable->pfnMemImageWrite = ur_tracing_layer::urEnqueueMemImageWrite;

  dditable.pfnMemImageCopy = pDdiTable->pfnMemImageCopy;
  pDdiTable->pfnMemImageCopy = ur_tracing_layer::urEnqueueMemImageCopy;

  dditable.pfnMemBufferMap = pDdiTable->pfnMemBufferMap;
  pDdiTable->pfnMemBufferMap = ur_tracing_layer::urEnqueueMemBufferMap;

  dditable.pfnMemUnmap = pDdiTable->pfnMemUnmap;
  pDdiTable->pfnMemUnmap = ur_tracing_layer::urEnqueueMemUnmap;

  dditable.pfnUSMFill = pDdiTable->pfnUSMFill;
  pDdiTable->pfnUSMFill = ur_tracing_layer::urEnqueueUSMFill;

  dditable.pfnUSMMemcpy = pDdiTable->pfnUSMMemcpy;
  pDdiTable->pfnUSMMemcpy = ur_tracing_layer::urEnqueueUSMMemcpy;

  dditable.pfnUSMPrefetch = pDdiTable->pfnUSMPrefetch;
  pDdiTable->pfnUSMPrefetch = ur_tracing_layer::urEnqueueUSMPrefetch;

  dditable.pfnUSMAdvise = pDdiTable->pfnUSMAdvise;
  pDdiTable->pfnUSMAdvise = ur_tracing_layer::urEnqueueUSMAdvise;

  dditable.pfnUSMFill2D = pDdiTable->pfnUSMFill2D;
  pDdiTable->pfnUSMFill2D = ur_tracing_layer::urEnqueueUSMFill2D;

  dditable.pfnUSMMemcpy2D = pDdiTable->pfnUSMMemcpy2D;
  pDdiTable->pfnUSMMemcpy2D = ur_tracing_layer::urEnqueueUSMMemcpy2D;

  dditable.pfnDeviceGlobalVariableWrite =
      pDdiTable->pfnDeviceGlobalVariableWrite;
  pDdiTable->pfnDeviceGlobalVariableWrite =
      ur_tracing_layer::urEnqueueDeviceGlobalVariableWrite;

  dditable.pfnDeviceGlobalVariableRead = pDdiTable->pfnDeviceGlobalVariableRead;
  pDdiTable->pfnDeviceGlobalVariableRead =
      ur_tracing_layer::urEnqueueDeviceGlobalVariableRead;

  dditable.pfnReadHostPipe = pDdiTable->pfnReadHostPipe;
  pDdiTable->pfnReadHostPipe = ur_tracing_layer::urEnqueueReadHostPipe;

  dditable.pfnWriteHostPipe = pDdiTable->pfnWriteHostPipe;
  pDdiTable->pfnWriteHostPipe = ur_tracing_layer::urEnqueueWriteHostPipe;

  dditable.pfnEventsWaitWithBarrierExt = pDdiTable->pfnEventsWaitWithBarrierExt;
  pDdiTable->pfnEventsWaitWithBarrierExt =
      ur_tracing_layer::urEnqueueEventsWaitWithBarrierExt;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's EnqueueExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_exp_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.EnqueueExp;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnKernelLaunchCustomExp = pDdiTable->pfnKernelLaunchCustomExp;
  pDdiTable->pfnKernelLaunchCustomExp =
      ur_tracing_layer::urEnqueueKernelLaunchCustomExp;

  dditable.pfnUSMDeviceAllocExp = pDdiTable->pfnUSMDeviceAllocExp;
  pDdiTable->pfnUSMDeviceAllocExp =
      ur_tracing_layer::urEnqueueUSMDeviceAllocExp;

  dditable.pfnUSMSharedAllocExp = pDdiTable->pfnUSMSharedAllocExp;
  pDdiTable->pfnUSMSharedAllocExp =
      ur_tracing_layer::urEnqueueUSMSharedAllocExp;

  dditable.pfnUSMHostAllocExp = pDdiTable->pfnUSMHostAllocExp;
  pDdiTable->pfnUSMHostAllocExp = ur_tracing_layer::urEnqueueUSMHostAllocExp;

  dditable.pfnUSMFreeExp = pDdiTable->pfnUSMFreeExp;
  pDdiTable->pfnUSMFreeExp = ur_tracing_layer::urEnqueueUSMFreeExp;

  dditable.pfnCooperativeKernelLaunchExp =
      pDdiTable->pfnCooperativeKernelLaunchExp;
  pDdiTable->pfnCooperativeKernelLaunchExp =
      ur_tracing_layer::urEnqueueCooperativeKernelLaunchExp;

  dditable.pfnTimestampRecordingExp = pDdiTable->pfnTimestampRecordingExp;
  pDdiTable->pfnTimestampRecordingExp =
      ur_tracing_layer::urEnqueueTimestampRecordingExp;

  dditable.pfnNativeCommandExp = pDdiTable->pfnNativeCommandExp;
  pDdiTable->pfnNativeCommandExp = ur_tracing_layer::urEnqueueNativeCommandExp;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Event table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetEventProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_event_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Event;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urEventGetInfo;

  dditable.pfnGetProfilingInfo = pDdiTable->pfnGetProfilingInfo;
  pDdiTable->pfnGetProfilingInfo = ur_tracing_layer::urEventGetProfilingInfo;

  dditable.pfnWait = pDdiTable->pfnWait;
  pDdiTable->pfnWait = ur_tracing_layer::urEventWait;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urEventRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urEventRelease;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urEventGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urEventCreateWithNativeHandle;

  dditable.pfnSetCallback = pDdiTable->pfnSetCallback;
  pDdiTable->pfnSetCallback = ur_tracing_layer::urEventSetCallback;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetKernelProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_kernel_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Kernel;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnCreate = pDdiTable->pfnCreate;
  pDdiTable->pfnCreate = ur_tracing_layer::urKernelCreate;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urKernelGetInfo;

  dditable.pfnGetGroupInfo = pDdiTable->pfnGetGroupInfo;
  pDdiTable->pfnGetGroupInfo = ur_tracing_layer::urKernelGetGroupInfo;

  dditable.pfnGetSubGroupInfo = pDdiTable->pfnGetSubGroupInfo;
  pDdiTable->pfnGetSubGroupInfo = ur_tracing_layer::urKernelGetSubGroupInfo;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urKernelRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urKernelRelease;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urKernelGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urKernelCreateWithNativeHandle;

  dditable.pfnGetSuggestedLocalWorkSize =
      pDdiTable->pfnGetSuggestedLocalWorkSize;
  pDdiTable->pfnGetSuggestedLocalWorkSize =
      ur_tracing_layer::urKernelGetSuggestedLocalWorkSize;

  dditable.pfnSetArgValue = pDdiTable->pfnSetArgValue;
  pDdiTable->pfnSetArgValue = ur_tracing_layer::urKernelSetArgValue;

  dditable.pfnSetArgLocal = pDdiTable->pfnSetArgLocal;
  pDdiTable->pfnSetArgLocal = ur_tracing_layer::urKernelSetArgLocal;

  dditable.pfnSetArgPointer = pDdiTable->pfnSetArgPointer;
  pDdiTable->pfnSetArgPointer = ur_tracing_layer::urKernelSetArgPointer;

  dditable.pfnSetExecInfo = pDdiTable->pfnSetExecInfo;
  pDdiTable->pfnSetExecInfo = ur_tracing_layer::urKernelSetExecInfo;

  dditable.pfnSetArgSampler = pDdiTable->pfnSetArgSampler;
  pDdiTable->pfnSetArgSampler = ur_tracing_layer::urKernelSetArgSampler;

  dditable.pfnSetArgMemObj = pDdiTable->pfnSetArgMemObj;
  pDdiTable->pfnSetArgMemObj = ur_tracing_layer::urKernelSetArgMemObj;

  dditable.pfnSetSpecializationConstants =
      pDdiTable->pfnSetSpecializationConstants;
  pDdiTable->pfnSetSpecializationConstants =
      ur_tracing_layer::urKernelSetSpecializationConstants;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's KernelExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetKernelExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_kernel_exp_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.KernelExp;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnSuggestMaxCooperativeGroupCountExp =
      pDdiTable->pfnSuggestMaxCooperativeGroupCountExp;
  pDdiTable->pfnSuggestMaxCooperativeGroupCountExp =
      ur_tracing_layer::urKernelSuggestMaxCooperativeGroupCountExp;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_mem_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Mem;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnImageCreate = pDdiTable->pfnImageCreate;
  pDdiTable->pfnImageCreate = ur_tracing_layer::urMemImageCreate;

  dditable.pfnBufferCreate = pDdiTable->pfnBufferCreate;
  pDdiTable->pfnBufferCreate = ur_tracing_layer::urMemBufferCreate;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urMemRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urMemRelease;

  dditable.pfnBufferPartition = pDdiTable->pfnBufferPartition;
  pDdiTable->pfnBufferPartition = ur_tracing_layer::urMemBufferPartition;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urMemGetNativeHandle;

  dditable.pfnBufferCreateWithNativeHandle =
      pDdiTable->pfnBufferCreateWithNativeHandle;
  pDdiTable->pfnBufferCreateWithNativeHandle =
      ur_tracing_layer::urMemBufferCreateWithNativeHandle;

  dditable.pfnImageCreateWithNativeHandle =
      pDdiTable->pfnImageCreateWithNativeHandle;
  pDdiTable->pfnImageCreateWithNativeHandle =
      ur_tracing_layer::urMemImageCreateWithNativeHandle;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urMemGetInfo;

  dditable.pfnImageGetInfo = pDdiTable->pfnImageGetInfo;
  pDdiTable->pfnImageGetInfo = ur_tracing_layer::urMemImageGetInfo;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's PhysicalMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_physical_mem_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.PhysicalMem;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnCreate = pDdiTable->pfnCreate;
  pDdiTable->pfnCreate = ur_tracing_layer::urPhysicalMemCreate;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urPhysicalMemRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urPhysicalMemRelease;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urPhysicalMemGetInfo;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Platform table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_platform_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Platform;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnGet = pDdiTable->pfnGet;
  pDdiTable->pfnGet = ur_tracing_layer::urPlatformGet;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urPlatformGetInfo;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urPlatformGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urPlatformCreateWithNativeHandle;

  dditable.pfnGetApiVersion = pDdiTable->pfnGetApiVersion;
  pDdiTable->pfnGetApiVersion = ur_tracing_layer::urPlatformGetApiVersion;

  dditable.pfnGetBackendOption = pDdiTable->pfnGetBackendOption;
  pDdiTable->pfnGetBackendOption = ur_tracing_layer::urPlatformGetBackendOption;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetProgramProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_program_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Program;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnCreateWithIL = pDdiTable->pfnCreateWithIL;
  pDdiTable->pfnCreateWithIL = ur_tracing_layer::urProgramCreateWithIL;

  dditable.pfnCreateWithBinary = pDdiTable->pfnCreateWithBinary;
  pDdiTable->pfnCreateWithBinary = ur_tracing_layer::urProgramCreateWithBinary;

  dditable.pfnBuild = pDdiTable->pfnBuild;
  pDdiTable->pfnBuild = ur_tracing_layer::urProgramBuild;

  dditable.pfnCompile = pDdiTable->pfnCompile;
  pDdiTable->pfnCompile = ur_tracing_layer::urProgramCompile;

  dditable.pfnLink = pDdiTable->pfnLink;
  pDdiTable->pfnLink = ur_tracing_layer::urProgramLink;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urProgramRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urProgramRelease;

  dditable.pfnGetFunctionPointer = pDdiTable->pfnGetFunctionPointer;
  pDdiTable->pfnGetFunctionPointer =
      ur_tracing_layer::urProgramGetFunctionPointer;

  dditable.pfnGetGlobalVariablePointer = pDdiTable->pfnGetGlobalVariablePointer;
  pDdiTable->pfnGetGlobalVariablePointer =
      ur_tracing_layer::urProgramGetGlobalVariablePointer;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urProgramGetInfo;

  dditable.pfnGetBuildInfo = pDdiTable->pfnGetBuildInfo;
  pDdiTable->pfnGetBuildInfo = ur_tracing_layer::urProgramGetBuildInfo;

  dditable.pfnSetSpecializationConstants =
      pDdiTable->pfnSetSpecializationConstants;
  pDdiTable->pfnSetSpecializationConstants =
      ur_tracing_layer::urProgramSetSpecializationConstants;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urProgramGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urProgramCreateWithNativeHandle;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_program_exp_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.ProgramExp;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnBuildExp = pDdiTable->pfnBuildExp;
  pDdiTable->pfnBuildExp = ur_tracing_layer::urProgramBuildExp;

  dditable.pfnCompileExp = pDdiTable->pfnCompileExp;
  pDdiTable->pfnCompileExp = ur_tracing_layer::urProgramCompileExp;

  dditable.pfnLinkExp = pDdiTable->pfnLinkExp;
  pDdiTable->pfnLinkExp = ur_tracing_layer::urProgramLinkExp;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Queue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetQueueProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_queue_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Queue;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urQueueGetInfo;

  dditable.pfnCreate = pDdiTable->pfnCreate;
  pDdiTable->pfnCreate = ur_tracing_layer::urQueueCreate;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urQueueRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urQueueRelease;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urQueueGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urQueueCreateWithNativeHandle;

  dditable.pfnFinish = pDdiTable->pfnFinish;
  pDdiTable->pfnFinish = ur_tracing_layer::urQueueFinish;

  dditable.pfnFlush = pDdiTable->pfnFlush;
  pDdiTable->pfnFlush = ur_tracing_layer::urQueueFlush;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Sampler table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_sampler_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Sampler;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnCreate = pDdiTable->pfnCreate;
  pDdiTable->pfnCreate = ur_tracing_layer::urSamplerCreate;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urSamplerRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urSamplerRelease;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urSamplerGetInfo;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urSamplerGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urSamplerCreateWithNativeHandle;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetUSMProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.USM;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnHostAlloc = pDdiTable->pfnHostAlloc;
  pDdiTable->pfnHostAlloc = ur_tracing_layer::urUSMHostAlloc;

  dditable.pfnDeviceAlloc = pDdiTable->pfnDeviceAlloc;
  pDdiTable->pfnDeviceAlloc = ur_tracing_layer::urUSMDeviceAlloc;

  dditable.pfnSharedAlloc = pDdiTable->pfnSharedAlloc;
  pDdiTable->pfnSharedAlloc = ur_tracing_layer::urUSMSharedAlloc;

  dditable.pfnFree = pDdiTable->pfnFree;
  pDdiTable->pfnFree = ur_tracing_layer::urUSMFree;

  dditable.pfnGetMemAllocInfo = pDdiTable->pfnGetMemAllocInfo;
  pDdiTable->pfnGetMemAllocInfo = ur_tracing_layer::urUSMGetMemAllocInfo;

  dditable.pfnPoolCreate = pDdiTable->pfnPoolCreate;
  pDdiTable->pfnPoolCreate = ur_tracing_layer::urUSMPoolCreate;

  dditable.pfnPoolRetain = pDdiTable->pfnPoolRetain;
  pDdiTable->pfnPoolRetain = ur_tracing_layer::urUSMPoolRetain;

  dditable.pfnPoolRelease = pDdiTable->pfnPoolRelease;
  pDdiTable->pfnPoolRelease = ur_tracing_layer::urUSMPoolRelease;

  dditable.pfnPoolGetInfo = pDdiTable->pfnPoolGetInfo;
  pDdiTable->pfnPoolGetInfo = ur_tracing_layer::urUSMPoolGetInfo;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USMExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_exp_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.USMExp;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnPoolCreateExp = pDdiTable->pfnPoolCreateExp;
  pDdiTable->pfnPoolCreateExp = ur_tracing_layer::urUSMPoolCreateExp;

  dditable.pfnPoolDestroyExp = pDdiTable->pfnPoolDestroyExp;
  pDdiTable->pfnPoolDestroyExp = ur_tracing_layer::urUSMPoolDestroyExp;

  dditable.pfnPoolSetThresholdExp = pDdiTable->pfnPoolSetThresholdExp;
  pDdiTable->pfnPoolSetThresholdExp =
      ur_tracing_layer::urUSMPoolSetThresholdExp;

  dditable.pfnPoolGetDefaultDevicePoolExp =
      pDdiTable->pfnPoolGetDefaultDevicePoolExp;
  pDdiTable->pfnPoolGetDefaultDevicePoolExp =
      ur_tracing_layer::urUSMPoolGetDefaultDevicePoolExp;

  dditable.pfnPoolGetInfoExp = pDdiTable->pfnPoolGetInfoExp;
  pDdiTable->pfnPoolGetInfoExp = ur_tracing_layer::urUSMPoolGetInfoExp;

  dditable.pfnPoolSetDevicePoolExp = pDdiTable->pfnPoolSetDevicePoolExp;
  pDdiTable->pfnPoolSetDevicePoolExp =
      ur_tracing_layer::urUSMPoolSetDevicePoolExp;

  dditable.pfnPoolGetDevicePoolExp = pDdiTable->pfnPoolGetDevicePoolExp;
  pDdiTable->pfnPoolGetDevicePoolExp =
      ur_tracing_layer::urUSMPoolGetDevicePoolExp;

  dditable.pfnPoolTrimToExp = pDdiTable->pfnPoolTrimToExp;
  pDdiTable->pfnPoolTrimToExp = ur_tracing_layer::urUSMPoolTrimToExp;

  dditable.pfnPitchedAllocExp = pDdiTable->pfnPitchedAllocExp;
  pDdiTable->pfnPitchedAllocExp = ur_tracing_layer::urUSMPitchedAllocExp;

  dditable.pfnImportExp = pDdiTable->pfnImportExp;
  pDdiTable->pfnImportExp = ur_tracing_layer::urUSMImportExp;

  dditable.pfnReleaseExp = pDdiTable->pfnReleaseExp;
  pDdiTable->pfnReleaseExp = ur_tracing_layer::urUSMReleaseExp;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's UsmP2PExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_p2p_exp_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.UsmP2PExp;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnEnablePeerAccessExp = pDdiTable->pfnEnablePeerAccessExp;
  pDdiTable->pfnEnablePeerAccessExp =
      ur_tracing_layer::urUsmP2PEnablePeerAccessExp;

  dditable.pfnDisablePeerAccessExp = pDdiTable->pfnDisablePeerAccessExp;
  pDdiTable->pfnDisablePeerAccessExp =
      ur_tracing_layer::urUsmP2PDisablePeerAccessExp;

  dditable.pfnPeerAccessGetInfoExp = pDdiTable->pfnPeerAccessGetInfoExp;
  pDdiTable->pfnPeerAccessGetInfoExp =
      ur_tracing_layer::urUsmP2PPeerAccessGetInfoExp;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's VirtualMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_virtual_mem_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.VirtualMem;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnGranularityGetInfo = pDdiTable->pfnGranularityGetInfo;
  pDdiTable->pfnGranularityGetInfo =
      ur_tracing_layer::urVirtualMemGranularityGetInfo;

  dditable.pfnReserve = pDdiTable->pfnReserve;
  pDdiTable->pfnReserve = ur_tracing_layer::urVirtualMemReserve;

  dditable.pfnFree = pDdiTable->pfnFree;
  pDdiTable->pfnFree = ur_tracing_layer::urVirtualMemFree;

  dditable.pfnMap = pDdiTable->pfnMap;
  pDdiTable->pfnMap = ur_tracing_layer::urVirtualMemMap;

  dditable.pfnUnmap = pDdiTable->pfnUnmap;
  pDdiTable->pfnUnmap = ur_tracing_layer::urVirtualMemUnmap;

  dditable.pfnSetAccess = pDdiTable->pfnSetAccess;
  pDdiTable->pfnSetAccess = ur_tracing_layer::urVirtualMemSetAccess;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urVirtualMemGetInfo;

  return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Device table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_device_dditable_t *pDdiTable) {
  auto &dditable = ur_tracing_layer::getContext()->urDdiTable.Device;

  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (UR_MAJOR_VERSION(ur_tracing_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_tracing_layer::getContext()->version) >
          UR_MINOR_VERSION(version))
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  dditable.pfnGet = pDdiTable->pfnGet;
  pDdiTable->pfnGet = ur_tracing_layer::urDeviceGet;

  dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
  pDdiTable->pfnGetInfo = ur_tracing_layer::urDeviceGetInfo;

  dditable.pfnRetain = pDdiTable->pfnRetain;
  pDdiTable->pfnRetain = ur_tracing_layer::urDeviceRetain;

  dditable.pfnRelease = pDdiTable->pfnRelease;
  pDdiTable->pfnRelease = ur_tracing_layer::urDeviceRelease;

  dditable.pfnPartition = pDdiTable->pfnPartition;
  pDdiTable->pfnPartition = ur_tracing_layer::urDevicePartition;

  dditable.pfnSelectBinary = pDdiTable->pfnSelectBinary;
  pDdiTable->pfnSelectBinary = ur_tracing_layer::urDeviceSelectBinary;

  dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
  pDdiTable->pfnGetNativeHandle = ur_tracing_layer::urDeviceGetNativeHandle;

  dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_tracing_layer::urDeviceCreateWithNativeHandle;

  dditable.pfnGetGlobalTimestamps = pDdiTable->pfnGetGlobalTimestamps;
  pDdiTable->pfnGetGlobalTimestamps =
      ur_tracing_layer::urDeviceGetGlobalTimestamps;

  return result;
}

ur_result_t context_t::init(ur_dditable_t *dditable,
                            const std::set<std::string> &enabledLayerNames,
                            codeloc_data codelocData) {
  ur_result_t result = UR_RESULT_SUCCESS;

  if (!enabledLayerNames.count(name)) {
    return result;
  }

  // Recreate the logger in case env variables have been modified between
  // program launch and the call to `urLoaderInit`
  logger = logger::create_logger("tracing", true, true);

  ur_tracing_layer::getContext()->codelocData = codelocData;

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetGlobalProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &dditable->Global);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetBindlessImagesExpProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->BindlessImagesExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetCommandBufferExpProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->CommandBufferExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetContextProcAddrTable(UR_API_VERSION_CURRENT,
                                                         &dditable->Context);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetEnqueueProcAddrTable(UR_API_VERSION_CURRENT,
                                                         &dditable->Enqueue);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetEnqueueExpProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->EnqueueExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetEventProcAddrTable(UR_API_VERSION_CURRENT,
                                                       &dditable->Event);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetKernelProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &dditable->Kernel);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetKernelExpProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->KernelExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetMemProcAddrTable(UR_API_VERSION_CURRENT,
                                                     &dditable->Mem);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetPhysicalMemProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->PhysicalMem);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetPlatformProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->Platform);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetProgramProcAddrTable(UR_API_VERSION_CURRENT,
                                                         &dditable->Program);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetProgramExpProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->ProgramExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetQueueProcAddrTable(UR_API_VERSION_CURRENT,
                                                       &dditable->Queue);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetSamplerProcAddrTable(UR_API_VERSION_CURRENT,
                                                         &dditable->Sampler);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetUSMProcAddrTable(UR_API_VERSION_CURRENT,
                                                     &dditable->USM);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetUSMExpProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &dditable->USMExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetUsmP2PExpProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->UsmP2PExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetVirtualMemProcAddrTable(
        UR_API_VERSION_CURRENT, &dditable->VirtualMem);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_tracing_layer::urGetDeviceProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &dditable->Device);
  }

  return result;
}
} /* namespace ur_tracing_layer */
