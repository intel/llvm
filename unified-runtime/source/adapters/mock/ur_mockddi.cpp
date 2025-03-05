/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT

 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_mockddi.cpp
 *
 */
#include "ur_mock.hpp"
#include "ur_mock_helpers.hpp"

namespace driver {
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
    uint32_t *pNumAdapters) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_adapter_get_params_t params = {&NumEntries, &phAdapters, &pNumAdapters};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urAdapterGet"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urAdapterGet"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    if (pNumAdapters) {
      *pNumAdapters = 1;
    }
    if (phAdapters) {
      *phAdapters = d_context.adapter;
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urAdapterGet"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRelease
__urdlllocal ur_result_t UR_APICALL urAdapterRelease(
    /// [in][release] Adapter handle to release
    ur_adapter_handle_t hAdapter) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_adapter_release_params_t params = {&hAdapter};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urAdapterRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urAdapterRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urAdapterRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRetain
__urdlllocal ur_result_t UR_APICALL urAdapterRetain(
    /// [in][retain] Adapter handle to retain
    ur_adapter_handle_t hAdapter) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_adapter_retain_params_t params = {&hAdapter};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urAdapterRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urAdapterRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urAdapterRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    int32_t *pError) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_adapter_get_last_error_params_t params = {&hAdapter, &ppMessage, &pError};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urAdapterGetLastError"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urAdapterGetLastError"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urAdapterGetLastError"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_adapter_get_info_params_t params = {&hAdapter, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urAdapterGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urAdapterGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urAdapterGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    uint32_t *pNumPlatforms) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_platform_get_params_t params = {&phAdapters, &NumAdapters, &NumEntries,
                                     &phPlatforms, &pNumPlatforms};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPlatformGet"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPlatformGet"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    if (pNumPlatforms) {
      *pNumPlatforms = 1;
    }
    if (phPlatforms) {
      *phPlatforms = d_context.platform;
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPlatformGet"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_platform_get_info_params_t params = {&hPlatform, &propName, &propSize,
                                          &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPlatformGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPlatformGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPlatformGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetApiVersion
__urdlllocal ur_result_t UR_APICALL urPlatformGetApiVersion(
    /// [in] handle of the platform
    ur_platform_handle_t hPlatform,
    /// [out] api version
    ur_api_version_t *pVersion) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_platform_get_api_version_params_t params = {&hPlatform, &pVersion};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPlatformGetApiVersion"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPlatformGetApiVersion"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPlatformGetApiVersion"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformGetNativeHandle(
    /// [in] handle of the platform.
    ur_platform_handle_t hPlatform,
    /// [out] a pointer to the native handle of the platform.
    ur_native_handle_t *phNativePlatform) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_platform_get_native_handle_params_t params = {&hPlatform,
                                                   &phNativePlatform};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPlatformGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPlatformGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativePlatform = reinterpret_cast<ur_native_handle_t>(hPlatform);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPlatformGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_platform_handle_t *phPlatform) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_platform_create_with_native_handle_params_t params = {
      &hNativePlatform, &hAdapter, &pProperties, &phPlatform};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urPlatformCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urPlatformCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phPlatform = reinterpret_cast<ur_platform_handle_t>(hNativePlatform);
    mock::retainDummyHandle(*phPlatform);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urPlatformCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const char **ppPlatformOption) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_platform_get_backend_option_params_t params = {
      &hPlatform, &pFrontendOption, &ppPlatformOption};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPlatformGetBackendOption"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPlatformGetBackendOption"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPlatformGetBackendOption"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    uint32_t *pNumDevices) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_get_params_t params = {&hPlatform, &DeviceType, &NumEntries,
                                   &phDevices, &pNumDevices};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDeviceGet"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDeviceGet"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    if (pNumDevices) {
      *pNumDevices = 1;
    }
    if (phDevices) {
      *phDevices = d_context.device;
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDeviceGet"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_get_info_params_t params = {&hDevice, &propName, &propSize,
                                        &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDeviceGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDeviceGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDeviceGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRetain
__urdlllocal ur_result_t UR_APICALL urDeviceRetain(
    /// [in][retain] handle of the device to get a reference of.
    ur_device_handle_t hDevice) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_retain_params_t params = {&hDevice};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDeviceRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDeviceRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDeviceRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRelease
__urdlllocal ur_result_t UR_APICALL urDeviceRelease(
    /// [in][release] handle of the device to release.
    ur_device_handle_t hDevice) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_release_params_t params = {&hDevice};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDeviceRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDeviceRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDeviceRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    uint32_t *pNumDevicesRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_partition_params_t params = {&hDevice, &pProperties, &NumDevices,
                                         &phSubDevices, &pNumDevicesRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDevicePartition"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDevicePartition"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDevicePartition"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    uint32_t *pSelectedBinary) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_select_binary_params_t params = {&hDevice, &pBinaries, &NumBinaries,
                                             &pSelectedBinary};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDeviceSelectBinary"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDeviceSelectBinary"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDeviceSelectBinary"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceGetNativeHandle(
    /// [in] handle of the device.
    ur_device_handle_t hDevice,
    /// [out] a pointer to the native handle of the device.
    ur_native_handle_t *phNativeDevice) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_get_native_handle_params_t params = {&hDevice, &phNativeDevice};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDeviceGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDeviceGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeDevice = reinterpret_cast<ur_native_handle_t>(hDevice);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDeviceGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_device_handle_t *phDevice) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_create_with_native_handle_params_t params = {
      &hNativeDevice, &hAdapter, &pProperties, &phDevice};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urDeviceCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urDeviceCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phDevice = reinterpret_cast<ur_device_handle_t>(hNativeDevice);
    mock::retainDummyHandle(*phDevice);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urDeviceCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    uint64_t *pHostTimestamp) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_device_get_global_timestamps_params_t params = {
      &hDevice, &pDeviceTimestamp, &pHostTimestamp};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urDeviceGetGlobalTimestamps"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urDeviceGetGlobalTimestamps"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urDeviceGetGlobalTimestamps"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_context_handle_t *phContext) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_context_create_params_t params = {&DeviceCount, &phDevices, &pProperties,
                                       &phContext};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urContextCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urContextCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phContext = mock::createDummyHandle<ur_context_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urContextCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL urContextRetain(
    /// [in][retain] handle of the context to get a reference of.
    ur_context_handle_t hContext) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_context_retain_params_t params = {&hContext};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urContextRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urContextRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hContext);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urContextRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL urContextRelease(
    /// [in][release] handle of the context to release.
    ur_context_handle_t hContext) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_context_release_params_t params = {&hContext};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urContextRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urContextRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hContext);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urContextRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_context_get_info_params_t params = {&hContext, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urContextGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urContextGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urContextGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextGetNativeHandle(
    /// [in] handle of the context.
    ur_context_handle_t hContext,
    /// [out] a pointer to the native handle of the context.
    ur_native_handle_t *phNativeContext) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_context_get_native_handle_params_t params = {&hContext, &phNativeContext};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urContextGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urContextGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeContext = reinterpret_cast<ur_native_handle_t>(hContext);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urContextGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_context_handle_t *phContext) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_context_create_with_native_handle_params_t params = {
      &hNativeContext, &hAdapter,    &numDevices,
      &phDevices,      &pProperties, &phContext};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urContextCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urContextCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phContext = reinterpret_cast<ur_context_handle_t>(hNativeContext);
    mock::retainDummyHandle(*phContext);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urContextCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextSetExtendedDeleter
__urdlllocal ur_result_t UR_APICALL urContextSetExtendedDeleter(
    /// [in] handle of the context.
    ur_context_handle_t hContext,
    /// [in] Function pointer to extended deleter.
    ur_context_extended_deleter_t pfnDeleter,
    /// [in][out][optional] pointer to data to be passed to callback.
    void *pUserData) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_context_set_extended_deleter_params_t params = {&hContext, &pfnDeleter,
                                                     &pUserData};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urContextSetExtendedDeleter"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urContextSetExtendedDeleter"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urContextSetExtendedDeleter"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_mem_handle_t *phMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_image_create_params_t params = {&hContext,   &flags, &pImageFormat,
                                         &pImageDesc, &pHost, &phMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemImageCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemImageCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phMem = mock::createDummyHandle<ur_mem_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemImageCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_mem_handle_t *phBuffer) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_buffer_create_params_t params = {&hContext, &flags, &size,
                                          &pProperties, &phBuffer};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemBufferCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemBufferCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    if (pProperties && (pProperties)->pHost &&
        flags & UR_MEM_FLAG_USE_HOST_POINTER) {
      *phBuffer = mock::createDummyHandleWithData<ur_mem_handle_t>(
          reinterpret_cast<unsigned char *>((pProperties)->pHost), size);
    } else {
      *phBuffer = mock::createDummyHandle<ur_mem_handle_t>(size);
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemBufferCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL urMemRetain(
    /// [in][retain] handle of the memory object to get access
    ur_mem_handle_t hMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_retain_params_t params = {&hMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL urMemRelease(
    /// [in][release] handle of the memory object to release
    ur_mem_handle_t hMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_release_params_t params = {&hMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_mem_handle_t *phMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_buffer_partition_params_t params = {
      &hBuffer, &flags, &bufferCreateType, &pRegion, &phMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemBufferPartition"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemBufferPartition"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phMem = mock::createDummyHandle<ur_mem_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemBufferPartition"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_native_handle_t *phNativeMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_get_native_handle_params_t params = {&hMem, &hDevice, &phNativeMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeMem = reinterpret_cast<ur_native_handle_t>(hMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_mem_handle_t *phMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_buffer_create_with_native_handle_params_t params = {
      &hNativeMem, &hContext, &pProperties, &phMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urMemBufferCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urMemBufferCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phMem = reinterpret_cast<ur_mem_handle_t>(hNativeMem);
    mock::retainDummyHandle(*phMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urMemBufferCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_mem_handle_t *phMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_image_create_with_native_handle_params_t params = {
      &hNativeMem, &hContext, &pImageFormat, &pImageDesc, &pProperties, &phMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urMemImageCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urMemImageCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phMem = reinterpret_cast<ur_mem_handle_t>(hNativeMem);
    mock::retainDummyHandle(*phMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urMemImageCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_get_info_params_t params = {&hMemory, &propName, &propSize,
                                     &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_mem_image_get_info_params_t params = {&hMemory, &propName, &propSize,
                                           &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urMemImageGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urMemImageGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urMemImageGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerCreate
__urdlllocal ur_result_t UR_APICALL urSamplerCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to the sampler description
    const ur_sampler_desc_t *pDesc,
    /// [out][alloc] pointer to handle of sampler object created
    ur_sampler_handle_t *phSampler) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_sampler_create_params_t params = {&hContext, &pDesc, &phSampler};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urSamplerCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urSamplerCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phSampler = mock::createDummyHandle<ur_sampler_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urSamplerCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRetain
__urdlllocal ur_result_t UR_APICALL urSamplerRetain(
    /// [in][retain] handle of the sampler object to get access
    ur_sampler_handle_t hSampler) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_sampler_retain_params_t params = {&hSampler};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urSamplerRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urSamplerRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hSampler);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urSamplerRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRelease
__urdlllocal ur_result_t UR_APICALL urSamplerRelease(
    /// [in][release] handle of the sampler object to release
    ur_sampler_handle_t hSampler) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_sampler_release_params_t params = {&hSampler};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urSamplerRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urSamplerRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hSampler);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urSamplerRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_sampler_get_info_params_t params = {&hSampler, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urSamplerGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urSamplerGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urSamplerGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerGetNativeHandle(
    /// [in] handle of the sampler.
    ur_sampler_handle_t hSampler,
    /// [out] a pointer to the native handle of the sampler.
    ur_native_handle_t *phNativeSampler) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_sampler_get_native_handle_params_t params = {&hSampler, &phNativeSampler};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urSamplerGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urSamplerGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeSampler = reinterpret_cast<ur_native_handle_t>(hSampler);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urSamplerGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_sampler_handle_t *phSampler) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_sampler_create_with_native_handle_params_t params = {
      &hNativeSampler, &hContext, &pProperties, &phSampler};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urSamplerCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urSamplerCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phSampler = reinterpret_cast<ur_sampler_handle_t>(hNativeSampler);
    mock::retainDummyHandle(*phSampler);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urSamplerCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_host_alloc_params_t params = {&hContext, &pUSMDesc, &pool, &size,
                                       &ppMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMHostAlloc"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMHostAlloc"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *ppMem = mock::createDummyHandle<void *>(size);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMHostAlloc"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_device_alloc_params_t params = {&hContext, &hDevice, &pUSMDesc,
                                         &pool,     &size,    &ppMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMDeviceAlloc"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMDeviceAlloc"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *ppMem = mock::createDummyHandle<void *>(size);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMDeviceAlloc"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_shared_alloc_params_t params = {&hContext, &hDevice, &pUSMDesc,
                                         &pool,     &size,    &ppMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMSharedAlloc"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMSharedAlloc"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *ppMem = mock::createDummyHandle<void *>(size);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMSharedAlloc"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to USM memory object
    void *pMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_free_params_t params = {&hContext, &pMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMFree"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMFree"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(pMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMFree"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_get_mem_alloc_info_params_t params = {
      &hContext, &pMem, &propName, &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMGetMemAllocInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMGetMemAllocInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMGetMemAllocInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_usm_pool_handle_t *ppPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_create_params_t params = {&hContext, &pPoolDesc, &ppPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *ppPool = mock::createDummyHandle<ur_usm_pool_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRetain
__urdlllocal ur_result_t UR_APICALL urUSMPoolRetain(
    /// [in][retain] pointer to USM memory pool
    ur_usm_pool_handle_t pPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_retain_params_t params = {&pPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(pPool);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRelease
__urdlllocal ur_result_t UR_APICALL urUSMPoolRelease(
    /// [in][release] pointer to USM memory pool
    ur_usm_pool_handle_t pPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_release_params_t params = {&pPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(pPool);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_get_info_params_t params = {&hPool, &propName, &propSize,
                                          &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_virtual_mem_granularity_get_info_params_t params = {
      &hContext, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urVirtualMemGranularityGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urVirtualMemGranularityGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urVirtualMemGranularityGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppStart) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_virtual_mem_reserve_params_t params = {&hContext, &pStart, &size,
                                            &ppStart};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urVirtualMemReserve"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urVirtualMemReserve"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urVirtualMemReserve"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemFree
__urdlllocal ur_result_t UR_APICALL urVirtualMemFree(
    /// [in] handle of the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to the start of the virtual memory range to free.
    const void *pStart,
    /// [in] size in bytes of the virtual memory range to free.
    size_t size) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_virtual_mem_free_params_t params = {&hContext, &pStart, &size};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urVirtualMemFree"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urVirtualMemFree"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urVirtualMemFree"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_virtual_mem_access_flags_t flags) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_virtual_mem_map_params_t params = {&hContext,     &pStart, &size,
                                        &hPhysicalMem, &offset, &flags};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urVirtualMemMap"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urVirtualMemMap"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urVirtualMemMap"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urVirtualMemUnmap
__urdlllocal ur_result_t UR_APICALL urVirtualMemUnmap(
    /// [in] handle to the context object.
    ur_context_handle_t hContext,
    /// [in] pointer to the start of the mapped virtual memory range
    const void *pStart,
    /// [in] size in bytes of the virtual memory range.
    size_t size) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_virtual_mem_unmap_params_t params = {&hContext, &pStart, &size};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urVirtualMemUnmap"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urVirtualMemUnmap"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urVirtualMemUnmap"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_virtual_mem_access_flags_t flags) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_virtual_mem_set_access_params_t params = {&hContext, &pStart, &size,
                                               &flags};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urVirtualMemSetAccess"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urVirtualMemSetAccess"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urVirtualMemSetAccess"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_virtual_mem_get_info_params_t params = {
      &hContext, &pStart,     &size,        &propName,
      &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urVirtualMemGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urVirtualMemGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urVirtualMemGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_physical_mem_handle_t *phPhysicalMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_physical_mem_create_params_t params = {&hContext, &hDevice, &size,
                                            &pProperties, &phPhysicalMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPhysicalMemCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPhysicalMemCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phPhysicalMem = mock::createDummyHandle<ur_physical_mem_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPhysicalMemCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRetain
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRetain(
    /// [in][retain] handle of the physical memory object to retain.
    ur_physical_mem_handle_t hPhysicalMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_physical_mem_retain_params_t params = {&hPhysicalMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPhysicalMemRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPhysicalMemRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hPhysicalMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPhysicalMemRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRelease
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRelease(
    /// [in][release] handle of the physical memory object to release.
    ur_physical_mem_handle_t hPhysicalMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_physical_mem_release_params_t params = {&hPhysicalMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPhysicalMemRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPhysicalMemRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hPhysicalMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPhysicalMemRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_physical_mem_get_info_params_t params = {
      &hPhysicalMem, &propName, &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urPhysicalMemGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urPhysicalMemGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urPhysicalMemGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_program_handle_t *phProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_create_with_il_params_t params = {&hContext, &pIL, &length,
                                               &pProperties, &phProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramCreateWithIL"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramCreateWithIL"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phProgram = mock::createDummyHandle<ur_program_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramCreateWithIL"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_program_handle_t *phProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_create_with_binary_params_t params = {
      &hContext,   &numDevices,  &phDevices, &pLengths,
      &ppBinaries, &pProperties, &phProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramCreateWithBinary"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramCreateWithBinary"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phProgram = mock::createDummyHandle<ur_program_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramCreateWithBinary"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuild
__urdlllocal ur_result_t UR_APICALL urProgramBuild(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in] Handle of the program to build.
    ur_program_handle_t hProgram,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_build_params_t params = {&hContext, &hProgram, &pOptions};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramBuild"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramBuild"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramBuild"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCompile
__urdlllocal ur_result_t UR_APICALL urProgramCompile(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in][out] handle of the program to compile.
    ur_program_handle_t hProgram,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_compile_params_t params = {&hContext, &hProgram, &pOptions};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramCompile"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramCompile"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramCompile"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_program_handle_t *phProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }

  ur_program_link_params_t params = {&hContext, &count, &phPrograms, &pOptions,
                                     &phProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramLink"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramLink"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phProgram = mock::createDummyHandle<ur_program_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramLink"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL urProgramRetain(
    /// [in][retain] handle for the Program to retain
    ur_program_handle_t hProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_retain_params_t params = {&hProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hProgram);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
__urdlllocal ur_result_t UR_APICALL urProgramRelease(
    /// [in][release] handle for the Program to release
    ur_program_handle_t hProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_release_params_t params = {&hProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hProgram);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppFunctionPointer) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_get_function_pointer_params_t params = {
      &hDevice, &hProgram, &pFunctionName, &ppFunctionPointer};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramGetFunctionPointer"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramGetFunctionPointer"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramGetFunctionPointer"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppGlobalVariablePointerRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_get_global_variable_pointer_params_t params = {
      &hDevice, &hProgram, &pGlobalVariableName, &pGlobalVariableSizeRet,
      &ppGlobalVariablePointerRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urProgramGetGlobalVariablePointer"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urProgramGetGlobalVariablePointer"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urProgramGetGlobalVariablePointer"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_get_info_params_t params = {&hProgram, &propName, &propSize,
                                         &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_get_build_info_params_t params = {
      &hProgram, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramGetBuildInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramGetBuildInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramGetBuildInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const ur_specialization_constant_info_t *pSpecConstants) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_set_specialization_constants_params_t params = {&hProgram, &count,
                                                             &pSpecConstants};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urProgramSetSpecializationConstants"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urProgramSetSpecializationConstants"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urProgramSetSpecializationConstants"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramGetNativeHandle(
    /// [in] handle of the program.
    ur_program_handle_t hProgram,
    /// [out] a pointer to the native handle of the program.
    ur_native_handle_t *phNativeProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_get_native_handle_params_t params = {&hProgram, &phNativeProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeProgram = reinterpret_cast<ur_native_handle_t>(hProgram);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_program_handle_t *phProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_create_with_native_handle_params_t params = {
      &hNativeProgram, &hContext, &pProperties, &phProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urProgramCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urProgramCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phProgram = reinterpret_cast<ur_program_handle_t>(hNativeProgram);
    mock::retainDummyHandle(*phProgram);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urProgramCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
__urdlllocal ur_result_t UR_APICALL urKernelCreate(
    /// [in] handle of the program instance
    ur_program_handle_t hProgram,
    /// [in] pointer to null-terminated string.
    const char *pKernelName,
    /// [out][alloc] pointer to handle of kernel object created.
    ur_kernel_handle_t *phKernel) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_create_params_t params = {&hProgram, &pKernelName, &phKernel};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phKernel = mock::createDummyHandle<ur_kernel_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const void *pArgValue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_set_arg_value_params_t params = {&hKernel, &argIndex, &argSize,
                                             &pProperties, &pArgValue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelSetArgValue"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelSetArgValue"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelSetArgValue"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const ur_kernel_arg_local_properties_t *pProperties) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_set_arg_local_params_t params = {&hKernel, &argIndex, &argSize,
                                             &pProperties};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelSetArgLocal"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelSetArgLocal"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelSetArgLocal"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_get_info_params_t params = {&hKernel, &propName, &propSize,
                                        &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_get_group_info_params_t params = {
      &hKernel, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelGetGroupInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelGetGroupInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelGetGroupInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_get_sub_group_info_params_t params = {
      &hKernel, &hDevice, &propName, &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelGetSubGroupInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelGetSubGroupInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelGetSubGroupInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL urKernelRetain(
    /// [in][retain] handle for the Kernel to retain
    ur_kernel_handle_t hKernel) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_retain_params_t params = {&hKernel};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hKernel);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t UR_APICALL urKernelRelease(
    /// [in][release] handle for the Kernel to release
    ur_kernel_handle_t hKernel) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_release_params_t params = {&hKernel};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hKernel);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const void *pArgValue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_set_arg_pointer_params_t params = {&hKernel, &argIndex,
                                               &pProperties, &pArgValue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelSetArgPointer"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelSetArgPointer"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelSetArgPointer"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const void *pPropValue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_set_exec_info_params_t params = {&hKernel, &propName, &propSize,
                                             &pProperties, &pPropValue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelSetExecInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelSetExecInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelSetExecInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_sampler_handle_t hArgValue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_set_arg_sampler_params_t params = {&hKernel, &argIndex,
                                               &pProperties, &hArgValue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelSetArgSampler"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelSetArgSampler"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelSetArgSampler"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_mem_handle_t hArgValue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_set_arg_mem_obj_params_t params = {&hKernel, &argIndex,
                                               &pProperties, &hArgValue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelSetArgMemObj"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelSetArgMemObj"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelSetArgMemObj"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetSpecializationConstants
__urdlllocal ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] the number of elements in the pSpecConstants array
    uint32_t count,
    /// [in] array of specialization constant value descriptions
    const ur_specialization_constant_info_t *pSpecConstants) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_set_specialization_constants_params_t params = {&hKernel, &count,
                                                            &pSpecConstants};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urKernelSetSpecializationConstants"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urKernelSetSpecializationConstants"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urKernelSetSpecializationConstants"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelGetNativeHandle(
    /// [in] handle of the kernel.
    ur_kernel_handle_t hKernel,
    /// [out] a pointer to the native handle of the kernel.
    ur_native_handle_t *phNativeKernel) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_get_native_handle_params_t params = {&hKernel, &phNativeKernel};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urKernelGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urKernelGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeKernel = reinterpret_cast<ur_native_handle_t>(hKernel);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urKernelGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_kernel_handle_t *phKernel) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_create_with_native_handle_params_t params = {
      &hNativeKernel, &hContext, &hProgram, &pProperties, &phKernel};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urKernelCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urKernelCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phKernel = reinterpret_cast<ur_kernel_handle_t>(hNativeKernel);
    mock::retainDummyHandle(*phKernel);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urKernelCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pSuggestedLocalWorkSize) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_get_suggested_local_work_size_params_t params = {
      &hKernel,           &hQueue,          &numWorkDim,
      &pGlobalWorkOffset, &pGlobalWorkSize, &pSuggestedLocalWorkSize};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urKernelGetSuggestedLocalWorkSize"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urKernelGetSuggestedLocalWorkSize"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urKernelGetSuggestedLocalWorkSize"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_get_info_params_t params = {&hQueue, &propName, &propSize,
                                       &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urQueueGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urQueueGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_queue_handle_t *phQueue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_create_params_t params = {&hContext, &hDevice, &pProperties,
                                     &phQueue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urQueueCreate"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urQueueCreate"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phQueue = mock::createDummyHandle<ur_queue_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueCreate"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRetain
__urdlllocal ur_result_t UR_APICALL urQueueRetain(
    /// [in][retain] handle of the queue object to get access
    ur_queue_handle_t hQueue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_retain_params_t params = {&hQueue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urQueueRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urQueueRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hQueue);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRelease
__urdlllocal ur_result_t UR_APICALL urQueueRelease(
    /// [in][release] handle of the queue object to release
    ur_queue_handle_t hQueue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_release_params_t params = {&hQueue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urQueueRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urQueueRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hQueue);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urQueueGetNativeHandle(
    /// [in] handle of the queue.
    ur_queue_handle_t hQueue,
    /// [in][optional] pointer to native descriptor
    ur_queue_native_desc_t *pDesc,
    /// [out] a pointer to the native handle of the queue.
    ur_native_handle_t *phNativeQueue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_get_native_handle_params_t params = {&hQueue, &pDesc,
                                                &phNativeQueue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urQueueGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urQueueGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeQueue = reinterpret_cast<ur_native_handle_t>(hQueue);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_queue_handle_t *phQueue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_create_with_native_handle_params_t params = {
      &hNativeQueue, &hContext, &hDevice, &pProperties, &phQueue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urQueueCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urQueueCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phQueue = reinterpret_cast<ur_queue_handle_t>(hNativeQueue);
    mock::retainDummyHandle(*phQueue);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFinish
__urdlllocal ur_result_t UR_APICALL urQueueFinish(
    /// [in] handle of the queue to be finished.
    ur_queue_handle_t hQueue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_finish_params_t params = {&hQueue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urQueueFinish"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urQueueFinish"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueFinish"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFlush
__urdlllocal ur_result_t UR_APICALL urQueueFlush(
    /// [in] handle of the queue to be flushed.
    ur_queue_handle_t hQueue) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_queue_flush_params_t params = {&hQueue};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urQueueFlush"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urQueueFlush"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urQueueFlush"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_get_info_params_t params = {&hEvent, &propName, &propSize,
                                       &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEventGetInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEventGetInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventGetInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_get_profiling_info_params_t params = {&hEvent, &propName, &propSize,
                                                 &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEventGetProfilingInfo"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEventGetProfilingInfo"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventGetProfilingInfo"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventWait
__urdlllocal ur_result_t UR_APICALL urEventWait(
    /// [in] number of events in the event list
    uint32_t numEvents,
    /// [in][range(0, numEvents)] pointer to a list of events to wait for
    /// completion
    const ur_event_handle_t *phEventWaitList) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_wait_params_t params = {&numEvents, &phEventWaitList};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEventWait"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEventWait"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventWait"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRetain
__urdlllocal ur_result_t UR_APICALL urEventRetain(
    /// [in][retain] handle of the event object
    ur_event_handle_t hEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_retain_params_t params = {&hEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEventRetain"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEventRetain"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hEvent);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventRetain"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRelease
__urdlllocal ur_result_t UR_APICALL urEventRelease(
    /// [in][release] handle of the event object
    ur_event_handle_t hEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_release_params_t params = {&hEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEventRelease"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEventRelease"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hEvent);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventRelease"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventGetNativeHandle(
    /// [in] handle of the event.
    ur_event_handle_t hEvent,
    /// [out] a pointer to the native handle of the event.
    ur_native_handle_t *phNativeEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_get_native_handle_params_t params = {&hEvent, &phNativeEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEventGetNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEventGetNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phNativeEvent = reinterpret_cast<ur_native_handle_t>(hEvent);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventGetNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_create_with_native_handle_params_t params = {
      &hNativeEvent, &hContext, &pProperties, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEventCreateWithNativeHandle"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEventCreateWithNativeHandle"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phEvent = reinterpret_cast<ur_event_handle_t>(hNativeEvent);
    mock::retainDummyHandle(*phEvent);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventCreateWithNativeHandle"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void *pUserData) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_event_set_callback_params_t params = {&hEvent, &execStatus, &pfnNotify,
                                           &pUserData};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEventSetCallback"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEventSetCallback"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEventSetCallback"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_kernel_launch_params_t params = {&hQueue,
                                              &hKernel,
                                              &workDim,
                                              &pGlobalWorkOffset,
                                              &pGlobalWorkSize,
                                              &pLocalWorkSize,
                                              &numEventsInWaitList,
                                              &phEventWaitList,
                                              &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueKernelLaunch"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueKernelLaunch"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueKernelLaunch"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_events_wait_params_t params = {&hQueue, &numEventsInWaitList,
                                            &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueEventsWait"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueEventsWait"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueEventsWait"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_events_wait_with_barrier_params_t params = {
      &hQueue, &numEventsInWaitList, &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEnqueueEventsWaitWithBarrier"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEnqueueEventsWaitWithBarrier"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urEnqueueEventsWaitWithBarrier"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_read_params_t params = {
      &hQueue, &hBuffer, &blockingRead,        &offset,
      &size,   &pDst,    &numEventsInWaitList, &phEventWaitList,
      &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferRead"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferRead"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferRead"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_write_params_t params = {
      &hQueue, &hBuffer, &blockingWrite,       &offset,
      &size,   &pSrc,    &numEventsInWaitList, &phEventWaitList,
      &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferWrite"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferWrite"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferWrite"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_read_rect_params_t params = {
      &hQueue,          &hBuffer,        &blockingRead,   &bufferOrigin,
      &hostOrigin,      &region,         &bufferRowPitch, &bufferSlicePitch,
      &hostRowPitch,    &hostSlicePitch, &pDst,           &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferReadRect"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferReadRect"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferReadRect"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_write_rect_params_t params = {
      &hQueue,          &hBuffer,        &blockingWrite,  &bufferOrigin,
      &hostOrigin,      &region,         &bufferRowPitch, &bufferSlicePitch,
      &hostRowPitch,    &hostSlicePitch, &pSrc,           &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferWriteRect"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferWriteRect"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferWriteRect"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_copy_params_t params = {
      &hQueue, &hBufferSrc,          &hBufferDst,      &srcOffset, &dstOffset,
      &size,   &numEventsInWaitList, &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferCopy"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferCopy"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferCopy"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_copy_rect_params_t params = {
      &hQueue,      &hBufferSrc,    &hBufferDst,          &srcOrigin,
      &dstOrigin,   &region,        &srcRowPitch,         &srcSlicePitch,
      &dstRowPitch, &dstSlicePitch, &numEventsInWaitList, &phEventWaitList,
      &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferCopyRect"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferCopyRect"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferCopyRect"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_fill_params_t params = {&hQueue,
                                                &hBuffer,
                                                &pPattern,
                                                &patternSize,
                                                &offset,
                                                &size,
                                                &numEventsInWaitList,
                                                &phEventWaitList,
                                                &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferFill"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferFill"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferFill"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_image_read_params_t params = {
      &hQueue,          &hImage, &blockingRead,
      &origin,          &region, &rowPitch,
      &slicePitch,      &pDst,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemImageRead"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemImageRead"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemImageRead"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_image_write_params_t params = {
      &hQueue,          &hImage, &blockingWrite,
      &origin,          &region, &rowPitch,
      &slicePitch,      &pSrc,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemImageWrite"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemImageWrite"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemImageWrite"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_image_copy_params_t params = {
      &hQueue, &hImageSrc,           &hImageDst,       &srcOrigin, &dstOrigin,
      &region, &numEventsInWaitList, &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemImageCopy"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemImageCopy"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemImageCopy"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppRetMap) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_buffer_map_params_t params = {
      &hQueue,  &hBuffer, &blockingMap,         &mapFlags,
      &offset,  &size,    &numEventsInWaitList, &phEventWaitList,
      &phEvent, &ppRetMap};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemBufferMap"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemBufferMap"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }

    auto parentDummyHandle = reinterpret_cast<mock::dummy_handle_t>(hBuffer);
    *ppRetMap = (void *)(parentDummyHandle->MData);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemBufferMap"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_mem_unmap_params_t params = {
      &hQueue,          &hMem,   &pMappedPtr, &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueMemUnmap"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueMemUnmap"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueMemUnmap"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_fill_params_t params = {
      &hQueue,          &pMem,   &patternSize,
      &pPattern,        &size,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMFill"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMFill"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMFill"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_memcpy_params_t params = {
      &hQueue,          &blocking, &pDst, &pSrc, &size, &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMMemcpy"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMMemcpy"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMMemcpy"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_prefetch_params_t params = {
      &hQueue,          &pMem,   &size, &flags, &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMPrefetch"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMPrefetch"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMPrefetch"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_advise_params_t params = {&hQueue, &pMem, &size, &advice,
                                           &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMAdvise"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMAdvise"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMAdvise"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_fill_2d_params_t params = {
      &hQueue,          &pMem,   &pitch,  &patternSize,
      &pPattern,        &width,  &height, &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMFill2D"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMFill2D"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMFill2D"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_memcpy_2d_params_t params = {
      &hQueue,          &blocking, &pDst,
      &dstPitch,        &pSrc,     &srcPitch,
      &width,           &height,   &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMMemcpy2D"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMMemcpy2D"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMMemcpy2D"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_device_global_variable_write_params_t params = {
      &hQueue,          &hProgram, &name, &blockingWrite,
      &count,           &offset,   &pSrc, &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEnqueueDeviceGlobalVariableWrite"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEnqueueDeviceGlobalVariableWrite"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urEnqueueDeviceGlobalVariableWrite"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_device_global_variable_read_params_t params = {
      &hQueue,          &hProgram, &name, &blockingRead,
      &count,           &offset,   &pDst, &numEventsInWaitList,
      &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEnqueueDeviceGlobalVariableRead"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEnqueueDeviceGlobalVariableRead"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urEnqueueDeviceGlobalVariableRead"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_read_host_pipe_params_t params = {
      &hQueue, &hProgram, &pipe_symbol,         &blocking,
      &pDst,   &size,     &numEventsInWaitList, &phEventWaitList,
      &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueReadHostPipe"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueReadHostPipe"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueReadHostPipe"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_write_host_pipe_params_t params = {
      &hQueue, &hProgram, &pipe_symbol,         &blocking,
      &pSrc,   &size,     &numEventsInWaitList, &phEventWaitList,
      &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueWriteHostPipe"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueWriteHostPipe"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueWriteHostPipe"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_device_alloc_exp_params_t params = {
      &hQueue,          &pPool, &size,   &pProperties, &numEventsInWaitList,
      &phEventWaitList, &ppMem, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMDeviceAllocExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMDeviceAllocExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMDeviceAllocExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_shared_alloc_exp_params_t params = {
      &hQueue,          &pPool, &size,   &pProperties, &numEventsInWaitList,
      &phEventWaitList, &ppMem, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMSharedAllocExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMSharedAllocExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMSharedAllocExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_host_alloc_exp_params_t params = {
      &hQueue,          &pPool, &size,   &pProperties, &numEventsInWaitList,
      &phEventWaitList, &ppMem, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMHostAllocExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMHostAllocExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMHostAllocExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_usm_free_exp_params_t params = {
      &hQueue, &pPool, &pMem, &numEventsInWaitList, &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueUSMFreeExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueUSMFreeExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueUSMFreeExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_usm_pool_handle_t *pPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_create_exp_params_t params = {&hContext, &hDevice, &pPoolDesc,
                                            &pPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolCreateExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolCreateExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *pPool = mock::createDummyHandle<ur_usm_pool_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolCreateExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolDestroyExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolDestroyExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to USM memory pool to be destroyed
    ur_usm_pool_handle_t hPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_destroy_exp_params_t params = {&hContext, &hDevice, &hPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolDestroyExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolDestroyExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolDestroyExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t newThreshold) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_set_threshold_exp_params_t params = {&hContext, &hDevice, &hPool,
                                                   &newThreshold};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolSetThresholdExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolSetThresholdExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolSetThresholdExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolGetDefaultDevicePoolExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *pPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_get_default_device_pool_exp_params_t params = {&hContext,
                                                             &hDevice, &pPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urUSMPoolGetDefaultDevicePoolExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urUSMPoolGetDefaultDevicePoolExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *pPool = mock::createDummyHandle<ur_usm_pool_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urUSMPoolGetDefaultDevicePoolExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_get_info_exp_params_t params = {&hPool, &propName, &pPropValue,
                                              &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolGetInfoExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolGetInfoExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolGetInfoExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolSetDevicePoolExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] handle to USM memory pool to set for a device
    ur_usm_pool_handle_t hPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_set_device_pool_exp_params_t params = {&hContext, &hDevice,
                                                     &hPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolSetDevicePoolExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolSetDevicePoolExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolSetDevicePoolExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolGetDevicePoolExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *pPool) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_get_device_pool_exp_params_t params = {&hContext, &hDevice,
                                                     &pPool};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolGetDevicePoolExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolGetDevicePoolExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *pPool = mock::createDummyHandle<ur_usm_pool_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolGetDevicePoolExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t minBytesToKeep) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pool_trim_to_exp_params_t params = {&hContext, &hDevice, &hPool,
                                             &minBytesToKeep};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPoolTrimToExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPoolTrimToExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPoolTrimToExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pResultPitch) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_pitched_alloc_exp_params_t params = {
      &hContext, &hDevice,          &pUSMDesc, &pool,        &widthInBytes,
      &height,   &elementSizeBytes, &ppMem,    &pResultPitch};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMPitchedAllocExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMPitchedAllocExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *ppMem = mock::createDummyHandle<void *>(widthInBytes * height);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMPitchedAllocExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_image_native_handle_t hImage) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_unsampled_image_handle_destroy_exp_params_t params = {
      &hContext, &hDevice, &hImage};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesUnsampledImageHandleDestroyExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesUnsampledImageHandleDestroyExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hImage);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesUnsampledImageHandleDestroyExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_image_native_handle_t hImage) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_sampled_image_handle_destroy_exp_params_t params = {
      &hContext, &hDevice, &hImage};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesSampledImageHandleDestroyExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesSampledImageHandleDestroyExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hImage);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesSampledImageHandleDestroyExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_image_mem_native_handle_t *phImageMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_image_allocate_exp_params_t params = {
      &hContext, &hDevice, &pImageFormat, &pImageDesc, &phImageMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesImageAllocateExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesImageAllocateExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phImageMem = mock::createDummyHandle<ur_exp_image_mem_native_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesImageAllocateExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesImageFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of image memory to be freed
    ur_exp_image_mem_native_handle_t hImageMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_image_free_exp_params_t params = {&hContext, &hDevice,
                                                       &hImageMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urBindlessImagesImageFreeExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesImageFreeExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hImageMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urBindlessImagesImageFreeExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_image_native_handle_t *phImage) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_unsampled_image_create_exp_params_t params = {
      &hContext, &hDevice, &hImageMem, &pImageFormat, &pImageDesc, &phImage};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesUnsampledImageCreateExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesUnsampledImageCreateExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phImage = mock::createDummyHandle<ur_exp_image_native_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesUnsampledImageCreateExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_image_native_handle_t *phImage) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_sampled_image_create_exp_params_t params = {
      &hContext,   &hDevice,  &hImageMem, &pImageFormat,
      &pImageDesc, &hSampler, &phImage};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesSampledImageCreateExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesSampledImageCreateExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phImage = mock::createDummyHandle<ur_exp_image_native_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesSampledImageCreateExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urBindlessImagesImageCopyExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesImageCopyExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urBindlessImagesImageCopyExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_image_get_info_exp_params_t params = {
      &hContext, &hImageMem, &propName, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesImageGetInfoExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesImageGetInfoExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesImageGetInfoExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_image_mem_native_handle_t *phImageMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_mipmap_get_level_exp_params_t params = {
      &hContext, &hDevice, &hImageMem, &mipmapLevel, &phImageMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesMipmapGetLevelExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesMipmapGetLevelExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phImageMem = mock::createDummyHandle<ur_exp_image_mem_native_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesMipmapGetLevelExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesMipmapFreeExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of image memory to be freed
    ur_exp_image_mem_native_handle_t hMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_mipmap_free_exp_params_t params = {&hContext, &hDevice,
                                                        &hMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesMipmapFreeExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesMipmapFreeExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urBindlessImagesMipmapFreeExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_external_mem_handle_t *phExternalMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_import_external_memory_exp_params_t params = {
      &hContext,      &hDevice,          &size,
      &memHandleType, &pExternalMemDesc, &phExternalMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesImportExternalMemoryExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesImportExternalMemoryExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phExternalMem = mock::createDummyHandle<ur_exp_external_mem_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesImportExternalMemoryExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_image_mem_native_handle_t *phImageMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_map_external_array_exp_params_t params = {
      &hContext,   &hDevice,      &pImageFormat,
      &pImageDesc, &hExternalMem, &phImageMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesMapExternalArrayExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesMapExternalArrayExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phImageMem = mock::createDummyHandle<ur_exp_image_mem_native_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesMapExternalArrayExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    void **ppRetMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_map_external_linear_memory_exp_params_t params = {
      &hContext, &hDevice, &offset, &size, &hExternalMem, &ppRetMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesMapExternalLinearMemoryExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesMapExternalLinearMemoryExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesMapExternalLinearMemoryExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesReleaseExternalMemoryExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesReleaseExternalMemoryExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of external memory to be destroyed
    ur_exp_external_mem_handle_t hExternalMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_release_external_memory_exp_params_t params = {
      &hContext, &hDevice, &hExternalMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesReleaseExternalMemoryExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesReleaseExternalMemoryExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hExternalMem);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesReleaseExternalMemoryExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_external_semaphore_handle_t *phExternalSemaphore) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_import_external_semaphore_exp_params_t params = {
      &hContext, &hDevice, &semHandleType, &pExternalSemaphoreDesc,
      &phExternalSemaphore};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesImportExternalSemaphoreExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesImportExternalSemaphoreExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phExternalSemaphore =
        mock::createDummyHandle<ur_exp_external_semaphore_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesImportExternalSemaphoreExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesReleaseExternalSemaphoreExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesReleaseExternalSemaphoreExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] handle of external semaphore to be destroyed
    ur_exp_external_semaphore_handle_t hExternalSemaphore) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_release_external_semaphore_exp_params_t params = {
      &hContext, &hDevice, &hExternalSemaphore};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesReleaseExternalSemaphoreExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesReleaseExternalSemaphoreExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hExternalSemaphore);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesReleaseExternalSemaphoreExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_wait_external_semaphore_exp_params_t params = {
      &hQueue,    &hSemaphore,          &hasWaitValue,
      &waitValue, &numEventsInWaitList, &phEventWaitList,
      &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesWaitExternalSemaphoreExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesWaitExternalSemaphoreExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesWaitExternalSemaphoreExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_bindless_images_signal_external_semaphore_exp_params_t params = {
      &hQueue,      &hSemaphore,          &hasSignalValue,
      &signalValue, &numEventsInWaitList, &phEventWaitList,
      &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urBindlessImagesSignalExternalSemaphoreExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urBindlessImagesSignalExternalSemaphoreExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urBindlessImagesSignalExternalSemaphoreExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_handle_t *phCommandBuffer) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_create_exp_params_t params = {
      &hContext, &hDevice, &pCommandBufferDesc, &phCommandBuffer};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urCommandBufferCreateExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urCommandBufferCreateExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phCommandBuffer =
        mock::createDummyHandle<ur_exp_command_buffer_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urCommandBufferCreateExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainExp(
    /// [in][retain] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_retain_exp_params_t params = {&hCommandBuffer};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urCommandBufferRetainExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urCommandBufferRetainExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::retainDummyHandle(hCommandBuffer);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urCommandBufferRetainExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseExp(
    /// [in][release] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_release_exp_params_t params = {&hCommandBuffer};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urCommandBufferReleaseExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urCommandBufferReleaseExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    mock::releaseDummyHandle(hCommandBuffer);
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urCommandBufferReleaseExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferFinalizeExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_finalize_exp_params_t params = {&hCommandBuffer};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urCommandBufferFinalizeExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urCommandBufferFinalizeExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urCommandBufferFinalizeExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendKernelLaunchExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendKernelLaunchExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendKernelLaunchExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendUSMMemcpyExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendUSMMemcpyExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendUSMMemcpyExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendUSMFillExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendUSMFillExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendUSMFillExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendMemBufferCopyExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendMemBufferCopyExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendMemBufferCopyExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendMemBufferWriteExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendMemBufferWriteExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendMemBufferWriteExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendMemBufferReadExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendMemBufferReadExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendMemBufferReadExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendMemBufferCopyRectExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendMemBufferCopyRectExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendMemBufferCopyRectExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendMemBufferWriteRectExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendMemBufferWriteRectExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendMemBufferWriteRectExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendMemBufferReadRectExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendMemBufferReadRectExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendMemBufferReadRectExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendMemBufferFillExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendMemBufferFillExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendMemBufferFillExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendUSMPrefetchExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendUSMPrefetchExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendUSMPrefetchExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferAppendUSMAdviseExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferAppendUSMAdviseExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    // optional output handle
    if (phCommand) {
      *phCommand =
          mock::createDummyHandle<ur_exp_command_buffer_command_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferAppendUSMAdviseExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_enqueue_exp_params_t params = {&hCommandBuffer, &hQueue,
                                                   &numEventsInWaitList,
                                                   &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urCommandBufferEnqueueExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urCommandBufferEnqueueExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urCommandBufferEnqueueExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
        *pUpdateKernelLaunch) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_update_kernel_launch_exp_params_t params = {
      &hCommandBuffer, &numKernelUpdates, &pUpdateKernelLaunch};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferUpdateKernelLaunchExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferUpdateKernelLaunchExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferUpdateKernelLaunchExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferUpdateSignalEventExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferUpdateSignalEventExp(
    /// [in] Handle of the command-buffer command to update.
    ur_exp_command_buffer_command_handle_t hCommand,
    /// [out][alloc] Event to be signaled.
    ur_event_handle_t *phSignalEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_update_signal_event_exp_params_t params = {&hCommand,
                                                               &phSignalEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferUpdateSignalEventExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferUpdateSignalEventExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phSignalEvent = mock::createDummyHandle<ur_event_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferUpdateSignalEventExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const ur_event_handle_t *phEventWaitList) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_update_wait_events_exp_params_t params = {
      &hCommand, &numEventsInWaitList, &phEventWaitList};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urCommandBufferUpdateWaitEventsExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urCommandBufferUpdateWaitEventsExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urCommandBufferUpdateWaitEventsExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_command_buffer_get_info_exp_params_t params = {
      &hCommandBuffer, &propName, &propSize, &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urCommandBufferGetInfoExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urCommandBufferGetInfoExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urCommandBufferGetInfoExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEnqueueCooperativeKernelLaunchExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEnqueueCooperativeKernelLaunchExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urEnqueueCooperativeKernelLaunchExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    uint32_t *pGroupCountRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_kernel_suggest_max_cooperative_group_count_exp_params_t params = {
      &hKernel,
      &hDevice,
      &workDim,
      &pLocalWorkSize,
      &dynamicSharedMemorySize,
      &pGroupCountRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urKernelSuggestMaxCooperativeGroupCountExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urKernelSuggestMaxCooperativeGroupCountExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urKernelSuggestMaxCooperativeGroupCountExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_timestamp_recording_exp_params_t params = {
      &hQueue, &blocking, &numEventsInWaitList, &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEnqueueTimestampRecordingExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEnqueueTimestampRecordingExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urEnqueueTimestampRecordingExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

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

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEnqueueKernelLaunchCustomExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEnqueueKernelLaunchCustomExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urEnqueueKernelLaunchCustomExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const char *pOptions) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_build_exp_params_t params = {&hProgram, &numDevices, &phDevices,
                                          &pOptions};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramBuildExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramBuildExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramBuildExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    const char *pOptions) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_program_compile_exp_params_t params = {&hProgram, &numDevices, &phDevices,
                                            &pOptions};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramCompileExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramCompileExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramCompileExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_program_handle_t *phProgram) try {
  ur_result_t result = UR_RESULT_SUCCESS;
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }

  ur_program_link_exp_params_t params = {&hContext, &numDevices, &phDevices,
                                         &count,    &phPrograms, &pOptions,
                                         &phProgram};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urProgramLinkExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urProgramLinkExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    *phProgram = mock::createDummyHandle<ur_program_handle_t>();
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urProgramLinkExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMImportExp
__urdlllocal ur_result_t UR_APICALL urUSMImportExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to host memory object
    void *pMem,
    /// [in] size in bytes of the host memory object to be imported
    size_t size) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_import_exp_params_t params = {&hContext, &pMem, &size};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMImportExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMImportExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMImportExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMReleaseExp
__urdlllocal ur_result_t UR_APICALL urUSMReleaseExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to host memory object
    void *pMem) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_release_exp_params_t params = {&hContext, &pMem};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUSMReleaseExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUSMReleaseExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUSMReleaseExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PEnablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    /// [in] handle of the command device object
    ur_device_handle_t commandDevice,
    /// [in] handle of the peer device object
    ur_device_handle_t peerDevice) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_p2p_enable_peer_access_exp_params_t params = {&commandDevice,
                                                       &peerDevice};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUsmP2PEnablePeerAccessExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urUsmP2PEnablePeerAccessExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUsmP2PEnablePeerAccessExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PDisablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    /// [in] handle of the command device object
    ur_device_handle_t commandDevice,
    /// [in] handle of the peer device object
    ur_device_handle_t peerDevice) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_p2p_disable_peer_access_exp_params_t params = {&commandDevice,
                                                        &peerDevice};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUsmP2PDisablePeerAccessExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urUsmP2PDisablePeerAccessExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUsmP2PDisablePeerAccessExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    size_t *pPropSizeRet) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_usm_p2p_peer_access_get_info_exp_params_t params = {
      &commandDevice, &peerDevice, &propName,
      &propSize,      &pPropValue, &pPropSizeRet};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urUsmP2PPeerAccessGetInfoExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urUsmP2PPeerAccessGetInfoExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urUsmP2PPeerAccessGetInfoExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_events_wait_with_barrier_ext_params_t params = {
      &hQueue, &pProperties, &numEventsInWaitList, &phEventWaitList, &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback(
          "urEnqueueEventsWaitWithBarrierExt"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback(
          "urEnqueueEventsWaitWithBarrierExt"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback(
          "urEnqueueEventsWaitWithBarrierExt"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
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
    ur_event_handle_t *phEvent) try {
  ur_result_t result = UR_RESULT_SUCCESS;

  ur_enqueue_native_command_exp_params_t params = {&hQueue,
                                                   &pfnNativeEnqueue,
                                                   &data,
                                                   &numMemsInMemList,
                                                   &phMemList,
                                                   &pProperties,
                                                   &numEventsInWaitList,
                                                   &phEventWaitList,
                                                   &phEvent};

  auto beforeCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_before_callback("urEnqueueNativeCommandExp"));
  if (beforeCallback) {
    result = beforeCallback(&params);
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
  }

  auto replaceCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_replace_callback("urEnqueueNativeCommandExp"));
  if (replaceCallback) {
    result = replaceCallback(&params);
  } else {

    // optional output handle
    if (phEvent) {
      *phEvent = mock::createDummyHandle<ur_event_handle_t>();
    }
    result = UR_RESULT_SUCCESS;
  }

  if (result != UR_RESULT_SUCCESS) {
    return result;
  }

  auto afterCallback = reinterpret_cast<ur_mock_callback_t>(
      mock::getCallbacks().get_after_callback("urEnqueueNativeCommandExp"));
  if (afterCallback) {
    return afterCallback(&params);
  }

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

} // namespace driver

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_global_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnAdapterGet = driver::urAdapterGet;

  pDdiTable->pfnAdapterRelease = driver::urAdapterRelease;

  pDdiTable->pfnAdapterRetain = driver::urAdapterRetain;

  pDdiTable->pfnAdapterGetLastError = driver::urAdapterGetLastError;

  pDdiTable->pfnAdapterGetInfo = driver::urAdapterGetInfo;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's BindlessImagesExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_bindless_images_exp_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnUnsampledImageHandleDestroyExp =
      driver::urBindlessImagesUnsampledImageHandleDestroyExp;

  pDdiTable->pfnSampledImageHandleDestroyExp =
      driver::urBindlessImagesSampledImageHandleDestroyExp;

  pDdiTable->pfnImageAllocateExp = driver::urBindlessImagesImageAllocateExp;

  pDdiTable->pfnImageFreeExp = driver::urBindlessImagesImageFreeExp;

  pDdiTable->pfnUnsampledImageCreateExp =
      driver::urBindlessImagesUnsampledImageCreateExp;

  pDdiTable->pfnSampledImageCreateExp =
      driver::urBindlessImagesSampledImageCreateExp;

  pDdiTable->pfnImageCopyExp = driver::urBindlessImagesImageCopyExp;

  pDdiTable->pfnImageGetInfoExp = driver::urBindlessImagesImageGetInfoExp;

  pDdiTable->pfnMipmapGetLevelExp = driver::urBindlessImagesMipmapGetLevelExp;

  pDdiTable->pfnMipmapFreeExp = driver::urBindlessImagesMipmapFreeExp;

  pDdiTable->pfnImportExternalMemoryExp =
      driver::urBindlessImagesImportExternalMemoryExp;

  pDdiTable->pfnMapExternalArrayExp =
      driver::urBindlessImagesMapExternalArrayExp;

  pDdiTable->pfnMapExternalLinearMemoryExp =
      driver::urBindlessImagesMapExternalLinearMemoryExp;

  pDdiTable->pfnReleaseExternalMemoryExp =
      driver::urBindlessImagesReleaseExternalMemoryExp;

  pDdiTable->pfnImportExternalSemaphoreExp =
      driver::urBindlessImagesImportExternalSemaphoreExp;

  pDdiTable->pfnReleaseExternalSemaphoreExp =
      driver::urBindlessImagesReleaseExternalSemaphoreExp;

  pDdiTable->pfnWaitExternalSemaphoreExp =
      driver::urBindlessImagesWaitExternalSemaphoreExp;

  pDdiTable->pfnSignalExternalSemaphoreExp =
      driver::urBindlessImagesSignalExternalSemaphoreExp;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's CommandBufferExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_command_buffer_exp_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnCreateExp = driver::urCommandBufferCreateExp;

  pDdiTable->pfnRetainExp = driver::urCommandBufferRetainExp;

  pDdiTable->pfnReleaseExp = driver::urCommandBufferReleaseExp;

  pDdiTable->pfnFinalizeExp = driver::urCommandBufferFinalizeExp;

  pDdiTable->pfnAppendKernelLaunchExp =
      driver::urCommandBufferAppendKernelLaunchExp;

  pDdiTable->pfnAppendUSMMemcpyExp = driver::urCommandBufferAppendUSMMemcpyExp;

  pDdiTable->pfnAppendUSMFillExp = driver::urCommandBufferAppendUSMFillExp;

  pDdiTable->pfnAppendMemBufferCopyExp =
      driver::urCommandBufferAppendMemBufferCopyExp;

  pDdiTable->pfnAppendMemBufferWriteExp =
      driver::urCommandBufferAppendMemBufferWriteExp;

  pDdiTable->pfnAppendMemBufferReadExp =
      driver::urCommandBufferAppendMemBufferReadExp;

  pDdiTable->pfnAppendMemBufferCopyRectExp =
      driver::urCommandBufferAppendMemBufferCopyRectExp;

  pDdiTable->pfnAppendMemBufferWriteRectExp =
      driver::urCommandBufferAppendMemBufferWriteRectExp;

  pDdiTable->pfnAppendMemBufferReadRectExp =
      driver::urCommandBufferAppendMemBufferReadRectExp;

  pDdiTable->pfnAppendMemBufferFillExp =
      driver::urCommandBufferAppendMemBufferFillExp;

  pDdiTable->pfnAppendUSMPrefetchExp =
      driver::urCommandBufferAppendUSMPrefetchExp;

  pDdiTable->pfnAppendUSMAdviseExp = driver::urCommandBufferAppendUSMAdviseExp;

  pDdiTable->pfnEnqueueExp = driver::urCommandBufferEnqueueExp;

  pDdiTable->pfnUpdateKernelLaunchExp =
      driver::urCommandBufferUpdateKernelLaunchExp;

  pDdiTable->pfnUpdateSignalEventExp =
      driver::urCommandBufferUpdateSignalEventExp;

  pDdiTable->pfnUpdateWaitEventsExp =
      driver::urCommandBufferUpdateWaitEventsExp;

  pDdiTable->pfnGetInfoExp = driver::urCommandBufferGetInfoExp;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_context_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnCreate = driver::urContextCreate;

  pDdiTable->pfnRetain = driver::urContextRetain;

  pDdiTable->pfnRelease = driver::urContextRelease;

  pDdiTable->pfnGetInfo = driver::urContextGetInfo;

  pDdiTable->pfnGetNativeHandle = driver::urContextGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle =
      driver::urContextCreateWithNativeHandle;

  pDdiTable->pfnSetExtendedDeleter = driver::urContextSetExtendedDeleter;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnKernelLaunch = driver::urEnqueueKernelLaunch;

  pDdiTable->pfnEventsWait = driver::urEnqueueEventsWait;

  pDdiTable->pfnEventsWaitWithBarrier = driver::urEnqueueEventsWaitWithBarrier;

  pDdiTable->pfnMemBufferRead = driver::urEnqueueMemBufferRead;

  pDdiTable->pfnMemBufferWrite = driver::urEnqueueMemBufferWrite;

  pDdiTable->pfnMemBufferReadRect = driver::urEnqueueMemBufferReadRect;

  pDdiTable->pfnMemBufferWriteRect = driver::urEnqueueMemBufferWriteRect;

  pDdiTable->pfnMemBufferCopy = driver::urEnqueueMemBufferCopy;

  pDdiTable->pfnMemBufferCopyRect = driver::urEnqueueMemBufferCopyRect;

  pDdiTable->pfnMemBufferFill = driver::urEnqueueMemBufferFill;

  pDdiTable->pfnMemImageRead = driver::urEnqueueMemImageRead;

  pDdiTable->pfnMemImageWrite = driver::urEnqueueMemImageWrite;

  pDdiTable->pfnMemImageCopy = driver::urEnqueueMemImageCopy;

  pDdiTable->pfnMemBufferMap = driver::urEnqueueMemBufferMap;

  pDdiTable->pfnMemUnmap = driver::urEnqueueMemUnmap;

  pDdiTable->pfnUSMFill = driver::urEnqueueUSMFill;

  pDdiTable->pfnUSMMemcpy = driver::urEnqueueUSMMemcpy;

  pDdiTable->pfnUSMPrefetch = driver::urEnqueueUSMPrefetch;

  pDdiTable->pfnUSMAdvise = driver::urEnqueueUSMAdvise;

  pDdiTable->pfnUSMFill2D = driver::urEnqueueUSMFill2D;

  pDdiTable->pfnUSMMemcpy2D = driver::urEnqueueUSMMemcpy2D;

  pDdiTable->pfnDeviceGlobalVariableWrite =
      driver::urEnqueueDeviceGlobalVariableWrite;

  pDdiTable->pfnDeviceGlobalVariableRead =
      driver::urEnqueueDeviceGlobalVariableRead;

  pDdiTable->pfnReadHostPipe = driver::urEnqueueReadHostPipe;

  pDdiTable->pfnWriteHostPipe = driver::urEnqueueWriteHostPipe;

  pDdiTable->pfnEventsWaitWithBarrierExt =
      driver::urEnqueueEventsWaitWithBarrierExt;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's EnqueueExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_exp_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnKernelLaunchCustomExp = driver::urEnqueueKernelLaunchCustomExp;

  pDdiTable->pfnUSMDeviceAllocExp = driver::urEnqueueUSMDeviceAllocExp;

  pDdiTable->pfnUSMSharedAllocExp = driver::urEnqueueUSMSharedAllocExp;

  pDdiTable->pfnUSMHostAllocExp = driver::urEnqueueUSMHostAllocExp;

  pDdiTable->pfnUSMFreeExp = driver::urEnqueueUSMFreeExp;

  pDdiTable->pfnCooperativeKernelLaunchExp =
      driver::urEnqueueCooperativeKernelLaunchExp;

  pDdiTable->pfnTimestampRecordingExp = driver::urEnqueueTimestampRecordingExp;

  pDdiTable->pfnNativeCommandExp = driver::urEnqueueNativeCommandExp;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Event table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_event_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnGetInfo = driver::urEventGetInfo;

  pDdiTable->pfnGetProfilingInfo = driver::urEventGetProfilingInfo;

  pDdiTable->pfnWait = driver::urEventWait;

  pDdiTable->pfnRetain = driver::urEventRetain;

  pDdiTable->pfnRelease = driver::urEventRelease;

  pDdiTable->pfnGetNativeHandle = driver::urEventGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle = driver::urEventCreateWithNativeHandle;

  pDdiTable->pfnSetCallback = driver::urEventSetCallback;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_kernel_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnCreate = driver::urKernelCreate;

  pDdiTable->pfnGetInfo = driver::urKernelGetInfo;

  pDdiTable->pfnGetGroupInfo = driver::urKernelGetGroupInfo;

  pDdiTable->pfnGetSubGroupInfo = driver::urKernelGetSubGroupInfo;

  pDdiTable->pfnRetain = driver::urKernelRetain;

  pDdiTable->pfnRelease = driver::urKernelRelease;

  pDdiTable->pfnGetNativeHandle = driver::urKernelGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle = driver::urKernelCreateWithNativeHandle;

  pDdiTable->pfnGetSuggestedLocalWorkSize =
      driver::urKernelGetSuggestedLocalWorkSize;

  pDdiTable->pfnSetArgValue = driver::urKernelSetArgValue;

  pDdiTable->pfnSetArgLocal = driver::urKernelSetArgLocal;

  pDdiTable->pfnSetArgPointer = driver::urKernelSetArgPointer;

  pDdiTable->pfnSetExecInfo = driver::urKernelSetExecInfo;

  pDdiTable->pfnSetArgSampler = driver::urKernelSetArgSampler;

  pDdiTable->pfnSetArgMemObj = driver::urKernelSetArgMemObj;

  pDdiTable->pfnSetSpecializationConstants =
      driver::urKernelSetSpecializationConstants;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's KernelExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_kernel_exp_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnSuggestMaxCooperativeGroupCountExp =
      driver::urKernelSuggestMaxCooperativeGroupCountExp;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_mem_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnImageCreate = driver::urMemImageCreate;

  pDdiTable->pfnBufferCreate = driver::urMemBufferCreate;

  pDdiTable->pfnRetain = driver::urMemRetain;

  pDdiTable->pfnRelease = driver::urMemRelease;

  pDdiTable->pfnBufferPartition = driver::urMemBufferPartition;

  pDdiTable->pfnGetNativeHandle = driver::urMemGetNativeHandle;

  pDdiTable->pfnBufferCreateWithNativeHandle =
      driver::urMemBufferCreateWithNativeHandle;

  pDdiTable->pfnImageCreateWithNativeHandle =
      driver::urMemImageCreateWithNativeHandle;

  pDdiTable->pfnGetInfo = driver::urMemGetInfo;

  pDdiTable->pfnImageGetInfo = driver::urMemImageGetInfo;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's PhysicalMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_physical_mem_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnCreate = driver::urPhysicalMemCreate;

  pDdiTable->pfnRetain = driver::urPhysicalMemRetain;

  pDdiTable->pfnRelease = driver::urPhysicalMemRelease;

  pDdiTable->pfnGetInfo = driver::urPhysicalMemGetInfo;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Platform table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_platform_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnGet = driver::urPlatformGet;

  pDdiTable->pfnGetInfo = driver::urPlatformGetInfo;

  pDdiTable->pfnGetNativeHandle = driver::urPlatformGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle =
      driver::urPlatformCreateWithNativeHandle;

  pDdiTable->pfnGetApiVersion = driver::urPlatformGetApiVersion;

  pDdiTable->pfnGetBackendOption = driver::urPlatformGetBackendOption;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_program_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnCreateWithIL = driver::urProgramCreateWithIL;

  pDdiTable->pfnCreateWithBinary = driver::urProgramCreateWithBinary;

  pDdiTable->pfnBuild = driver::urProgramBuild;

  pDdiTable->pfnCompile = driver::urProgramCompile;

  pDdiTable->pfnLink = driver::urProgramLink;

  pDdiTable->pfnRetain = driver::urProgramRetain;

  pDdiTable->pfnRelease = driver::urProgramRelease;

  pDdiTable->pfnGetFunctionPointer = driver::urProgramGetFunctionPointer;

  pDdiTable->pfnGetGlobalVariablePointer =
      driver::urProgramGetGlobalVariablePointer;

  pDdiTable->pfnGetInfo = driver::urProgramGetInfo;

  pDdiTable->pfnGetBuildInfo = driver::urProgramGetBuildInfo;

  pDdiTable->pfnSetSpecializationConstants =
      driver::urProgramSetSpecializationConstants;

  pDdiTable->pfnGetNativeHandle = driver::urProgramGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle =
      driver::urProgramCreateWithNativeHandle;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_program_exp_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnBuildExp = driver::urProgramBuildExp;

  pDdiTable->pfnCompileExp = driver::urProgramCompileExp;

  pDdiTable->pfnLinkExp = driver::urProgramLinkExp;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Queue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_queue_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnGetInfo = driver::urQueueGetInfo;

  pDdiTable->pfnCreate = driver::urQueueCreate;

  pDdiTable->pfnRetain = driver::urQueueRetain;

  pDdiTable->pfnRelease = driver::urQueueRelease;

  pDdiTable->pfnGetNativeHandle = driver::urQueueGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle = driver::urQueueCreateWithNativeHandle;

  pDdiTable->pfnFinish = driver::urQueueFinish;

  pDdiTable->pfnFlush = driver::urQueueFlush;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Sampler table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_sampler_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnCreate = driver::urSamplerCreate;

  pDdiTable->pfnRetain = driver::urSamplerRetain;

  pDdiTable->pfnRelease = driver::urSamplerRelease;

  pDdiTable->pfnGetInfo = driver::urSamplerGetInfo;

  pDdiTable->pfnGetNativeHandle = driver::urSamplerGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle =
      driver::urSamplerCreateWithNativeHandle;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnHostAlloc = driver::urUSMHostAlloc;

  pDdiTable->pfnDeviceAlloc = driver::urUSMDeviceAlloc;

  pDdiTable->pfnSharedAlloc = driver::urUSMSharedAlloc;

  pDdiTable->pfnFree = driver::urUSMFree;

  pDdiTable->pfnGetMemAllocInfo = driver::urUSMGetMemAllocInfo;

  pDdiTable->pfnPoolCreate = driver::urUSMPoolCreate;

  pDdiTable->pfnPoolRetain = driver::urUSMPoolRetain;

  pDdiTable->pfnPoolRelease = driver::urUSMPoolRelease;

  pDdiTable->pfnPoolGetInfo = driver::urUSMPoolGetInfo;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USMExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_exp_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnPoolCreateExp = driver::urUSMPoolCreateExp;

  pDdiTable->pfnPoolDestroyExp = driver::urUSMPoolDestroyExp;

  pDdiTable->pfnPoolSetThresholdExp = driver::urUSMPoolSetThresholdExp;

  pDdiTable->pfnPoolGetDefaultDevicePoolExp =
      driver::urUSMPoolGetDefaultDevicePoolExp;

  pDdiTable->pfnPoolGetInfoExp = driver::urUSMPoolGetInfoExp;

  pDdiTable->pfnPoolSetDevicePoolExp = driver::urUSMPoolSetDevicePoolExp;

  pDdiTable->pfnPoolGetDevicePoolExp = driver::urUSMPoolGetDevicePoolExp;

  pDdiTable->pfnPoolTrimToExp = driver::urUSMPoolTrimToExp;

  pDdiTable->pfnPitchedAllocExp = driver::urUSMPitchedAllocExp;

  pDdiTable->pfnImportExp = driver::urUSMImportExp;

  pDdiTable->pfnReleaseExp = driver::urUSMReleaseExp;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's UsmP2PExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_p2p_exp_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnEnablePeerAccessExp = driver::urUsmP2PEnablePeerAccessExp;

  pDdiTable->pfnDisablePeerAccessExp = driver::urUsmP2PDisablePeerAccessExp;

  pDdiTable->pfnPeerAccessGetInfoExp = driver::urUsmP2PPeerAccessGetInfoExp;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's VirtualMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_virtual_mem_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnGranularityGetInfo = driver::urVirtualMemGranularityGetInfo;

  pDdiTable->pfnReserve = driver::urVirtualMemReserve;

  pDdiTable->pfnFree = driver::urVirtualMemFree;

  pDdiTable->pfnMap = driver::urVirtualMemMap;

  pDdiTable->pfnUnmap = driver::urVirtualMemUnmap;

  pDdiTable->pfnSetAccess = driver::urVirtualMemSetAccess;

  pDdiTable->pfnGetInfo = driver::urVirtualMemGetInfo;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Device table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_device_dditable_t *pDdiTable) try {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (driver::d_context.version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  pDdiTable->pfnGet = driver::urDeviceGet;

  pDdiTable->pfnGetInfo = driver::urDeviceGetInfo;

  pDdiTable->pfnRetain = driver::urDeviceRetain;

  pDdiTable->pfnRelease = driver::urDeviceRelease;

  pDdiTable->pfnPartition = driver::urDevicePartition;

  pDdiTable->pfnSelectBinary = driver::urDeviceSelectBinary;

  pDdiTable->pfnGetNativeHandle = driver::urDeviceGetNativeHandle;

  pDdiTable->pfnCreateWithNativeHandle = driver::urDeviceCreateWithNativeHandle;

  pDdiTable->pfnGetGlobalTimestamps = driver::urDeviceGetGlobalTimestamps;

  return result;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

#if defined(__cplusplus)
}
#endif
