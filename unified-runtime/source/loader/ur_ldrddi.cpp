/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_ldrddi.cpp
 *
 */
#include "ur_lib_loader.hpp"
#include "ur_loader.hpp"

namespace ur_loader {

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

  auto context = getContext();

  uint32_t numAdapters = 0;
  for (auto &platform : context->platforms) {
    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;

    uint32_t adapter;
    ur_adapter_handle_t *adapterHandle =
        numAdapters < NumEntries ? &phAdapters[numAdapters] : nullptr;
    platform.dditable.Adapter.pfnGet(1, adapterHandle, &adapter);

    numAdapters += adapter;
  }

  if (pNumAdapters != nullptr) {
    *pNumAdapters = numAdapters;
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRelease
__urdlllocal ur_result_t UR_APICALL urAdapterRelease(
    /// [in][release] Adapter handle to release
    ur_adapter_handle_t hAdapter) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnRelease = dditable->Adapter.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hAdapter);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterRetain
__urdlllocal ur_result_t UR_APICALL urAdapterRetain(
    /// [in][retain] Adapter handle to retain
    ur_adapter_handle_t hAdapter) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnRetain = dditable->Adapter.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hAdapter);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnGetLastError = dditable->Adapter.pfnGetLastError;
  if (nullptr == pfnGetLastError)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetLastError(hAdapter, ppMessage, pError);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnGetInfo = dditable->Adapter.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hAdapter, propName, propSize, pPropValue, pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterSetLoggerCallback
__urdlllocal ur_result_t UR_APICALL urAdapterSetLoggerCallback(
    /// [in] handle of the adapter
    ur_adapter_handle_t hAdapter,
    /// [in] Function pointer to callback from the logger.
    ur_logger_callback_t pfnLoggerCallback,
    /// [in][out][optional] pointer to data to be passed to callback
    void *pUserData,
    /// [in] logging level
    ur_logger_level_t level) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnSetLoggerCallback = dditable->Adapter.pfnSetLoggerCallback;
  if (nullptr == pfnSetLoggerCallback)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetLoggerCallback(hAdapter, pfnLoggerCallback, pUserData, level);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterSetLoggerCallbackLevel
__urdlllocal ur_result_t UR_APICALL urAdapterSetLoggerCallbackLevel(
    /// [in] handle of the adapter
    ur_adapter_handle_t hAdapter,
    /// [in] logging level
    ur_logger_level_t level) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnSetLoggerCallbackLevel = dditable->Adapter.pfnSetLoggerCallbackLevel;
  if (nullptr == pfnSetLoggerCallbackLevel)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetLoggerCallbackLevel(hAdapter, level);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGet
__urdlllocal ur_result_t UR_APICALL urPlatformGet(
    /// [in] adapter to query for platforms.
    ur_adapter_handle_t hAdapter,
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnGet = dditable->Platform.pfnGet;
  if (nullptr == pfnGet)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGet(hAdapter, NumEntries, phPlatforms, pNumPlatforms);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPlatform);

  auto *pfnGetInfo = dditable->Platform.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hPlatform, propName, propSize, pPropValue, pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetApiVersion
__urdlllocal ur_result_t UR_APICALL urPlatformGetApiVersion(
    /// [in] handle of the platform
    ur_platform_handle_t hPlatform,
    /// [out] api version
    ur_api_version_t *pVersion) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPlatform);

  auto *pfnGetApiVersion = dditable->Platform.pfnGetApiVersion;
  if (nullptr == pfnGetApiVersion)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetApiVersion(hPlatform, pVersion);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPlatformGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urPlatformGetNativeHandle(
    /// [in] handle of the platform.
    ur_platform_handle_t hPlatform,
    /// [out] a pointer to the native handle of the platform.
    ur_native_handle_t *phNativePlatform) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPlatform);

  auto *pfnGetNativeHandle = dditable->Platform.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hPlatform, phNativePlatform);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnCreateWithNativeHandle =
      dditable->Platform.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativePlatform, hAdapter, pProperties,
                                   phPlatform);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPlatform);

  auto *pfnGetBackendOption = dditable->Platform.pfnGetBackendOption;
  if (nullptr == pfnGetBackendOption)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetBackendOption(hPlatform, pFrontendOption, ppPlatformOption);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPlatform);

  auto *pfnGet = dditable->Device.pfnGet;
  if (nullptr == pfnGet)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGet(hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnGetInfo = dditable->Device.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hDevice, propName, propSize, pPropValue, pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRetain
__urdlllocal ur_result_t UR_APICALL urDeviceRetain(
    /// [in][retain] handle of the device to get a reference of.
    ur_device_handle_t hDevice) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnRetain = dditable->Device.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hDevice);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceRelease
__urdlllocal ur_result_t UR_APICALL urDeviceRelease(
    /// [in][release] handle of the device to release.
    ur_device_handle_t hDevice) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnRelease = dditable->Device.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hDevice);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnPartition = dditable->Device.pfnPartition;
  if (nullptr == pfnPartition)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPartition(hDevice, pProperties, NumDevices, phSubDevices,
                      pNumDevicesRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnSelectBinary = dditable->Device.pfnSelectBinary;
  if (nullptr == pfnSelectBinary)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSelectBinary(hDevice, pBinaries, NumBinaries, pSelectedBinary);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urDeviceGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urDeviceGetNativeHandle(
    /// [in] handle of the device.
    ur_device_handle_t hDevice,
    /// [out] a pointer to the native handle of the device.
    ur_native_handle_t *phNativeDevice) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnGetNativeHandle = dditable->Device.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hDevice, phNativeDevice);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnCreateWithNativeHandle = dditable->Device.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativeDevice, hAdapter, pProperties,
                                   phDevice);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnGetGlobalTimestamps = dditable->Device.pfnGetGlobalTimestamps;
  if (nullptr == pfnGetGlobalTimestamps)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetGlobalTimestamps(hDevice, pDeviceTimestamp, pHostTimestamp);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(phDevices[0]);

  auto *pfnCreate = dditable->Context.pfnCreate;
  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreate(DeviceCount, phDevices, pProperties, phContext);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL urContextRetain(
    /// [in][retain] handle of the context to get a reference of.
    ur_context_handle_t hContext) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnRetain = dditable->Context.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hContext);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL urContextRelease(
    /// [in][release] handle of the context to release.
    ur_context_handle_t hContext) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnRelease = dditable->Context.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hContext);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGetInfo = dditable->Context.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hContext, propName, propSize, pPropValue, pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextGetNativeHandle(
    /// [in] handle of the context.
    ur_context_handle_t hContext,
    /// [out] a pointer to the native handle of the context.
    ur_native_handle_t *phNativeContext) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGetNativeHandle = dditable->Context.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hContext, phNativeContext);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hAdapter);

  auto *pfnCreateWithNativeHandle = dditable->Context.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativeContext, hAdapter, numDevices,
                                   phDevices, pProperties, phContext);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnSetExtendedDeleter = dditable->Context.pfnSetExtendedDeleter;
  if (nullptr == pfnSetExtendedDeleter)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetExtendedDeleter(hContext, pfnDeleter, pUserData);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImageCreate = dditable->Mem.pfnImageCreate;
  if (nullptr == pfnImageCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImageCreate(hContext, flags, pImageFormat, pImageDesc, pHost,
                        phMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnBufferCreate = dditable->Mem.pfnBufferCreate;
  if (nullptr == pfnBufferCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnBufferCreate(hContext, flags, size, pProperties, phBuffer);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL urMemRetain(
    /// [in][retain] handle of the memory object to get access
    ur_mem_handle_t hMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hMem);

  auto *pfnRetain = dditable->Mem.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL urMemRelease(
    /// [in][release] handle of the memory object to release
    ur_mem_handle_t hMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hMem);

  auto *pfnRelease = dditable->Mem.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hBuffer);

  auto *pfnBufferPartition = dditable->Mem.pfnBufferPartition;
  if (nullptr == pfnBufferPartition)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion, phMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hMem);

  auto *pfnGetNativeHandle = dditable->Mem.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hMem, hDevice, phNativeMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnBufferCreateWithNativeHandle =
      dditable->Mem.pfnBufferCreateWithNativeHandle;
  if (nullptr == pfnBufferCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnBufferCreateWithNativeHandle(hNativeMem, hContext, pProperties,
                                         phMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImageCreateWithNativeHandle =
      dditable->Mem.pfnImageCreateWithNativeHandle;
  if (nullptr == pfnImageCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImageCreateWithNativeHandle(hNativeMem, hContext, pImageFormat,
                                        pImageDesc, pProperties, phMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hMemory);

  auto *pfnGetInfo = dditable->Mem.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hMemory);

  auto *pfnImageGetInfo = dditable->Mem.pfnImageGetInfo;
  if (nullptr == pfnImageGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImageGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreate = dditable->Sampler.pfnCreate;
  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreate(hContext, pDesc, phSampler);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRetain
__urdlllocal ur_result_t UR_APICALL urSamplerRetain(
    /// [in][retain] handle of the sampler object to get access
    ur_sampler_handle_t hSampler) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hSampler);

  auto *pfnRetain = dditable->Sampler.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hSampler);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerRelease
__urdlllocal ur_result_t UR_APICALL urSamplerRelease(
    /// [in][release] handle of the sampler object to release
    ur_sampler_handle_t hSampler) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hSampler);

  auto *pfnRelease = dditable->Sampler.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hSampler);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hSampler);

  auto *pfnGetInfo = dditable->Sampler.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hSampler, propName, propSize, pPropValue, pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urSamplerGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urSamplerGetNativeHandle(
    /// [in] handle of the sampler.
    ur_sampler_handle_t hSampler,
    /// [out] a pointer to the native handle of the sampler.
    ur_native_handle_t *phNativeSampler) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hSampler);

  auto *pfnGetNativeHandle = dditable->Sampler.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hSampler, phNativeSampler);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateWithNativeHandle = dditable->Sampler.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativeSampler, hContext, pProperties,
                                   phSampler);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnHostAlloc = dditable->USM.pfnHostAlloc;
  if (nullptr == pfnHostAlloc)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnHostAlloc(hContext, pUSMDesc, pool, size, ppMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnDeviceAlloc = dditable->USM.pfnDeviceAlloc;
  if (nullptr == pfnDeviceAlloc)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnDeviceAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnSharedAlloc = dditable->USM.pfnSharedAlloc;
  if (nullptr == pfnSharedAlloc)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSharedAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to USM memory object
    void *pMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnFree = dditable->USM.pfnFree;
  if (nullptr == pfnFree)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnFree(hContext, pMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGetMemAllocInfo = dditable->USM.pfnGetMemAllocInfo;
  if (nullptr == pfnGetMemAllocInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetMemAllocInfo(hContext, pMem, propName, propSize, pPropValue,
                            pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPoolCreate = dditable->USM.pfnPoolCreate;
  if (nullptr == pfnPoolCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolCreate(hContext, pPoolDesc, ppPool);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRetain
__urdlllocal ur_result_t UR_APICALL urUSMPoolRetain(
    /// [in][retain] pointer to USM memory pool
    ur_usm_pool_handle_t pPool) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(pPool);

  auto *pfnPoolRetain = dditable->USM.pfnPoolRetain;
  if (nullptr == pfnPoolRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolRetain(pPool);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolRelease
__urdlllocal ur_result_t UR_APICALL urUSMPoolRelease(
    /// [in][release] pointer to USM memory pool
    ur_usm_pool_handle_t pPool) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(pPool);

  auto *pfnPoolRelease = dditable->USM.pfnPoolRelease;
  if (nullptr == pfnPoolRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolRelease(pPool);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPool);

  auto *pfnPoolGetInfo = dditable->USM.pfnPoolGetInfo;
  if (nullptr == pfnPoolGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolGetInfo(hPool, propName, propSize, pPropValue, pPropSizeRet);
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
    /// [in] allocation size in bytes for which the alignment is being
    /// queried.
    size_t allocationSize,
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGranularityGetInfo = dditable->VirtualMem.pfnGranularityGetInfo;
  if (nullptr == pfnGranularityGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGranularityGetInfo(hContext, hDevice, allocationSize, propName,
                               propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnReserve = dditable->VirtualMem.pfnReserve;
  if (nullptr == pfnReserve)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnReserve(hContext, pStart, size, ppStart);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnFree = dditable->VirtualMem.pfnFree;
  if (nullptr == pfnFree)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnFree(hContext, pStart, size);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnMap = dditable->VirtualMem.pfnMap;
  if (nullptr == pfnMap)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMap(hContext, pStart, size, hPhysicalMem, offset, flags);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnUnmap = dditable->VirtualMem.pfnUnmap;
  if (nullptr == pfnUnmap)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUnmap(hContext, pStart, size);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnSetAccess = dditable->VirtualMem.pfnSetAccess;
  if (nullptr == pfnSetAccess)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetAccess(hContext, pStart, size, flags);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGetInfo = dditable->VirtualMem.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hContext, pStart, size, propName, propSize, pPropValue,
                    pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreate = dditable->PhysicalMem.pfnCreate;
  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreate(hContext, hDevice, size, pProperties, phPhysicalMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRetain
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRetain(
    /// [in][retain] handle of the physical memory object to retain.
    ur_physical_mem_handle_t hPhysicalMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPhysicalMem);

  auto *pfnRetain = dditable->PhysicalMem.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hPhysicalMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urPhysicalMemRelease
__urdlllocal ur_result_t UR_APICALL urPhysicalMemRelease(
    /// [in][release] handle of the physical memory object to release.
    ur_physical_mem_handle_t hPhysicalMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPhysicalMem);

  auto *pfnRelease = dditable->PhysicalMem.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hPhysicalMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPhysicalMem);

  auto *pfnGetInfo = dditable->PhysicalMem.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hPhysicalMem, propName, propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateWithIL = dditable->Program.pfnCreateWithIL;
  if (nullptr == pfnCreateWithIL)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithIL(hContext, pIL, length, pProperties, phProgram);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateWithBinary = dditable->Program.pfnCreateWithBinary;
  if (nullptr == pfnCreateWithBinary)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithBinary(hContext, numDevices, phDevices, pLengths,
                             ppBinaries, pProperties, phProgram);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnBuild = dditable->Program.pfnBuild;
  if (nullptr == pfnBuild)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnBuild(hContext, hProgram, pOptions);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCompile = dditable->Program.pfnCompile;
  if (nullptr == pfnCompile)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCompile(hContext, hProgram, pOptions);
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
  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnLink = dditable->Program.pfnLink;
  if (nullptr == pfnLink)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnLink(hContext, count, phPrograms, pOptions, phProgram);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL urProgramRetain(
    /// [in][retain] handle for the Program to retain
    ur_program_handle_t hProgram) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnRetain = dditable->Program.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hProgram);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
__urdlllocal ur_result_t UR_APICALL urProgramRelease(
    /// [in][release] handle for the Program to release
    ur_program_handle_t hProgram) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnRelease = dditable->Program.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hProgram);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnGetFunctionPointer = dditable->Program.pfnGetFunctionPointer;
  if (nullptr == pfnGetFunctionPointer)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetFunctionPointer(hDevice, hProgram, pFunctionName,
                               ppFunctionPointer);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnGetGlobalVariablePointer =
      dditable->Program.pfnGetGlobalVariablePointer;
  if (nullptr == pfnGetGlobalVariablePointer)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetGlobalVariablePointer(hDevice, hProgram, pGlobalVariableName,
                                     pGlobalVariableSizeRet,
                                     ppGlobalVariablePointerRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnGetInfo = dditable->Program.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hProgram, propName, propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnGetBuildInfo = dditable->Program.pfnGetBuildInfo;
  if (nullptr == pfnGetBuildInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetBuildInfo(hProgram, hDevice, propName, propSize, pPropValue,
                         pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnSetSpecializationConstants =
      dditable->Program.pfnSetSpecializationConstants;
  if (nullptr == pfnSetSpecializationConstants)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetSpecializationConstants(hProgram, count, pSpecConstants);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramGetNativeHandle(
    /// [in] handle of the program.
    ur_program_handle_t hProgram,
    /// [out] a pointer to the native handle of the program.
    ur_native_handle_t *phNativeProgram) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnGetNativeHandle = dditable->Program.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hProgram, phNativeProgram);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateWithNativeHandle = dditable->Program.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativeProgram, hContext, pProperties,
                                   phProgram);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnCreate = dditable->Kernel.pfnCreate;
  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreate(hProgram, pKernelName, phKernel);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSetArgValue = dditable->Kernel.pfnSetArgValue;
  if (nullptr == pfnSetArgValue)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetArgValue(hKernel, argIndex, argSize, pProperties, pArgValue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSetArgLocal = dditable->Kernel.pfnSetArgLocal;
  if (nullptr == pfnSetArgLocal)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetArgLocal(hKernel, argIndex, argSize, pProperties);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnGetInfo = dditable->Kernel.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hKernel, propName, propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnGetGroupInfo = dditable->Kernel.pfnGetGroupInfo;
  if (nullptr == pfnGetGroupInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetGroupInfo(hKernel, hDevice, propName, propSize, pPropValue,
                         pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnGetSubGroupInfo = dditable->Kernel.pfnGetSubGroupInfo;
  if (nullptr == pfnGetSubGroupInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetSubGroupInfo(hKernel, hDevice, propName, propSize, pPropValue,
                            pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL urKernelRetain(
    /// [in][retain] handle for the Kernel to retain
    ur_kernel_handle_t hKernel) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnRetain = dditable->Kernel.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hKernel);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t UR_APICALL urKernelRelease(
    /// [in][release] handle for the Kernel to release
    ur_kernel_handle_t hKernel) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnRelease = dditable->Kernel.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hKernel);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSetArgPointer = dditable->Kernel.pfnSetArgPointer;
  if (nullptr == pfnSetArgPointer)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetArgPointer(hKernel, argIndex, pProperties, pArgValue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSetExecInfo = dditable->Kernel.pfnSetExecInfo;
  if (nullptr == pfnSetExecInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetExecInfo(hKernel, propName, propSize, pProperties, pPropValue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSetArgSampler = dditable->Kernel.pfnSetArgSampler;
  if (nullptr == pfnSetArgSampler)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetArgSampler(hKernel, argIndex, pProperties, hArgValue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSetArgMemObj = dditable->Kernel.pfnSetArgMemObj;
  if (nullptr == pfnSetArgMemObj)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSetSpecializationConstants =
      dditable->Kernel.pfnSetSpecializationConstants;
  if (nullptr == pfnSetSpecializationConstants)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetSpecializationConstants(hKernel, count, pSpecConstants);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urKernelGetNativeHandle(
    /// [in] handle of the kernel.
    ur_kernel_handle_t hKernel,
    /// [out] a pointer to the native handle of the kernel.
    ur_native_handle_t *phNativeKernel) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnGetNativeHandle = dditable->Kernel.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hKernel, phNativeKernel);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateWithNativeHandle = dditable->Kernel.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativeKernel, hContext, hProgram,
                                   pProperties, phKernel);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnGetSuggestedLocalWorkSize =
      dditable->Kernel.pfnGetSuggestedLocalWorkSize;
  if (nullptr == pfnGetSuggestedLocalWorkSize)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetSuggestedLocalWorkSize(hKernel, hQueue, numWorkDim,
                                      pGlobalWorkOffset, pGlobalWorkSize,
                                      pSuggestedLocalWorkSize);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSuggestMaxCooperativeGroupCount
__urdlllocal ur_result_t UR_APICALL urKernelSuggestMaxCooperativeGroupCount(
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hKernel);

  auto *pfnSuggestMaxCooperativeGroupCount =
      dditable->Kernel.pfnSuggestMaxCooperativeGroupCount;
  if (nullptr == pfnSuggestMaxCooperativeGroupCount)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSuggestMaxCooperativeGroupCount(
      hKernel, hDevice, workDim, pLocalWorkSize, dynamicSharedMemorySize,
      pGroupCountRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnGetInfo = dditable->Queue.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hQueue, propName, propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreate = dditable->Queue.pfnCreate;
  if (nullptr == pfnCreate)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreate(hContext, hDevice, pProperties, phQueue);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRetain
__urdlllocal ur_result_t UR_APICALL urQueueRetain(
    /// [in][retain] handle of the queue object to get access
    ur_queue_handle_t hQueue) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnRetain = dditable->Queue.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hQueue);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueRelease
__urdlllocal ur_result_t UR_APICALL urQueueRelease(
    /// [in][release] handle of the queue object to release
    ur_queue_handle_t hQueue) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnRelease = dditable->Queue.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hQueue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnGetNativeHandle = dditable->Queue.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hQueue, pDesc, phNativeQueue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateWithNativeHandle = dditable->Queue.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativeQueue, hContext, hDevice, pProperties,
                                   phQueue);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFinish
__urdlllocal ur_result_t UR_APICALL urQueueFinish(
    /// [in] handle of the queue to be finished.
    ur_queue_handle_t hQueue) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnFinish = dditable->Queue.pfnFinish;
  if (nullptr == pfnFinish)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnFinish(hQueue);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueFlush
__urdlllocal ur_result_t UR_APICALL urQueueFlush(
    /// [in] handle of the queue to be flushed.
    ur_queue_handle_t hQueue) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnFlush = dditable->Queue.pfnFlush;
  if (nullptr == pfnFlush)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnFlush(hQueue);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hEvent);

  auto *pfnGetInfo = dditable->Event.pfnGetInfo;
  if (nullptr == pfnGetInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfo(hEvent, propName, propSize, pPropValue, pPropSizeRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hEvent);

  auto *pfnGetProfilingInfo = dditable->Event.pfnGetProfilingInfo;
  if (nullptr == pfnGetProfilingInfo)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetProfilingInfo(hEvent, propName, propSize, pPropValue,
                             pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventWait
__urdlllocal ur_result_t UR_APICALL urEventWait(
    /// [in] number of events in the event list
    uint32_t numEvents,
    /// [in][range(0, numEvents)] pointer to a list of events to wait for
    /// completion
    const ur_event_handle_t *phEventWaitList) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(phEventWaitList[0]);

  auto *pfnWait = dditable->Event.pfnWait;
  if (nullptr == pfnWait)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnWait(numEvents, phEventWaitList);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRetain
__urdlllocal ur_result_t UR_APICALL urEventRetain(
    /// [in][retain] handle of the event object
    ur_event_handle_t hEvent) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hEvent);

  auto *pfnRetain = dditable->Event.pfnRetain;
  if (nullptr == pfnRetain)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetain(hEvent);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventRelease
__urdlllocal ur_result_t UR_APICALL urEventRelease(
    /// [in][release] handle of the event object
    ur_event_handle_t hEvent) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hEvent);

  auto *pfnRelease = dditable->Event.pfnRelease;
  if (nullptr == pfnRelease)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRelease(hEvent);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEventGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urEventGetNativeHandle(
    /// [in] handle of the event.
    ur_event_handle_t hEvent,
    /// [out] a pointer to the native handle of the event.
    ur_native_handle_t *phNativeEvent) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hEvent);

  auto *pfnGetNativeHandle = dditable->Event.pfnGetNativeHandle;
  if (nullptr == pfnGetNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandle(hEvent, phNativeEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateWithNativeHandle = dditable->Event.pfnCreateWithNativeHandle;
  if (nullptr == pfnCreateWithNativeHandle)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateWithNativeHandle(hNativeEvent, hContext, pProperties,
                                   phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hEvent);

  auto *pfnSetCallback = dditable->Event.pfnSetCallback;
  if (nullptr == pfnSetCallback)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSetCallback(hEvent, execStatus, pfnNotify, pUserData);
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
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the offset used to calculate the global ID of a work-item
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
    /// [in] size of the launch prop list
    uint32_t numPropsInLaunchPropList,
    /// [in][optional][range(0, numPropsInLaunchPropList)] pointer to a list
    /// of launch properties
    const ur_kernel_launch_property_t *launchPropList,
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnKernelLaunch = dditable->Enqueue.pfnKernelLaunch;
  if (nullptr == pfnKernelLaunch)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                         pGlobalWorkSize, pLocalWorkSize,
                         numPropsInLaunchPropList, launchPropList,
                         numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnEventsWait = dditable->Enqueue.pfnEventsWait;
  if (nullptr == pfnEventsWait)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnEventsWait(hQueue, numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnEventsWaitWithBarrier = dditable->Enqueue.pfnEventsWaitWithBarrier;
  if (nullptr == pfnEventsWaitWithBarrier)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnEventsWaitWithBarrier(hQueue, numEventsInWaitList, phEventWaitList,
                                  phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferRead = dditable->Enqueue.pfnMemBufferRead;
  if (nullptr == pfnMemBufferRead)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size, pDst,
                          numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferWrite = dditable->Enqueue.pfnMemBufferWrite;
  if (nullptr == pfnMemBufferWrite)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size, pSrc,
                           numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferReadRect = dditable->Enqueue.pfnMemBufferReadRect;
  if (nullptr == pfnMemBufferReadRect)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferReadRect(
      hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
      numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferWriteRect = dditable->Enqueue.pfnMemBufferWriteRect;
  if (nullptr == pfnMemBufferWriteRect)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferWriteRect(
      hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
      bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
      numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferCopy = dditable->Enqueue.pfnMemBufferCopy;
  if (nullptr == pfnMemBufferCopy)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset,
                          size, numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferCopyRect = dditable->Enqueue.pfnMemBufferCopyRect;
  if (nullptr == pfnMemBufferCopyRect)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferCopyRect(hQueue, hBufferSrc, hBufferDst, srcOrigin,
                              dstOrigin, region, srcRowPitch, srcSlicePitch,
                              dstRowPitch, dstSlicePitch, numEventsInWaitList,
                              phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferFill = dditable->Enqueue.pfnMemBufferFill;
  if (nullptr == pfnMemBufferFill)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize, offset, size,
                          numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemImageRead = dditable->Enqueue.pfnMemImageRead;
  if (nullptr == pfnMemImageRead)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemImageRead(hQueue, hImage, blockingRead, origin, region, rowPitch,
                         slicePitch, pDst, numEventsInWaitList, phEventWaitList,
                         phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemImageWrite = dditable->Enqueue.pfnMemImageWrite;
  if (nullptr == pfnMemImageWrite)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemImageWrite(hQueue, hImage, blockingWrite, origin, region,
                          rowPitch, slicePitch, pSrc, numEventsInWaitList,
                          phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemImageCopy = dditable->Enqueue.pfnMemImageCopy;
  if (nullptr == pfnMemImageCopy)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemImageCopy(hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin,
                         region, numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemBufferMap = dditable->Enqueue.pfnMemBufferMap;
  if (nullptr == pfnMemBufferMap)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags, offset, size,
                         numEventsInWaitList, phEventWaitList, phEvent,
                         ppRetMap);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnMemUnmap = dditable->Enqueue.pfnMemUnmap;
  if (nullptr == pfnMemUnmap)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList,
                     phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMFill = dditable->Enqueue.pfnUSMFill;
  if (nullptr == pfnUSMFill)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMFill(hQueue, pMem, patternSize, pPattern, size,
                    numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMMemcpy = dditable->Enqueue.pfnUSMMemcpy;
  if (nullptr == pfnUSMMemcpy)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMMemcpy(hQueue, blocking, pDst, pSrc, size, numEventsInWaitList,
                      phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMPrefetch = dditable->Enqueue.pfnUSMPrefetch;
  if (nullptr == pfnUSMPrefetch)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMPrefetch(hQueue, pMem, size, flags, numEventsInWaitList,
                        phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMAdvise = dditable->Enqueue.pfnUSMAdvise;
  if (nullptr == pfnUSMAdvise)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMAdvise(hQueue, pMem, size, advice, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMFill2D = dditable->Enqueue.pfnUSMFill2D;
  if (nullptr == pfnUSMFill2D)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMFill2D(hQueue, pMem, pitch, patternSize, pPattern, width, height,
                      numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMMemcpy2D = dditable->Enqueue.pfnUSMMemcpy2D;
  if (nullptr == pfnUSMMemcpy2D)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMMemcpy2D(hQueue, blocking, pDst, dstPitch, pSrc, srcPitch, width,
                        height, numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnDeviceGlobalVariableWrite =
      dditable->Enqueue.pfnDeviceGlobalVariableWrite;
  if (nullptr == pfnDeviceGlobalVariableWrite)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnDeviceGlobalVariableWrite(hQueue, hProgram, name, blockingWrite,
                                      count, offset, pSrc, numEventsInWaitList,
                                      phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnDeviceGlobalVariableRead =
      dditable->Enqueue.pfnDeviceGlobalVariableRead;
  if (nullptr == pfnDeviceGlobalVariableRead)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnDeviceGlobalVariableRead(hQueue, hProgram, name, blockingRead,
                                     count, offset, pDst, numEventsInWaitList,
                                     phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnReadHostPipe = dditable->Enqueue.pfnReadHostPipe;
  if (nullptr == pfnReadHostPipe)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnReadHostPipe(hQueue, hProgram, pipe_symbol, blocking, pDst, size,
                         numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnWriteHostPipe = dditable->Enqueue.pfnWriteHostPipe;
  if (nullptr == pfnWriteHostPipe)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnWriteHostPipe(hQueue, hProgram, pipe_symbol, blocking, pSrc, size,
                          numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMDeviceAllocExp = dditable->EnqueueExp.pfnUSMDeviceAllocExp;
  if (nullptr == pfnUSMDeviceAllocExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMDeviceAllocExp(hQueue, pPool, size, pProperties,
                              numEventsInWaitList, phEventWaitList, ppMem,
                              phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMSharedAllocExp = dditable->EnqueueExp.pfnUSMSharedAllocExp;
  if (nullptr == pfnUSMSharedAllocExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMSharedAllocExp(hQueue, pPool, size, pProperties,
                              numEventsInWaitList, phEventWaitList, ppMem,
                              phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMHostAllocExp = dditable->EnqueueExp.pfnUSMHostAllocExp;
  if (nullptr == pfnUSMHostAllocExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMHostAllocExp(hQueue, pPool, size, pProperties,
                            numEventsInWaitList, phEventWaitList, ppMem,
                            phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnUSMFreeExp = dditable->EnqueueExp.pfnUSMFreeExp;
  if (nullptr == pfnUSMFreeExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUSMFreeExp(hQueue, pPool, pMem, numEventsInWaitList,
                       phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPoolCreateExp = dditable->USMExp.pfnPoolCreateExp;
  if (nullptr == pfnPoolCreateExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolCreateExp(hContext, hDevice, pPoolDesc, pPool);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPoolDestroyExp = dditable->USMExp.pfnPoolDestroyExp;
  if (nullptr == pfnPoolDestroyExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolDestroyExp(hContext, hDevice, hPool);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPoolGetDefaultDevicePoolExp =
      dditable->USMExp.pfnPoolGetDefaultDevicePoolExp;
  if (nullptr == pfnPoolGetDefaultDevicePoolExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolGetDefaultDevicePoolExp(hContext, hDevice, pPool);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPool);

  auto *pfnPoolGetInfoExp = dditable->USMExp.pfnPoolGetInfoExp;
  if (nullptr == pfnPoolGetInfoExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolGetInfoExp(hPool, propName, pPropValue, pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMPoolSetInfoExp
__urdlllocal ur_result_t UR_APICALL urUSMPoolSetInfoExp(
    /// [in] handle to USM memory pool for the property to be set
    ur_usm_pool_handle_t hPool,
    /// [in] setting property name
    ur_usm_pool_info_t propName,
    /// [in] pointer to value to assign
    void *pPropValue,
    /// [in] size of value to assign
    size_t propSize) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hPool);

  auto *pfnPoolSetInfoExp = dditable->USMExp.pfnPoolSetInfoExp;
  if (nullptr == pfnPoolSetInfoExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolSetInfoExp(hPool, propName, pPropValue, propSize);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPoolSetDevicePoolExp = dditable->USMExp.pfnPoolSetDevicePoolExp;
  if (nullptr == pfnPoolSetDevicePoolExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolSetDevicePoolExp(hContext, hDevice, hPool);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPoolGetDevicePoolExp = dditable->USMExp.pfnPoolGetDevicePoolExp;
  if (nullptr == pfnPoolGetDevicePoolExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolGetDevicePoolExp(hContext, hDevice, pPool);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPoolTrimToExp = dditable->USMExp.pfnPoolTrimToExp;
  if (nullptr == pfnPoolTrimToExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPoolTrimToExp(hContext, hDevice, hPool, minBytesToKeep);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnPitchedAllocExp = dditable->USMExp.pfnPitchedAllocExp;
  if (nullptr == pfnPitchedAllocExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPitchedAllocExp(hContext, hDevice, pUSMDesc, pool, widthInBytes,
                            height, elementSizeBytes, ppMem, pResultPitch);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnUnsampledImageHandleDestroyExp =
      dditable->BindlessImagesExp.pfnUnsampledImageHandleDestroyExp;
  if (nullptr == pfnUnsampledImageHandleDestroyExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUnsampledImageHandleDestroyExp(hContext, hDevice, hImage);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnSampledImageHandleDestroyExp =
      dditable->BindlessImagesExp.pfnSampledImageHandleDestroyExp;
  if (nullptr == pfnSampledImageHandleDestroyExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSampledImageHandleDestroyExp(hContext, hDevice, hImage);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImageAllocateExp = dditable->BindlessImagesExp.pfnImageAllocateExp;
  if (nullptr == pfnImageAllocateExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImageAllocateExp(hContext, hDevice, pImageFormat, pImageDesc,
                             phImageMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImageFreeExp = dditable->BindlessImagesExp.pfnImageFreeExp;
  if (nullptr == pfnImageFreeExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImageFreeExp(hContext, hDevice, hImageMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnUnsampledImageCreateExp =
      dditable->BindlessImagesExp.pfnUnsampledImageCreateExp;
  if (nullptr == pfnUnsampledImageCreateExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUnsampledImageCreateExp(hContext, hDevice, hImageMem, pImageFormat,
                                    pImageDesc, phImage);
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
    /// [in] pointer to sampler description to be used
    const ur_sampler_desc_t *pSamplerDesc,
    /// [out][alloc] pointer to handle of image object created
    ur_exp_image_native_handle_t *phImage) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnSampledImageCreateExp =
      dditable->BindlessImagesExp.pfnSampledImageCreateExp;
  if (nullptr == pfnSampledImageCreateExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSampledImageCreateExp(hContext, hDevice, hImageMem, pImageFormat,
                                  pImageDesc, pSamplerDesc, phImage);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnImageCopyExp = dditable->BindlessImagesExp.pfnImageCopyExp;
  if (nullptr == pfnImageCopyExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImageCopyExp(hQueue, pSrc, pDst, pSrcImageDesc, pDstImageDesc,
                         pSrcImageFormat, pDstImageFormat, pCopyRegion,
                         imageCopyFlags, numEventsInWaitList, phEventWaitList,
                         phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImageGetInfoExp = dditable->BindlessImagesExp.pfnImageGetInfoExp;
  if (nullptr == pfnImageGetInfoExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImageGetInfoExp(hContext, hImageMem, propName, pPropValue,
                            pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for
/// urBindlessImagesGetImageMemoryHandleTypeSupportExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesGetImageMemoryHandleTypeSupportExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] type of image backing memory handle to query support for
    ur_exp_image_mem_type_t imageMemHandleType,
    /// [out] returned indication of support for allocating the given image
    /// backing memory handle type
    ur_bool_t *pSupportedRet) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGetImageMemoryHandleTypeSupportExp =
      dditable->BindlessImagesExp.pfnGetImageMemoryHandleTypeSupportExp;
  if (nullptr == pfnGetImageMemoryHandleTypeSupportExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetImageMemoryHandleTypeSupportExp(hContext, hDevice, pImageDesc,
                                               pImageFormat, imageMemHandleType,
                                               pSupportedRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for
/// urBindlessImagesGetImageUnsampledHandleSupportExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesGetImageUnsampledHandleSupportExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] type of image backing memory handle to query support for
    ur_exp_image_mem_type_t imageMemHandleType,
    /// [out] returned indication of support for creating unsampled image
    /// handles
    ur_bool_t *pSupportedRet) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGetImageUnsampledHandleSupportExp =
      dditable->BindlessImagesExp.pfnGetImageUnsampledHandleSupportExp;
  if (nullptr == pfnGetImageUnsampledHandleSupportExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetImageUnsampledHandleSupportExp(hContext, hDevice, pImageDesc,
                                              pImageFormat, imageMemHandleType,
                                              pSupportedRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for
/// urBindlessImagesGetImageSampledHandleSupportExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesGetImageSampledHandleSupportExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] pointer to image description
    const ur_image_desc_t *pImageDesc,
    /// [in] pointer to image format specification
    const ur_image_format_t *pImageFormat,
    /// [in] type of image backing memory handle to query support for
    ur_exp_image_mem_type_t imageMemHandleType,
    /// [out] returned indication of support for creating sampled image
    /// handles
    ur_bool_t *pSupportedRet) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnGetImageSampledHandleSupportExp =
      dditable->BindlessImagesExp.pfnGetImageSampledHandleSupportExp;
  if (nullptr == pfnGetImageSampledHandleSupportExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetImageSampledHandleSupportExp(hContext, hDevice, pImageDesc,
                                            pImageFormat, imageMemHandleType,
                                            pSupportedRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnMipmapGetLevelExp = dditable->BindlessImagesExp.pfnMipmapGetLevelExp;
  if (nullptr == pfnMipmapGetLevelExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMipmapGetLevelExp(hContext, hDevice, hImageMem, mipmapLevel,
                              phImageMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnMipmapFreeExp = dditable->BindlessImagesExp.pfnMipmapFreeExp;
  if (nullptr == pfnMipmapFreeExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMipmapFreeExp(hContext, hDevice, hMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImportExternalMemoryExp =
      dditable->BindlessImagesExp.pfnImportExternalMemoryExp;
  if (nullptr == pfnImportExternalMemoryExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImportExternalMemoryExp(hContext, hDevice, size, memHandleType,
                                    pExternalMemDesc, phExternalMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnMapExternalArrayExp =
      dditable->BindlessImagesExp.pfnMapExternalArrayExp;
  if (nullptr == pfnMapExternalArrayExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMapExternalArrayExp(hContext, hDevice, pImageFormat, pImageDesc,
                                hExternalMem, phImageMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnMapExternalLinearMemoryExp =
      dditable->BindlessImagesExp.pfnMapExternalLinearMemoryExp;
  if (nullptr == pfnMapExternalLinearMemoryExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnMapExternalLinearMemoryExp(hContext, hDevice, offset, size,
                                       hExternalMem, ppRetMem);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnReleaseExternalMemoryExp =
      dditable->BindlessImagesExp.pfnReleaseExternalMemoryExp;
  if (nullptr == pfnReleaseExternalMemoryExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnReleaseExternalMemoryExp(hContext, hDevice, hExternalMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesFreeMappedLinearMemoryExp
__urdlllocal ur_result_t UR_APICALL urBindlessImagesFreeMappedLinearMemoryExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in][release] pointer to mapped linear memory region to be freed
    void *pMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnFreeMappedLinearMemoryExp =
      dditable->BindlessImagesExp.pfnFreeMappedLinearMemoryExp;
  if (nullptr == pfnFreeMappedLinearMemoryExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnFreeMappedLinearMemoryExp(hContext, hDevice, pMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urBindlessImagesSupportsImportingHandleTypeExp
__urdlllocal ur_result_t UR_APICALL
urBindlessImagesSupportsImportingHandleTypeExp(
    /// [in] handle of the device object
    ur_device_handle_t hDevice,
    /// [in] type of external memory handle
    ur_exp_external_mem_type_t memHandleType,
    /// [out] whether the device supports importing the specified external
    /// memory handle type
    ur_bool_t *pSupportedRet) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hDevice);

  auto *pfnSupportsImportingHandleTypeExp =
      dditable->BindlessImagesExp.pfnSupportsImportingHandleTypeExp;
  if (nullptr == pfnSupportsImportingHandleTypeExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSupportsImportingHandleTypeExp(hDevice, memHandleType,
                                           pSupportedRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImportExternalSemaphoreExp =
      dditable->BindlessImagesExp.pfnImportExternalSemaphoreExp;
  if (nullptr == pfnImportExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImportExternalSemaphoreExp(hContext, hDevice, semHandleType,
                                       pExternalSemaphoreDesc,
                                       phExternalSemaphore);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnReleaseExternalSemaphoreExp =
      dditable->BindlessImagesExp.pfnReleaseExternalSemaphoreExp;
  if (nullptr == pfnReleaseExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnReleaseExternalSemaphoreExp(hContext, hDevice, hExternalSemaphore);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnWaitExternalSemaphoreExp =
      dditable->BindlessImagesExp.pfnWaitExternalSemaphoreExp;
  if (nullptr == pfnWaitExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnWaitExternalSemaphoreExp(hQueue, hSemaphore, hasWaitValue,
                                     waitValue, numEventsInWaitList,
                                     phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnSignalExternalSemaphoreExp =
      dditable->BindlessImagesExp.pfnSignalExternalSemaphoreExp;
  if (nullptr == pfnSignalExternalSemaphoreExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnSignalExternalSemaphoreExp(hQueue, hSemaphore, hasSignalValue,
                                       signalValue, numEventsInWaitList,
                                       phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnCreateExp = dditable->CommandBufferExp.pfnCreateExp;
  if (nullptr == pfnCreateExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCreateExp(hContext, hDevice, pCommandBufferDesc, phCommandBuffer);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferRetainExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferRetainExp(
    /// [in][retain] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnRetainExp = dditable->CommandBufferExp.pfnRetainExp;
  if (nullptr == pfnRetainExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnRetainExp(hCommandBuffer);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferReleaseExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferReleaseExp(
    /// [in][release] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnReleaseExp = dditable->CommandBufferExp.pfnReleaseExp;
  if (nullptr == pfnReleaseExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnReleaseExp(hCommandBuffer);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferFinalizeExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnFinalizeExp = dditable->CommandBufferExp.pfnFinalizeExp;
  if (nullptr == pfnFinalizeExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnFinalizeExp(hCommandBuffer);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendKernelLaunchExp =
      dditable->CommandBufferExp.pfnAppendKernelLaunchExp;
  if (nullptr == pfnAppendKernelLaunchExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendKernelLaunchExp(
      hCommandBuffer, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numKernelAlternatives, phKernelAlternatives,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendUSMMemcpyExp =
      dditable->CommandBufferExp.pfnAppendUSMMemcpyExp;
  if (nullptr == pfnAppendUSMMemcpyExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendUSMMemcpyExp(hCommandBuffer, pDst, pSrc, size,
                               numSyncPointsInWaitList, pSyncPointWaitList,
                               numEventsInWaitList, phEventWaitList, pSyncPoint,
                               phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendUSMFillExp = dditable->CommandBufferExp.pfnAppendUSMFillExp;
  if (nullptr == pfnAppendUSMFillExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendUSMFillExp(hCommandBuffer, pMemory, pPattern, patternSize,
                             size, numSyncPointsInWaitList, pSyncPointWaitList,
                             numEventsInWaitList, phEventWaitList, pSyncPoint,
                             phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendMemBufferCopyExp =
      dditable->CommandBufferExp.pfnAppendMemBufferCopyExp;
  if (nullptr == pfnAppendMemBufferCopyExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendMemBufferCopyExp(
      hCommandBuffer, hSrcMem, hDstMem, srcOffset, dstOffset, size,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendMemBufferWriteExp =
      dditable->CommandBufferExp.pfnAppendMemBufferWriteExp;
  if (nullptr == pfnAppendMemBufferWriteExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendMemBufferWriteExp(hCommandBuffer, hBuffer, offset, size, pSrc,
                                    numSyncPointsInWaitList, pSyncPointWaitList,
                                    numEventsInWaitList, phEventWaitList,
                                    pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendMemBufferReadExp =
      dditable->CommandBufferExp.pfnAppendMemBufferReadExp;
  if (nullptr == pfnAppendMemBufferReadExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendMemBufferReadExp(hCommandBuffer, hBuffer, offset, size, pDst,
                                   numSyncPointsInWaitList, pSyncPointWaitList,
                                   numEventsInWaitList, phEventWaitList,
                                   pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendMemBufferCopyRectExp =
      dditable->CommandBufferExp.pfnAppendMemBufferCopyRectExp;
  if (nullptr == pfnAppendMemBufferCopyRectExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendMemBufferCopyRectExp(
      hCommandBuffer, hSrcMem, hDstMem, srcOrigin, dstOrigin, region,
      srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendMemBufferWriteRectExp =
      dditable->CommandBufferExp.pfnAppendMemBufferWriteRectExp;
  if (nullptr == pfnAppendMemBufferWriteRectExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendMemBufferWriteRectExp(
      hCommandBuffer, hBuffer, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendMemBufferReadRectExp =
      dditable->CommandBufferExp.pfnAppendMemBufferReadRectExp;
  if (nullptr == pfnAppendMemBufferReadRectExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendMemBufferReadRectExp(
      hCommandBuffer, hBuffer, bufferOffset, hostOffset, region, bufferRowPitch,
      bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendMemBufferFillExp =
      dditable->CommandBufferExp.pfnAppendMemBufferFillExp;
  if (nullptr == pfnAppendMemBufferFillExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendMemBufferFillExp(
      hCommandBuffer, hBuffer, pPattern, patternSize, offset, size,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendUSMPrefetchExp =
      dditable->CommandBufferExp.pfnAppendUSMPrefetchExp;
  if (nullptr == pfnAppendUSMPrefetchExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendUSMPrefetchExp(hCommandBuffer, pMemory, size, flags,
                                 numSyncPointsInWaitList, pSyncPointWaitList,
                                 numEventsInWaitList, phEventWaitList,
                                 pSyncPoint, phEvent, phCommand);
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
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendUSMAdviseExp =
      dditable->CommandBufferExp.pfnAppendUSMAdviseExp;
  if (nullptr == pfnAppendUSMAdviseExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendUSMAdviseExp(hCommandBuffer, pMemory, size, advice,
                               numSyncPointsInWaitList, pSyncPointWaitList,
                               numEventsInWaitList, phEventWaitList, pSyncPoint,
                               phEvent, phCommand);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferAppendNativeCommandExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferAppendNativeCommandExp(
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [in] Function calling the native underlying API, to be executed
    /// immediately.
    ur_exp_command_buffer_native_command_function_t pfnNativeCommand,
    /// [in][optional] Data used by pfnNativeCommand
    void *pData,
    /// [in][optional] A command-buffer object which will be added to
    /// hCommandBuffer as a child graph node containing the native commands.
    /// Required for CUDA and HIP adapters and will be ignored by other
    /// adapters, who use alternative backend mechanisms to add the native
    /// nodes to hCommandBuffer.
    ur_exp_command_buffer_handle_t hChildCommandBuffer,
    /// [in] The number of sync points in the provided dependency list.
    uint32_t numSyncPointsInWaitList,
    /// [in][optional] A list of sync points that this command depends on.
    /// Will be ignored if command-buffer is in-order.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    /// [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnAppendNativeCommandExp =
      dditable->CommandBufferExp.pfnAppendNativeCommandExp;
  if (nullptr == pfnAppendNativeCommandExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAppendNativeCommandExp(hCommandBuffer, pfnNativeCommand, pData,
                                   hChildCommandBuffer, numSyncPointsInWaitList,
                                   pSyncPointWaitList, pSyncPoint);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueCommandBufferExp
__urdlllocal ur_result_t UR_APICALL urEnqueueCommandBufferExp(
    /// [in] The queue to submit this command-buffer for execution.
    ur_queue_handle_t hQueue,
    /// [in] Handle of the command-buffer object.
    ur_exp_command_buffer_handle_t hCommandBuffer,
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnCommandBufferExp = dditable->EnqueueExp.pfnCommandBufferExp;
  if (nullptr == pfnCommandBufferExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCommandBufferExp(hQueue, hCommandBuffer, numEventsInWaitList,
                             phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnUpdateKernelLaunchExp =
      dditable->CommandBufferExp.pfnUpdateKernelLaunchExp;
  if (nullptr == pfnUpdateKernelLaunchExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUpdateKernelLaunchExp(hCommandBuffer, numKernelUpdates,
                                  pUpdateKernelLaunch);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferUpdateSignalEventExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferUpdateSignalEventExp(
    /// [in] Handle of the command-buffer command to update.
    ur_exp_command_buffer_command_handle_t hCommand,
    /// [out][alloc] Event to be signaled.
    ur_event_handle_t *phSignalEvent) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommand);

  auto *pfnUpdateSignalEventExp =
      dditable->CommandBufferExp.pfnUpdateSignalEventExp;
  if (nullptr == pfnUpdateSignalEventExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUpdateSignalEventExp(hCommand, phSignalEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommand);

  auto *pfnUpdateWaitEventsExp =
      dditable->CommandBufferExp.pfnUpdateWaitEventsExp;
  if (nullptr == pfnUpdateWaitEventsExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnUpdateWaitEventsExp(hCommand, numEventsInWaitList, phEventWaitList);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnGetInfoExp = dditable->CommandBufferExp.pfnGetInfoExp;
  if (nullptr == pfnGetInfoExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetInfoExp(hCommandBuffer, propName, propSize, pPropValue,
                       pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urCommandBufferGetNativeHandleExp
__urdlllocal ur_result_t UR_APICALL urCommandBufferGetNativeHandleExp(
    /// [in] Handle of the command-buffer.
    ur_exp_command_buffer_handle_t hCommandBuffer,
    /// [out] A pointer to the native handle of the command-buffer.
    ur_native_handle_t *phNativeCommandBuffer) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hCommandBuffer);

  auto *pfnGetNativeHandleExp =
      dditable->CommandBufferExp.pfnGetNativeHandleExp;
  if (nullptr == pfnGetNativeHandleExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnGetNativeHandleExp(hCommandBuffer, phNativeCommandBuffer);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnTimestampRecordingExp =
      dditable->EnqueueExp.pfnTimestampRecordingExp;
  if (nullptr == pfnTimestampRecordingExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnTimestampRecordingExp(hQueue, blocking, numEventsInWaitList,
                                  phEventWaitList, phEvent);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemoryExportAllocExportableMemoryExp
__urdlllocal ur_result_t UR_APICALL urMemoryExportAllocExportableMemoryExp(
    /// [in] Handle to context in which to allocate memory.
    ur_context_handle_t hContext,
    /// [in] Handle to device on which to allocate memory.
    ur_device_handle_t hDevice,
    /// [in] Requested alignment of the allocation.
    size_t alignment,
    /// [in] Requested size of the allocation.
    size_t size,
    /// [in] Type of the memory handle to be exported (e.g. file descriptor,
    /// or win32 NT handle).
    ur_exp_external_mem_type_t handleTypeToExport,
    /// [out][alloc] Pointer to allocated exportable memory.
    void **ppMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnAllocExportableMemoryExp =
      dditable->MemoryExportExp.pfnAllocExportableMemoryExp;
  if (nullptr == pfnAllocExportableMemoryExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnAllocExportableMemoryExp(hContext, hDevice, alignment, size,
                                     handleTypeToExport, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemoryExportFreeExportableMemoryExp
__urdlllocal ur_result_t UR_APICALL urMemoryExportFreeExportableMemoryExp(
    /// [in] Handle to context in which to free memory.
    ur_context_handle_t hContext,
    /// [in] Handle to device on which to free memory.
    ur_device_handle_t hDevice,
    /// [in][release] Pointer to exportable memory to be deallocated.
    void *pMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnFreeExportableMemoryExp =
      dditable->MemoryExportExp.pfnFreeExportableMemoryExp;
  if (nullptr == pfnFreeExportableMemoryExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnFreeExportableMemoryExp(hContext, hDevice, pMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemoryExportExportMemoryHandleExp
__urdlllocal ur_result_t UR_APICALL urMemoryExportExportMemoryHandleExp(
    /// [in] Handle to context in which the exportable memory was allocated.
    ur_context_handle_t hContext,
    /// [in] Handle to device on which the exportable memory was allocated.
    ur_device_handle_t hDevice,
    /// [in] Type of the memory handle to be exported (e.g. file descriptor,
    /// or win32 NT handle).
    ur_exp_external_mem_type_t handleTypeToExport,
    /// [in] Pointer to exportable memory handle.
    void *pMem,
    /// [out] Returned exportable handle to memory allocated in `pMem`
    void *pMemHandleRet) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnExportMemoryHandleExp =
      dditable->MemoryExportExp.pfnExportMemoryHandleExp;
  if (nullptr == pfnExportMemoryHandleExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnExportMemoryHandleExp(hContext, hDevice, handleTypeToExport, pMem,
                                  pMemHandleRet);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnBuildExp = dditable->ProgramExp.pfnBuildExp;
  if (nullptr == pfnBuildExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnBuildExp(hProgram, numDevices, phDevices, pOptions);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hProgram);

  auto *pfnCompileExp = dditable->ProgramExp.pfnCompileExp;
  if (nullptr == pfnCompileExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnCompileExp(hProgram, numDevices, phDevices, pOptions);
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
  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnLinkExp = dditable->ProgramExp.pfnLinkExp;
  if (nullptr == pfnLinkExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnLinkExp(hContext, numDevices, phDevices, count, phPrograms,
                    pOptions, phProgram);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMContextMemcpyExp
__urdlllocal ur_result_t UR_APICALL urUSMContextMemcpyExp(
    /// [in] Context associated with the device(s) that own the allocations
    /// `pSrc` and `pDst`.
    ur_context_handle_t hContext,
    /// [in] Destination pointer to copy to.
    void *pDst,
    /// [in] Source pointer to copy from.
    const void *pSrc,
    /// [in] Size in bytes to be copied.
    size_t size) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnContextMemcpyExp = dditable->USMExp.pfnContextMemcpyExp;
  if (nullptr == pfnContextMemcpyExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnContextMemcpyExp(hContext, pDst, pSrc, size);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnImportExp = dditable->USMExp.pfnImportExp;
  if (nullptr == pfnImportExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnImportExp(hContext, pMem, size);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMReleaseExp
__urdlllocal ur_result_t UR_APICALL urUSMReleaseExp(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] pointer to host memory object
    void *pMem) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hContext);

  auto *pfnReleaseExp = dditable->USMExp.pfnReleaseExp;
  if (nullptr == pfnReleaseExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnReleaseExp(hContext, pMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PEnablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    /// [in] handle of the command device object
    ur_device_handle_t commandDevice,
    /// [in] handle of the peer device object
    ur_device_handle_t peerDevice) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(commandDevice);

  auto *pfnEnablePeerAccessExp = dditable->UsmP2PExp.pfnEnablePeerAccessExp;
  if (nullptr == pfnEnablePeerAccessExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnEnablePeerAccessExp(commandDevice, peerDevice);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUsmP2PDisablePeerAccessExp
__urdlllocal ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    /// [in] handle of the command device object
    ur_device_handle_t commandDevice,
    /// [in] handle of the peer device object
    ur_device_handle_t peerDevice) {

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(commandDevice);

  auto *pfnDisablePeerAccessExp = dditable->UsmP2PExp.pfnDisablePeerAccessExp;
  if (nullptr == pfnDisablePeerAccessExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnDisablePeerAccessExp(commandDevice, peerDevice);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(commandDevice);

  auto *pfnPeerAccessGetInfoExp = dditable->UsmP2PExp.pfnPeerAccessGetInfoExp;
  if (nullptr == pfnPeerAccessGetInfoExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnPeerAccessGetInfoExp(commandDevice, peerDevice, propName, propSize,
                                 pPropValue, pPropSizeRet);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunchWithArgsExp
__urdlllocal ur_result_t UR_APICALL urEnqueueKernelLaunchWithArgsExp(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] number of dimensions, from 1 to 3, to specify the global and
    /// work-group work-items
    uint32_t workDim,
    /// [in][optional] pointer to an array of workDim unsigned values that
    /// specify the offset used to calculate the global ID of a work-item
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
    /// [in] Number of entries in pArgs
    uint32_t numArgs,
    /// [in][optional][range(0, numArgs)] pointer to a list of kernel arg
    /// properties.
    const ur_exp_kernel_arg_properties_t *pArgs,
    /// [in] size of the launch prop list
    uint32_t numPropsInLaunchPropList,
    /// [in][optional][range(0, numPropsInLaunchPropList)] pointer to a list
    /// of launch properties
    const ur_kernel_launch_property_t *launchPropList,
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnKernelLaunchWithArgsExp =
      dditable->EnqueueExp.pfnKernelLaunchWithArgsExp;
  if (nullptr == pfnKernelLaunchWithArgsExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnKernelLaunchWithArgsExp(
      hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numArgs, pArgs, numPropsInLaunchPropList, launchPropList,
      numEventsInWaitList, phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnEventsWaitWithBarrierExt =
      dditable->Enqueue.pfnEventsWaitWithBarrierExt;
  if (nullptr == pfnEventsWaitWithBarrierExt)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnEventsWaitWithBarrierExt(hQueue, pProperties, numEventsInWaitList,
                                     phEventWaitList, phEvent);
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

  auto *dditable = *reinterpret_cast<ur_dditable_t **>(hQueue);

  auto *pfnNativeCommandExp = dditable->EnqueueExp.pfnNativeCommandExp;
  if (nullptr == pfnNativeCommandExp)
    return UR_RESULT_ERROR_UNINITIALIZED;

  // forward to device-platform
  return pfnNativeCommandExp(hQueue, pfnNativeEnqueue, data, numMemsInMemList,
                             phMemList, pProperties, numEventsInWaitList,
                             phEventWaitList, phEvent);
}

} // namespace ur_loader

extern "C" {

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Adapter table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetAdapterProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_adapter_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetAdapterProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetAdapterProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Adapter);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnGet = ur_loader::urAdapterGet;
      pDdiTable->pfnRelease = ur_loader::urAdapterRelease;
      pDdiTable->pfnRetain = ur_loader::urAdapterRetain;
      pDdiTable->pfnGetLastError = ur_loader::urAdapterGetLastError;
      pDdiTable->pfnGetInfo = ur_loader::urAdapterGetInfo;
      pDdiTable->pfnSetLoggerCallback = ur_loader::urAdapterSetLoggerCallback;
      pDdiTable->pfnSetLoggerCallbackLevel =
          ur_loader::urAdapterSetLoggerCallbackLevel;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Adapter;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's BindlessImagesExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetBindlessImagesExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_bindless_images_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetBindlessImagesExpProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(
            platform.handle.get(), "urGetBindlessImagesExpProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus =
        getTable(version, &platform.dditable.BindlessImagesExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnUnsampledImageHandleDestroyExp =
          ur_loader::urBindlessImagesUnsampledImageHandleDestroyExp;
      pDdiTable->pfnSampledImageHandleDestroyExp =
          ur_loader::urBindlessImagesSampledImageHandleDestroyExp;
      pDdiTable->pfnImageAllocateExp =
          ur_loader::urBindlessImagesImageAllocateExp;
      pDdiTable->pfnImageFreeExp = ur_loader::urBindlessImagesImageFreeExp;
      pDdiTable->pfnUnsampledImageCreateExp =
          ur_loader::urBindlessImagesUnsampledImageCreateExp;
      pDdiTable->pfnSampledImageCreateExp =
          ur_loader::urBindlessImagesSampledImageCreateExp;
      pDdiTable->pfnImageCopyExp = ur_loader::urBindlessImagesImageCopyExp;
      pDdiTable->pfnImageGetInfoExp =
          ur_loader::urBindlessImagesImageGetInfoExp;
      pDdiTable->pfnGetImageMemoryHandleTypeSupportExp =
          ur_loader::urBindlessImagesGetImageMemoryHandleTypeSupportExp;
      pDdiTable->pfnGetImageUnsampledHandleSupportExp =
          ur_loader::urBindlessImagesGetImageUnsampledHandleSupportExp;
      pDdiTable->pfnGetImageSampledHandleSupportExp =
          ur_loader::urBindlessImagesGetImageSampledHandleSupportExp;
      pDdiTable->pfnMipmapGetLevelExp =
          ur_loader::urBindlessImagesMipmapGetLevelExp;
      pDdiTable->pfnMipmapFreeExp = ur_loader::urBindlessImagesMipmapFreeExp;
      pDdiTable->pfnImportExternalMemoryExp =
          ur_loader::urBindlessImagesImportExternalMemoryExp;
      pDdiTable->pfnMapExternalArrayExp =
          ur_loader::urBindlessImagesMapExternalArrayExp;
      pDdiTable->pfnMapExternalLinearMemoryExp =
          ur_loader::urBindlessImagesMapExternalLinearMemoryExp;
      pDdiTable->pfnReleaseExternalMemoryExp =
          ur_loader::urBindlessImagesReleaseExternalMemoryExp;
      pDdiTable->pfnFreeMappedLinearMemoryExp =
          ur_loader::urBindlessImagesFreeMappedLinearMemoryExp;
      pDdiTable->pfnSupportsImportingHandleTypeExp =
          ur_loader::urBindlessImagesSupportsImportingHandleTypeExp;
      pDdiTable->pfnImportExternalSemaphoreExp =
          ur_loader::urBindlessImagesImportExternalSemaphoreExp;
      pDdiTable->pfnReleaseExternalSemaphoreExp =
          ur_loader::urBindlessImagesReleaseExternalSemaphoreExp;
      pDdiTable->pfnWaitExternalSemaphoreExp =
          ur_loader::urBindlessImagesWaitExternalSemaphoreExp;
      pDdiTable->pfnSignalExternalSemaphoreExp =
          ur_loader::urBindlessImagesSignalExternalSemaphoreExp;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.BindlessImagesExp;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's CommandBufferExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetCommandBufferExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_command_buffer_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetCommandBufferExpProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(
            platform.handle.get(), "urGetCommandBufferExpProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus =
        getTable(version, &platform.dditable.CommandBufferExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnCreateExp = ur_loader::urCommandBufferCreateExp;
      pDdiTable->pfnRetainExp = ur_loader::urCommandBufferRetainExp;
      pDdiTable->pfnReleaseExp = ur_loader::urCommandBufferReleaseExp;
      pDdiTable->pfnFinalizeExp = ur_loader::urCommandBufferFinalizeExp;
      pDdiTable->pfnAppendKernelLaunchExp =
          ur_loader::urCommandBufferAppendKernelLaunchExp;
      pDdiTable->pfnAppendUSMMemcpyExp =
          ur_loader::urCommandBufferAppendUSMMemcpyExp;
      pDdiTable->pfnAppendUSMFillExp =
          ur_loader::urCommandBufferAppendUSMFillExp;
      pDdiTable->pfnAppendMemBufferCopyExp =
          ur_loader::urCommandBufferAppendMemBufferCopyExp;
      pDdiTable->pfnAppendMemBufferWriteExp =
          ur_loader::urCommandBufferAppendMemBufferWriteExp;
      pDdiTable->pfnAppendMemBufferReadExp =
          ur_loader::urCommandBufferAppendMemBufferReadExp;
      pDdiTable->pfnAppendMemBufferCopyRectExp =
          ur_loader::urCommandBufferAppendMemBufferCopyRectExp;
      pDdiTable->pfnAppendMemBufferWriteRectExp =
          ur_loader::urCommandBufferAppendMemBufferWriteRectExp;
      pDdiTable->pfnAppendMemBufferReadRectExp =
          ur_loader::urCommandBufferAppendMemBufferReadRectExp;
      pDdiTable->pfnAppendMemBufferFillExp =
          ur_loader::urCommandBufferAppendMemBufferFillExp;
      pDdiTable->pfnAppendUSMPrefetchExp =
          ur_loader::urCommandBufferAppendUSMPrefetchExp;
      pDdiTable->pfnAppendUSMAdviseExp =
          ur_loader::urCommandBufferAppendUSMAdviseExp;
      pDdiTable->pfnAppendNativeCommandExp =
          ur_loader::urCommandBufferAppendNativeCommandExp;
      pDdiTable->pfnUpdateKernelLaunchExp =
          ur_loader::urCommandBufferUpdateKernelLaunchExp;
      pDdiTable->pfnUpdateSignalEventExp =
          ur_loader::urCommandBufferUpdateSignalEventExp;
      pDdiTable->pfnUpdateWaitEventsExp =
          ur_loader::urCommandBufferUpdateWaitEventsExp;
      pDdiTable->pfnGetInfoExp = ur_loader::urCommandBufferGetInfoExp;
      pDdiTable->pfnGetNativeHandleExp =
          ur_loader::urCommandBufferGetNativeHandleExp;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.CommandBufferExp;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetContextProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_context_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetContextProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetContextProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Context);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnCreate = ur_loader::urContextCreate;
      pDdiTable->pfnRetain = ur_loader::urContextRetain;
      pDdiTable->pfnRelease = ur_loader::urContextRelease;
      pDdiTable->pfnGetInfo = ur_loader::urContextGetInfo;
      pDdiTable->pfnGetNativeHandle = ur_loader::urContextGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urContextCreateWithNativeHandle;
      pDdiTable->pfnSetExtendedDeleter = ur_loader::urContextSetExtendedDeleter;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Context;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetEnqueueProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetEnqueueProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Enqueue);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnKernelLaunch = ur_loader::urEnqueueKernelLaunch;
      pDdiTable->pfnEventsWait = ur_loader::urEnqueueEventsWait;
      pDdiTable->pfnEventsWaitWithBarrier =
          ur_loader::urEnqueueEventsWaitWithBarrier;
      pDdiTable->pfnMemBufferRead = ur_loader::urEnqueueMemBufferRead;
      pDdiTable->pfnMemBufferWrite = ur_loader::urEnqueueMemBufferWrite;
      pDdiTable->pfnMemBufferReadRect = ur_loader::urEnqueueMemBufferReadRect;
      pDdiTable->pfnMemBufferWriteRect = ur_loader::urEnqueueMemBufferWriteRect;
      pDdiTable->pfnMemBufferCopy = ur_loader::urEnqueueMemBufferCopy;
      pDdiTable->pfnMemBufferCopyRect = ur_loader::urEnqueueMemBufferCopyRect;
      pDdiTable->pfnMemBufferFill = ur_loader::urEnqueueMemBufferFill;
      pDdiTable->pfnMemImageRead = ur_loader::urEnqueueMemImageRead;
      pDdiTable->pfnMemImageWrite = ur_loader::urEnqueueMemImageWrite;
      pDdiTable->pfnMemImageCopy = ur_loader::urEnqueueMemImageCopy;
      pDdiTable->pfnMemBufferMap = ur_loader::urEnqueueMemBufferMap;
      pDdiTable->pfnMemUnmap = ur_loader::urEnqueueMemUnmap;
      pDdiTable->pfnUSMFill = ur_loader::urEnqueueUSMFill;
      pDdiTable->pfnUSMMemcpy = ur_loader::urEnqueueUSMMemcpy;
      pDdiTable->pfnUSMPrefetch = ur_loader::urEnqueueUSMPrefetch;
      pDdiTable->pfnUSMAdvise = ur_loader::urEnqueueUSMAdvise;
      pDdiTable->pfnUSMFill2D = ur_loader::urEnqueueUSMFill2D;
      pDdiTable->pfnUSMMemcpy2D = ur_loader::urEnqueueUSMMemcpy2D;
      pDdiTable->pfnDeviceGlobalVariableWrite =
          ur_loader::urEnqueueDeviceGlobalVariableWrite;
      pDdiTable->pfnDeviceGlobalVariableRead =
          ur_loader::urEnqueueDeviceGlobalVariableRead;
      pDdiTable->pfnReadHostPipe = ur_loader::urEnqueueReadHostPipe;
      pDdiTable->pfnWriteHostPipe = ur_loader::urEnqueueWriteHostPipe;
      pDdiTable->pfnEventsWaitWithBarrierExt =
          ur_loader::urEnqueueEventsWaitWithBarrierExt;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Enqueue;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's EnqueueExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEnqueueExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetEnqueueExpProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetEnqueueExpProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.EnqueueExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnKernelLaunchWithArgsExp =
          ur_loader::urEnqueueKernelLaunchWithArgsExp;
      pDdiTable->pfnUSMDeviceAllocExp = ur_loader::urEnqueueUSMDeviceAllocExp;
      pDdiTable->pfnUSMSharedAllocExp = ur_loader::urEnqueueUSMSharedAllocExp;
      pDdiTable->pfnUSMHostAllocExp = ur_loader::urEnqueueUSMHostAllocExp;
      pDdiTable->pfnUSMFreeExp = ur_loader::urEnqueueUSMFreeExp;
      pDdiTable->pfnCommandBufferExp = ur_loader::urEnqueueCommandBufferExp;
      pDdiTable->pfnTimestampRecordingExp =
          ur_loader::urEnqueueTimestampRecordingExp;
      pDdiTable->pfnNativeCommandExp = ur_loader::urEnqueueNativeCommandExp;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.EnqueueExp;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Event table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetEventProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_event_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetEventProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetEventProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Event);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnGetInfo = ur_loader::urEventGetInfo;
      pDdiTable->pfnGetProfilingInfo = ur_loader::urEventGetProfilingInfo;
      pDdiTable->pfnWait = ur_loader::urEventWait;
      pDdiTable->pfnRetain = ur_loader::urEventRetain;
      pDdiTable->pfnRelease = ur_loader::urEventRelease;
      pDdiTable->pfnGetNativeHandle = ur_loader::urEventGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urEventCreateWithNativeHandle;
      pDdiTable->pfnSetCallback = ur_loader::urEventSetCallback;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Event;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetKernelProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_kernel_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetKernelProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetKernelProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Kernel);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnCreate = ur_loader::urKernelCreate;
      pDdiTable->pfnGetInfo = ur_loader::urKernelGetInfo;
      pDdiTable->pfnGetGroupInfo = ur_loader::urKernelGetGroupInfo;
      pDdiTable->pfnGetSubGroupInfo = ur_loader::urKernelGetSubGroupInfo;
      pDdiTable->pfnRetain = ur_loader::urKernelRetain;
      pDdiTable->pfnRelease = ur_loader::urKernelRelease;
      pDdiTable->pfnGetNativeHandle = ur_loader::urKernelGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urKernelCreateWithNativeHandle;
      pDdiTable->pfnGetSuggestedLocalWorkSize =
          ur_loader::urKernelGetSuggestedLocalWorkSize;
      pDdiTable->pfnSetArgValue = ur_loader::urKernelSetArgValue;
      pDdiTable->pfnSetArgLocal = ur_loader::urKernelSetArgLocal;
      pDdiTable->pfnSetArgPointer = ur_loader::urKernelSetArgPointer;
      pDdiTable->pfnSetExecInfo = ur_loader::urKernelSetExecInfo;
      pDdiTable->pfnSetArgSampler = ur_loader::urKernelSetArgSampler;
      pDdiTable->pfnSetArgMemObj = ur_loader::urKernelSetArgMemObj;
      pDdiTable->pfnSetSpecializationConstants =
          ur_loader::urKernelSetSpecializationConstants;
      pDdiTable->pfnSuggestMaxCooperativeGroupCount =
          ur_loader::urKernelSuggestMaxCooperativeGroupCount;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Kernel;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_mem_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetMemProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetMemProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Mem);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnImageCreate = ur_loader::urMemImageCreate;
      pDdiTable->pfnBufferCreate = ur_loader::urMemBufferCreate;
      pDdiTable->pfnRetain = ur_loader::urMemRetain;
      pDdiTable->pfnRelease = ur_loader::urMemRelease;
      pDdiTable->pfnBufferPartition = ur_loader::urMemBufferPartition;
      pDdiTable->pfnGetNativeHandle = ur_loader::urMemGetNativeHandle;
      pDdiTable->pfnBufferCreateWithNativeHandle =
          ur_loader::urMemBufferCreateWithNativeHandle;
      pDdiTable->pfnImageCreateWithNativeHandle =
          ur_loader::urMemImageCreateWithNativeHandle;
      pDdiTable->pfnGetInfo = ur_loader::urMemGetInfo;
      pDdiTable->pfnImageGetInfo = ur_loader::urMemImageGetInfo;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Mem;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's MemoryExportExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetMemoryExportExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_memory_export_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetMemoryExportExpProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(
            platform.handle.get(), "urGetMemoryExportExpProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.MemoryExportExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnAllocExportableMemoryExp =
          ur_loader::urMemoryExportAllocExportableMemoryExp;
      pDdiTable->pfnFreeExportableMemoryExp =
          ur_loader::urMemoryExportFreeExportableMemoryExp;
      pDdiTable->pfnExportMemoryHandleExp =
          ur_loader::urMemoryExportExportMemoryHandleExp;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.MemoryExportExp;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's PhysicalMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetPhysicalMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_physical_mem_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetPhysicalMemProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetPhysicalMemProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.PhysicalMem);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnCreate = ur_loader::urPhysicalMemCreate;
      pDdiTable->pfnRetain = ur_loader::urPhysicalMemRetain;
      pDdiTable->pfnRelease = ur_loader::urPhysicalMemRelease;
      pDdiTable->pfnGetInfo = ur_loader::urPhysicalMemGetInfo;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.PhysicalMem;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Platform table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetPlatformProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_platform_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetPlatformProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetPlatformProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Platform);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnGet = ur_loader::urPlatformGet;
      pDdiTable->pfnGetInfo = ur_loader::urPlatformGetInfo;
      pDdiTable->pfnGetNativeHandle = ur_loader::urPlatformGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urPlatformCreateWithNativeHandle;
      pDdiTable->pfnGetApiVersion = ur_loader::urPlatformGetApiVersion;
      pDdiTable->pfnGetBackendOption = ur_loader::urPlatformGetBackendOption;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Platform;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_program_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetProgramProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetProgramProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Program);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnCreateWithIL = ur_loader::urProgramCreateWithIL;
      pDdiTable->pfnCreateWithBinary = ur_loader::urProgramCreateWithBinary;
      pDdiTable->pfnBuild = ur_loader::urProgramBuild;
      pDdiTable->pfnCompile = ur_loader::urProgramCompile;
      pDdiTable->pfnLink = ur_loader::urProgramLink;
      pDdiTable->pfnRetain = ur_loader::urProgramRetain;
      pDdiTable->pfnRelease = ur_loader::urProgramRelease;
      pDdiTable->pfnGetFunctionPointer = ur_loader::urProgramGetFunctionPointer;
      pDdiTable->pfnGetGlobalVariablePointer =
          ur_loader::urProgramGetGlobalVariablePointer;
      pDdiTable->pfnGetInfo = ur_loader::urProgramGetInfo;
      pDdiTable->pfnGetBuildInfo = ur_loader::urProgramGetBuildInfo;
      pDdiTable->pfnSetSpecializationConstants =
          ur_loader::urProgramSetSpecializationConstants;
      pDdiTable->pfnGetNativeHandle = ur_loader::urProgramGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urProgramCreateWithNativeHandle;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Program;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_program_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetProgramExpProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetProgramExpProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.ProgramExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnBuildExp = ur_loader::urProgramBuildExp;
      pDdiTable->pfnCompileExp = ur_loader::urProgramCompileExp;
      pDdiTable->pfnLinkExp = ur_loader::urProgramLinkExp;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.ProgramExp;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Queue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetQueueProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_queue_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetQueueProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetQueueProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Queue);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnGetInfo = ur_loader::urQueueGetInfo;
      pDdiTable->pfnCreate = ur_loader::urQueueCreate;
      pDdiTable->pfnRetain = ur_loader::urQueueRetain;
      pDdiTable->pfnRelease = ur_loader::urQueueRelease;
      pDdiTable->pfnGetNativeHandle = ur_loader::urQueueGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urQueueCreateWithNativeHandle;
      pDdiTable->pfnFinish = ur_loader::urQueueFinish;
      pDdiTable->pfnFlush = ur_loader::urQueueFlush;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Queue;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Sampler table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetSamplerProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_sampler_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetSamplerProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetSamplerProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Sampler);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnCreate = ur_loader::urSamplerCreate;
      pDdiTable->pfnRetain = ur_loader::urSamplerRetain;
      pDdiTable->pfnRelease = ur_loader::urSamplerRelease;
      pDdiTable->pfnGetInfo = ur_loader::urSamplerGetInfo;
      pDdiTable->pfnGetNativeHandle = ur_loader::urSamplerGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urSamplerCreateWithNativeHandle;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Sampler;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetUSMProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetUSMProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.USM);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnHostAlloc = ur_loader::urUSMHostAlloc;
      pDdiTable->pfnDeviceAlloc = ur_loader::urUSMDeviceAlloc;
      pDdiTable->pfnSharedAlloc = ur_loader::urUSMSharedAlloc;
      pDdiTable->pfnFree = ur_loader::urUSMFree;
      pDdiTable->pfnGetMemAllocInfo = ur_loader::urUSMGetMemAllocInfo;
      pDdiTable->pfnPoolCreate = ur_loader::urUSMPoolCreate;
      pDdiTable->pfnPoolRetain = ur_loader::urUSMPoolRetain;
      pDdiTable->pfnPoolRelease = ur_loader::urUSMPoolRelease;
      pDdiTable->pfnPoolGetInfo = ur_loader::urUSMPoolGetInfo;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.USM;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USMExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUSMExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetUSMExpProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetUSMExpProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.USMExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnPoolCreateExp = ur_loader::urUSMPoolCreateExp;
      pDdiTable->pfnPoolDestroyExp = ur_loader::urUSMPoolDestroyExp;
      pDdiTable->pfnPoolGetDefaultDevicePoolExp =
          ur_loader::urUSMPoolGetDefaultDevicePoolExp;
      pDdiTable->pfnPoolGetInfoExp = ur_loader::urUSMPoolGetInfoExp;
      pDdiTable->pfnPoolSetInfoExp = ur_loader::urUSMPoolSetInfoExp;
      pDdiTable->pfnPoolSetDevicePoolExp = ur_loader::urUSMPoolSetDevicePoolExp;
      pDdiTable->pfnPoolGetDevicePoolExp = ur_loader::urUSMPoolGetDevicePoolExp;
      pDdiTable->pfnPoolTrimToExp = ur_loader::urUSMPoolTrimToExp;
      pDdiTable->pfnPitchedAllocExp = ur_loader::urUSMPitchedAllocExp;
      pDdiTable->pfnContextMemcpyExp = ur_loader::urUSMContextMemcpyExp;
      pDdiTable->pfnImportExp = ur_loader::urUSMImportExp;
      pDdiTable->pfnReleaseExp = ur_loader::urUSMReleaseExp;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.USMExp;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's UsmP2PExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetUsmP2PExpProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_p2p_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetUsmP2PExpProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetUsmP2PExpProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.UsmP2PExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnEnablePeerAccessExp =
          ur_loader::urUsmP2PEnablePeerAccessExp;
      pDdiTable->pfnDisablePeerAccessExp =
          ur_loader::urUsmP2PDisablePeerAccessExp;
      pDdiTable->pfnPeerAccessGetInfoExp =
          ur_loader::urUsmP2PPeerAccessGetInfoExp;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.UsmP2PExp;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's VirtualMem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetVirtualMemProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_virtual_mem_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetVirtualMemProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetVirtualMemProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.VirtualMem);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnGranularityGetInfo =
          ur_loader::urVirtualMemGranularityGetInfo;
      pDdiTable->pfnReserve = ur_loader::urVirtualMemReserve;
      pDdiTable->pfnFree = ur_loader::urVirtualMemFree;
      pDdiTable->pfnMap = ur_loader::urVirtualMemMap;
      pDdiTable->pfnUnmap = ur_loader::urVirtualMemUnmap;
      pDdiTable->pfnSetAccess = ur_loader::urVirtualMemSetAccess;
      pDdiTable->pfnGetInfo = ur_loader::urVirtualMemGetInfo;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable =
          ur_loader::getContext()->platforms.front().dditable.VirtualMem;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Device table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    /// [in] API version requested
    ur_api_version_t version,
    /// [in,out] pointer to table of DDI function pointers
    ur_device_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (ur_loader::getContext()->version < version)
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

  ur_result_t result = UR_RESULT_SUCCESS;

  // Load the device-platform DDI tables
  for (auto &platform : ur_loader::getContext()->platforms) {
    // statically linked adapter inside of the loader
    if (platform.handle == nullptr)
      continue;

    if (platform.initStatus != UR_RESULT_SUCCESS)
      continue;
    auto getTable = reinterpret_cast<ur_pfnGetDeviceProcAddrTable_t>(
        ur_loader::LibLoader::getFunctionPtr(platform.handle.get(),
                                             "urGetDeviceProcAddrTable"));
    if (!getTable)
      continue;
    platform.initStatus = getTable(version, &platform.dditable.Device);
  }

  if (UR_RESULT_SUCCESS == result) {
    if (ur_loader::getContext()->platforms.size() != 1 ||
        ur_loader::getContext()->forceIntercept) {
      // return pointers to loader's DDIs
      pDdiTable->pfnGet = ur_loader::urDeviceGet;
      pDdiTable->pfnGetInfo = ur_loader::urDeviceGetInfo;
      pDdiTable->pfnRetain = ur_loader::urDeviceRetain;
      pDdiTable->pfnRelease = ur_loader::urDeviceRelease;
      pDdiTable->pfnPartition = ur_loader::urDevicePartition;
      pDdiTable->pfnSelectBinary = ur_loader::urDeviceSelectBinary;
      pDdiTable->pfnGetNativeHandle = ur_loader::urDeviceGetNativeHandle;
      pDdiTable->pfnCreateWithNativeHandle =
          ur_loader::urDeviceCreateWithNativeHandle;
      pDdiTable->pfnGetGlobalTimestamps =
          ur_loader::urDeviceGetGlobalTimestamps;
    } else {
      // return pointers directly to platform's DDIs
      *pDdiTable = ur_loader::getContext()->platforms.front().dditable.Device;
    }
  }

  return result;
}
}
