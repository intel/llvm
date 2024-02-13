/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_libapi.cpp
 *
 * @brief C++ library for ur
 *
 */
#include "ur_lib.hpp"

extern "C" {

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a loader config object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phLoaderConfig`
ur_result_t UR_APICALL urLoaderConfigCreate(
    ur_loader_config_handle_t *
        phLoaderConfig ///< [out] Pointer to handle of loader config object created.
    ) try {
    return ur_lib::urLoaderConfigCreate(phLoaderConfig);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the loader config object.
///
/// @details
///     - Get a reference to the loader config handle. Increment its reference
///       count
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
ur_result_t UR_APICALL urLoaderConfigRetain(
    ur_loader_config_handle_t
        hLoaderConfig ///< [in] loader config handle to retain
    ) try {
    return ur_lib::urLoaderConfigRetain(hLoaderConfig);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Release config handle.
///
/// @details
///     - Decrement reference count and destroy the config handle if reference
///       count becomes zero.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
ur_result_t UR_APICALL urLoaderConfigRelease(
    ur_loader_config_handle_t hLoaderConfig ///< [in] config handle to release
    ) try {
    return ur_lib::urLoaderConfigRelease(hLoaderConfig);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about the loader.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_LOADER_CONFIG_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the loader.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urLoaderConfigGetInfo(
    ur_loader_config_handle_t
        hLoaderConfig, ///< [in] handle of the loader config object
    ur_loader_config_info_t propName, ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return the info
    ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
    ) try {
    return ur_lib::urLoaderConfigGetInfo(hLoaderConfig, propName, propSize,
                                         pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enable a layer for the specified loader config.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pLayerName`
///     - ::UR_RESULT_ERROR_LAYER_NOT_PRESENT
///         + If layer specified with `pLayerName` can't be found by the loader.
ur_result_t UR_APICALL urLoaderConfigEnableLayer(
    ur_loader_config_handle_t
        hLoaderConfig, ///< [in] Handle to config object the layer will be enabled for.
    const char *
        pLayerName ///< [in] Null terminated string containing the name of the layer to
                   ///< enable.
    ) try {
    return ur_lib::urLoaderConfigEnableLayer(hLoaderConfig, pLayerName);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a function callback for use by the loader to retrieve code
///        location information.
///
/// @details
///     - The code location callback is optional and provides additional
///       information to the tracing layer about the entry point of the current
///       execution flow.
///     - This functionality can be used to match traced unified runtime
///       function calls with higher-level user calls.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnCodeloc`
ur_result_t UR_APICALL urLoaderConfigSetCodeLocationCallback(
    ur_loader_config_handle_t
        hLoaderConfig, ///< [in] Handle to config object the layer will be enabled for.
    ur_code_location_callback_t
        pfnCodeloc, ///< [in] Function pointer to code location callback.
    void *
        pUserData ///< [in][out][optional] pointer to data to be passed to callback.
    ) try {
    return ur_lib::urLoaderConfigSetCodeLocationCallback(hLoaderConfig,
                                                         pfnCodeloc, pUserData);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Initialize the 'oneAPI' loader
///
/// @details
///     - The application must call this function before calling any other
///       function.
///     - If this function is not called then all other functions will return
///       ::UR_RESULT_ERROR_UNINITIALIZED.
///     - Only one instance of the loader will be initialized per process.
///     - The application may call this function multiple times with different
///       flags or environment variables enabled.
///     - The application must call this function after forking new processes.
///       Each forked process must call this function.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe for scenarios
///       where multiple libraries may initialize the loader simultaneously.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_INIT_FLAGS_MASK & device_flags`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urLoaderInit(
    ur_device_init_flags_t device_flags, ///< [in] device initialization flags.
    ///< must be 0 (default) or a combination of ::ur_device_init_flag_t.
    ur_loader_config_handle_t
        hLoaderConfig ///< [in][optional] Handle of loader config handle.
    ) try {

    if (UR_DEVICE_INIT_FLAGS_MASK & device_flags) {
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }

    static ur_result_t result = UR_RESULT_SUCCESS;
    std::call_once(ur_lib::context->initOnce, [device_flags, hLoaderConfig]() {
        result = ur_lib::context->Init(device_flags, hLoaderConfig);
    });

    return result;
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Tear down the 'oneAPI' loader and release all its resources
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urLoaderTearDown(void) try {
    return ur_lib::urLoaderTearDown();
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all available adapters
///
/// @details
///     - Adapter implementations must return exactly one adapter handle from
///       this entry point.
///     - The loader may return more than one adapter handle when there are
///       multiple available.
///     - Each returned adapter has its reference count incremented and should
///       be released with a subsequent call to ::urAdapterRelease.
///     - Adapters may perform adapter-specific state initialization when the
///       first reference to them is taken.
///     - An application may call this entry point multiple times to acquire
///       multiple references to the adapter handle(s).
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_SIZE
ur_result_t UR_APICALL urAdapterGet(
    uint32_t
        NumEntries, ///< [in] the number of adapters to be added to phAdapters.
    ///< If phAdapters is not NULL, then NumEntries should be greater than
    ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
    ///< will be returned.
    ur_adapter_handle_t *
        phAdapters, ///< [out][optional][range(0, NumEntries)] array of handle of adapters.
    ///< If NumEntries is less than the number of adapters available, then
    ///< ::urAdapterGet shall only retrieve that number of platforms.
    uint32_t *
        pNumAdapters ///< [out][optional] returns the total number of adapters available.
    ) try {
    auto pfnAdapterGet = ur_lib::context->urDdiTable.Global.pfnAdapterGet;
    if (nullptr == pfnAdapterGet) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAdapterGet(NumEntries, phAdapters, pNumAdapters);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Releases the adapter handle reference indicating end of its usage
///
/// @details
///     - When the reference count of the adapter reaches zero, the adapter may
///       perform adapter-specififc resource teardown. Resources must be left in
///       a state where it safe for the adapter to be subsequently reinitialized
///       with ::urAdapterGet
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
ur_result_t UR_APICALL urAdapterRelease(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to release
    ) try {
    auto pfnAdapterRelease =
        ur_lib::context->urDdiTable.Global.pfnAdapterRelease;
    if (nullptr == pfnAdapterRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAdapterRelease(hAdapter);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the adapter handle.
///
/// @details
///     - Get a reference to the adapter handle. Increment its reference count
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
ur_result_t UR_APICALL urAdapterRetain(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to retain
    ) try {
    auto pfnAdapterRetain = ur_lib::context->urDdiTable.Global.pfnAdapterRetain;
    if (nullptr == pfnAdapterRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAdapterRetain(hAdapter);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the last adapter specific error.
///
/// @details
/// To be used after another entry-point has returned
/// ::UR_RESULT_ERROR_ADAPTER_SPECIFIC in order to retrieve a message describing
/// the circumstances of the underlying driver error and the error code
/// returned by the failed driver entry-point.
///
/// * Implementations *must* store the message and error code in thread-local
///   storage prior to returning ::UR_RESULT_ERROR_ADAPTER_SPECIFIC.
///
/// * The message and error code storage is will only be valid if a previously
///   called entry-point returned ::UR_RESULT_ERROR_ADAPTER_SPECIFIC.
///
/// * The memory pointed to by the C string returned in `ppMessage` is owned by
///   the adapter and *must* be null terminated.
///
/// * The application *may* call this function from simultaneous threads.
///
/// * The implementation of this function *should* be lock-free.
///
/// Example usage:
///
/// ```cpp
/// if (::urQueueCreate(hContext, hDevice, nullptr, &hQueue) ==
///         ::UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
///     const char* pMessage;
///     int32_t error;
///     ::urAdapterGetLastError(hAdapter, &pMessage, &error);
/// }
/// ```
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMessage`
///         + `NULL == pError`
ur_result_t UR_APICALL urAdapterGetLastError(
    ur_adapter_handle_t hAdapter, ///< [in] handle of the adapter instance
    const char **
        ppMessage, ///< [out] pointer to a C string where the adapter specific error message
                   ///< will be stored.
    int32_t *
        pError ///< [out] pointer to an integer where the adapter specific error code will
               ///< be stored.
    ) try {
    auto pfnAdapterGetLastError =
        ur_lib::context->urDdiTable.Global.pfnAdapterGetLastError;
    if (nullptr == pfnAdapterGetLastError) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAdapterGetLastError(hAdapter, ppMessage, pError);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves information about the adapter
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_ADAPTER_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urAdapterGetInfo(
    ur_adapter_handle_t hAdapter, ///< [in] handle of the adapter
    ur_adapter_info_t propName,   ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If Size is not equal to or greater to the real number of bytes needed
    ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual number of bytes being queried by pPropValue.
    ) try {
    auto pfnAdapterGetInfo =
        ur_lib::context->urDdiTable.Global.pfnAdapterGetInfo;
    if (nullptr == pfnAdapterGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAdapterGetInfo(hAdapter, propName, propSize, pPropValue,
                             pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all available platforms for the given adapters
///
/// @details
///     - Multiple calls to this function will return identical platforms
///       handles, in the same order.
///     - The application may call this function from simultaneous threads, the
///       implementation must be thread-safe
///
/// @remarks
///   _Analogues_
///     - **clGetPlatformIDs**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phAdapters`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NumEntries == 0 && phPlatforms != NULL`
ur_result_t UR_APICALL urPlatformGet(
    ur_adapter_handle_t *
        phAdapters, ///< [in][range(0, NumAdapters)] array of adapters to query for platforms.
    uint32_t NumAdapters, ///< [in] number of adapters pointed to by phAdapters
    uint32_t
        NumEntries, ///< [in] the number of platforms to be added to phPlatforms.
    ///< If phPlatforms is not NULL, then NumEntries should be greater than
    ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
    ///< will be returned.
    ur_platform_handle_t *
        phPlatforms, ///< [out][optional][range(0, NumEntries)] array of handle of platforms.
    ///< If NumEntries is less than the number of platforms available, then
    ///< ::urPlatformGet shall only retrieve that number of platforms.
    uint32_t *
        pNumPlatforms ///< [out][optional] returns the total number of platforms available.
    ) try {
    auto pfnGet = ur_lib::context->urDdiTable.Platform.pfnGet;
    if (nullptr == pfnGet) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGet(phAdapters, NumAdapters, NumEntries, phPlatforms,
                  pNumPlatforms);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about platform
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clGetPlatformInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PLATFORM_INFO_BACKEND < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_PLATFORM
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urPlatformGetInfo(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_platform_info_t propName,    ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPlatformInfo.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If Size is not equal to or greater to the real number of bytes needed
    ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPlatformInfo is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual number of bytes being queried by pPlatformInfo.
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Platform.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hPlatform, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the API version supported by the specified platform
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pVersion`
ur_result_t UR_APICALL urPlatformGetApiVersion(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_api_version_t *pVersion      ///< [out] api version
    ) try {
    auto pfnGetApiVersion =
        ur_lib::context->urDdiTable.Platform.pfnGetApiVersion;
    if (nullptr == pfnGetApiVersion) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetApiVersion(hPlatform, pVersion);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native platform handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativePlatform`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform.
    ur_native_handle_t *
        phNativePlatform ///< [out] a pointer to the native handle of the platform.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Platform.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hPlatform, phNativePlatform);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime platform object from native platform handle.
///
/// @details
///     - Creates runtime platform handle from native driver platform handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPlatform`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urPlatformCreateWithNativeHandle(
    ur_native_handle_t
        hNativePlatform, ///< [in][nocheck] the native handle of the platform.
    const ur_platform_native_properties_t *
        pProperties, ///< [in][optional] pointer to native platform properties struct.
    ur_platform_handle_t *
        phPlatform ///< [out] pointer to the handle of the platform object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Platform.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativePlatform, pProperties, phPlatform);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the platform specific compiler backend option from a generic
///        frontend option.
///
/// @details
///     - The string returned via the ppPlatformOption is a NULL terminated C
///       style string.
///     - The string returned via the ppPlatformOption is thread local.
///     - The memory in the string returned via the ppPlatformOption is owned by
///       the adapter.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pFrontendOption`
///         + `NULL == ppPlatformOption`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + If `pFrontendOption` is not a valid frontend option.
ur_result_t UR_APICALL urPlatformGetBackendOption(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance.
    const char
        *pFrontendOption, ///< [in] string containing the frontend option.
    const char **
        ppPlatformOption ///< [out] returns the correct platform specific compiler option based on
                         ///< the frontend option.
    ) try {
    auto pfnGetBackendOption =
        ur_lib::context->urDdiTable.Platform.pfnGetBackendOption;
    if (nullptr == pfnGetBackendOption) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetBackendOption(hPlatform, pFrontendOption, ppPlatformOption);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves devices within a platform
///
/// @details
///     - Multiple calls to this function will return identical device handles,
///       in the same order.
///     - The number and order of handles returned from this function can be
///       affected by environment variables that filter devices exposed through
///       API.
///     - The returned devices are taken a reference of and must be released
///       with a subsequent call to ::urDeviceRelease.
///     - The application may call this function from simultaneous threads, the
///       implementation must be thread-safe
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceIDs**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_TYPE_VPU < DeviceType`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NumEntries == 0 && phDevices != NULL`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NumEntries > 0 && phDevices == NULL`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urDeviceGet(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    ur_device_type_t DeviceType,    ///< [in] the type of the devices.
    uint32_t
        NumEntries, ///< [in] the number of devices to be added to phDevices.
    ///< If phDevices is not NULL, then NumEntries should be greater than zero.
    ///< Otherwise ::UR_RESULT_ERROR_INVALID_SIZE
    ///< will be returned.
    ur_device_handle_t *
        phDevices, ///< [out][optional][range(0, NumEntries)] array of handle of devices.
    ///< If NumEntries is less than the number of devices available, then
    ///< platform shall only retrieve that number of devices.
    uint32_t *pNumDevices ///< [out][optional] pointer to the number of devices.
    ///< pNumDevices will be updated with the total number of devices available.
    ) try {
    auto pfnGet = ur_lib::context->urDdiTable.Device.pfnGet;
    if (nullptr == pfnGet) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGet(hPlatform, DeviceType, NumEntries, phDevices, pNumDevices);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about device
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urDeviceGetInfo(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    ur_device_info_t propName,  ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return the info
    ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Device.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hDevice, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes a reference of the device handle indicating it's in use until
///        paired ::urDeviceRelease is called
///
/// @details
///     - Increments the device reference count if `hDevice` is a valid
///       sub-device created by a call to `urDevicePartition`. If `hDevice` is a
///       root level device (e.g. obtained with `urDeviceGet`), the reference
///       count remains unchanged.
///     - It is not valid to use the device handle, which has all of its
///       references released.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clRetainDevice**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
ur_result_t UR_APICALL urDeviceRetain(
    ur_device_handle_t
        hDevice ///< [in] handle of the device to get a reference of.
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Device.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hDevice);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Releases the device handle reference indicating end of its usage
///
/// @details
///     - Decrements the device reference count if `hDevice` is a valid
///       sub-device created by a call to `urDevicePartition`. If `hDevice` is a
///       root level device (e.g. obtained with `urDeviceGet`), the reference
///       count remains unchanged.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clReleaseDevice**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
ur_result_t UR_APICALL urDeviceRelease(
    ur_device_handle_t hDevice ///< [in] handle of the device to release.
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Device.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hDevice);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Partition the device into sub-devices
///
/// @details
///     - Repeated calls to this function with the same inputs will produce the
///       same output in the same order.
///     - The function may be called to request a further partitioning of a
///       sub-device into sub-sub-devices, and so on.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clCreateSubDevices**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pProperties`
///         + `NULL == pProperties->pProperties`
///     - ::UR_RESULT_ERROR_DEVICE_PARTITION_FAILED
///     - ::UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT
ur_result_t UR_APICALL urDevicePartition(
    ur_device_handle_t hDevice, ///< [in] handle of the device to partition.
    const ur_device_partition_properties_t
        *pProperties,    ///< [in] Device partition properties.
    uint32_t NumDevices, ///< [in] the number of sub-devices.
    ur_device_handle_t *
        phSubDevices, ///< [out][optional][range(0, NumDevices)] array of handle of devices.
    ///< If NumDevices is less than the number of sub-devices available, then
    ///< the function shall only retrieve that number of sub-devices.
    uint32_t *
        pNumDevicesRet ///< [out][optional] pointer to the number of sub-devices the device can be
    ///< partitioned into according to the partitioning property.
    ) try {
    auto pfnPartition = ur_lib::context->urDdiTable.Device.pfnPartition;
    if (nullptr == pfnPartition) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnPartition(hDevice, pProperties, NumDevices, phSubDevices,
                        pNumDevicesRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Selects the most appropriate device binary based on runtime
///        information and the IR characteristics.
///
/// @details
///     - The input binaries are various AOT images, and possibly an IL binary
///       for JIT compilation.
///     - The selected binary will be able to be run on the target device.
///     - If no suitable binary can be found then function returns
///       ${X}_INVALID_BINARY.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pBinaries`
///         + `NULL == pSelectedBinary`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NumBinaries == 0`
ur_result_t UR_APICALL urDeviceSelectBinary(
    ur_device_handle_t
        hDevice, ///< [in] handle of the device to select binary for.
    const ur_device_binary_t
        *pBinaries,       ///< [in] the array of binaries to select from.
    uint32_t NumBinaries, ///< [in] the number of binaries passed in ppBinaries.
                          ///< Must greater than or equal to zero otherwise
                          ///< ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t *
        pSelectedBinary ///< [out] the index of the selected binary in the input array of binaries.
    ///< If a suitable binary was not found the function returns ::UR_RESULT_ERROR_INVALID_BINARY.
    ) try {
    auto pfnSelectBinary = ur_lib::context->urDdiTable.Device.pfnSelectBinary;
    if (nullptr == pfnSelectBinary) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSelectBinary(hDevice, pBinaries, NumBinaries, pSelectedBinary);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native device handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeDevice`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urDeviceGetNativeHandle(
    ur_device_handle_t hDevice, ///< [in] handle of the device.
    ur_native_handle_t
        *phNativeDevice ///< [out] a pointer to the native handle of the device.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Device.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hDevice, phNativeDevice);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime device object from native device handle.
///
/// @details
///     - Creates runtime device handle from native driver device handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevice`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urDeviceCreateWithNativeHandle(
    ur_native_handle_t
        hNativeDevice, ///< [in][nocheck] the native handle of the device.
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    const ur_device_native_properties_t *
        pProperties, ///< [in][optional] pointer to native device properties struct.
    ur_device_handle_t
        *phDevice ///< [out] pointer to the handle of the device object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Device.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativeDevice, hPlatform, pProperties,
                                     phDevice);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns synchronized Host and Device global timestamps.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceAndHostTimer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
ur_result_t UR_APICALL urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    uint64_t *
        pDeviceTimestamp, ///< [out][optional] pointer to the Device's global timestamp that
                          ///< correlates with the Host's global timestamp value
    uint64_t *
        pHostTimestamp ///< [out][optional] pointer to the Host's global timestamp that
                       ///< correlates with the Device's global timestamp value
    ) try {
    auto pfnGetGlobalTimestamps =
        ur_lib::context->urDdiTable.Device.pfnGetGlobalTimestamps;
    if (nullptr == pfnGetGlobalTimestamps) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetGlobalTimestamps(hDevice, pDeviceTimestamp, pHostTimestamp);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a context with the given devices.
///
/// @details
///     - All devices should be from the same platform.
///     - Context is used for resource sharing between all the devices
///       associated with it.
///     - Context also serves for resource isolation such that resources do not
///       cross context boundaries.
///     - The returned context is a reference and must be released with a
///       subsequent call to ::urContextRelease.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clCreateContext**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///         + `NULL == phContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_CONTEXT_FLAGS_MASK & pProperties->flags`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
ur_result_t UR_APICALL urContextCreate(
    uint32_t DeviceCount, ///< [in] the number of devices given in phDevices
    const ur_device_handle_t
        *phDevices, ///< [in][range(0, DeviceCount)] array of handle of devices.
    const ur_context_properties_t *
        pProperties, ///< [in][optional] pointer to context creation properties.
    ur_context_handle_t
        *phContext ///< [out] pointer to handle of context object created
    ) try {
    auto pfnCreate = ur_lib::context->urDdiTable.Context.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreate(DeviceCount, phDevices, pProperties, phContext);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes a reference of the context handle indicating it's in use until
///        paired ::urContextRelease is called
///
/// @details
///     - It is not valid to use a context handle, which has all of its
///       references released.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clRetainContext**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
ur_result_t UR_APICALL urContextRetain(
    ur_context_handle_t
        hContext ///< [in] handle of the context to get a reference of.
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Context.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hContext);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Releases the context handle reference indicating end of its usage
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clReleaseContext**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
ur_result_t UR_APICALL urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Context.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hContext);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about context
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clGetContextInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urContextGetInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context
    ur_context_info_t propName,   ///< [in] type of the info to retrieve
    size_t
        propSize, ///< [in] the number of bytes of memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< if propSize is not equal to or greater than the real number of bytes
    ///< needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Context.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hContext, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native context handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeContext`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urContextGetNativeHandle(
    ur_context_handle_t hContext, ///< [in] handle of the context.
    ur_native_handle_t *
        phNativeContext ///< [out] a pointer to the native handle of the context.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Context.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hContext, phNativeContext);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime context object from native context handle.
///
/// @details
///     - Creates runtime context handle from native driver context handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///         + `NULL == phContext`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t
        hNativeContext,  ///< [in][nocheck] the native handle of the context.
    uint32_t numDevices, ///< [in] number of devices associated with the context
    const ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] list of devices associated with the context
    const ur_context_native_properties_t *
        pProperties, ///< [in][optional] pointer to native context properties struct
    ur_context_handle_t *
        phContext ///< [out] pointer to the handle of the context object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Context.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativeContext, numDevices, phDevices,
                                     pProperties, phContext);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Call extended deleter function as callback.
///
/// @details
///     - Calls extended deleter, a user-defined callback to delete context on
///       some platforms.
///     - This is done for performance reasons.
///     - This API might be called directly by an application instead of a
///       runtime backend.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnDeleter`
ur_result_t UR_APICALL urContextSetExtendedDeleter(
    ur_context_handle_t hContext, ///< [in] handle of the context.
    ur_context_extended_deleter_t
        pfnDeleter, ///< [in] Function pointer to extended deleter.
    void *
        pUserData ///< [in][out][optional] pointer to data to be passed to callback.
    ) try {
    auto pfnSetExtendedDeleter =
        ur_lib::context->urDdiTable.Context.pfnSetExtendedDeleter;
    if (nullptr == pfnSetExtendedDeleter) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetExtendedDeleter(hContext, pfnDeleter, pUserData);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create an image object
///
/// @remarks
///   _Analogues_
///     - **clCreateImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///         + `pHost == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pHost != NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    void *pHost,           ///< [in][optional] pointer to the buffer data
    ur_mem_handle_t *phMem ///< [out] pointer to handle of image object created
    ) try {
    auto pfnImageCreate = ur_lib::context->urDdiTable.Mem.pfnImageCreate;
    if (nullptr == pfnImageCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImageCreate(hContext, flags, pImageFormat, pImageDesc, pHost,
                          phMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a memory buffer
///
/// @details
///     - See also ::ur_buffer_channel_properties_t.
///     - See also ::ur_buffer_alloc_location_properties_t.
///
/// @remarks
///   _Analogues_
///     - **clCreateBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phBuffer`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_BUFFER_SIZE
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///         + `pProperties == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pProperties != NULL && pProperties->pHost == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pProperties != NULL && pProperties->pHost != NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    size_t size, ///< [in] size in bytes of the memory object to be allocated
    const ur_buffer_properties_t
        *pProperties, ///< [in][optional] pointer to buffer creation properties
    ur_mem_handle_t
        *phBuffer ///< [out] pointer to handle of the memory buffer created
    ) try {
    auto pfnBufferCreate = ur_lib::context->urDdiTable.Mem.pfnBufferCreate;
    if (nullptr == pfnBufferCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnBufferCreate(hContext, flags, size, pProperties, phBuffer);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference the memory object. Increment the memory object's
///        reference count
///
/// @details
///     - Useful in library function to retain access to the memory object after
///       the caller released the object
///
/// @remarks
///   _Analogues_
///     - **clRetainMemoryObject**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Mem.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the memory object's reference count and delete the object if
///        the reference count becomes zero.
///
/// @remarks
///   _Analogues_
///     - **clReleaseMemoryObject**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Mem.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a sub buffer representing a region in an existing buffer
///
/// @remarks
///   _Analogues_
///     - **clCreateSubBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_FLAGS_MASK & flags`
///         + `::UR_BUFFER_CREATE_TYPE_REGION < bufferCreateType`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pRegion`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_BUFFER_SIZE
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t
        hBuffer,          ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    ur_buffer_create_type_t bufferCreateType, ///< [in] buffer creation type
    const ur_buffer_region_t
        *pRegion, ///< [in] pointer to buffer create region information
    ur_mem_handle_t
        *phMem ///< [out] pointer to the handle of sub buffer created
    ) try {
    auto pfnBufferPartition =
        ur_lib::context->urDdiTable.Mem.pfnBufferPartition;
    if (nullptr == pfnBufferPartition) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion, phMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native mem handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeMem`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t hMem, ///< [in] handle of the mem.
    ur_device_handle_t
        hDevice, ///< [in] handle of the device that the native handle will be resident on.
    ur_native_handle_t
        *phNativeMem ///< [out] a pointer to the native handle of the mem.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Mem.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hMem, hDevice, phNativeMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime buffer memory object from native memory handle.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t
        hNativeMem, ///< [in][nocheck] the native handle to the memory.
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const ur_mem_native_properties_t *
        pProperties, ///< [in][optional] pointer to native memory creation properties.
    ur_mem_handle_t
        *phMem ///< [out] pointer to handle of buffer memory object created.
    ) try {
    auto pfnBufferCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Mem.pfnBufferCreateWithNativeHandle;
    if (nullptr == pfnBufferCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnBufferCreateWithNativeHandle(hNativeMem, hContext, pProperties,
                                           phMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime image memory object from native memory handle.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t
        hNativeMem, ///< [in][nocheck] the native handle to the memory.
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification.
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description.
    const ur_mem_native_properties_t *
        pProperties, ///< [in][optional] pointer to native memory creation properties.
    ur_mem_handle_t
        *phMem ///< [out] pointer to handle of image memory object created.
    ) try {
    auto pfnImageCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Mem.pfnImageCreateWithNativeHandle;
    if (nullptr == pfnImageCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImageCreateWithNativeHandle(hNativeMem, hContext, pImageFormat,
                                          pImageDesc, pProperties, phMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve information about a memory object.
///
/// @details
///     - Query information that is common to all memory objects.
///
/// @remarks
///   _Analogues_
///     - **clGetMemObjectInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_INFO_CONTEXT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urMemGetInfo(
    ur_mem_handle_t
        hMemory,            ///< [in] handle to the memory object being queried.
    ur_mem_info_t propName, ///< [in] type of the info to retrieve.
    size_t
        propSize, ///< [in] the number of bytes of memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is less than the real number of bytes needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Mem.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve information about an image object.
///
/// @details
///     - Query information specific to an image object.
///
/// @remarks
///   _Analogues_
///     - **clGetImageInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_IMAGE_INFO_DEPTH < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urMemImageGetInfo(
    ur_mem_handle_t hMemory, ///< [in] handle to the image object being queried.
    ur_image_info_t propName, ///< [in] type of image info to retrieve.
    size_t
        propSize, ///< [in] the number of bytes of memory pointer to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is less than the real number of bytes needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
    ) try {
    auto pfnImageGetInfo = ur_lib::context->urDdiTable.Mem.pfnImageGetInfo;
    if (nullptr == pfnImageGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImageGetInfo(hMemory, propName, propSize, pPropValue,
                           pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a sampler object in a context
///
/// @details
///     - The props parameter specifies a list of sampler property names and
///       their corresponding values.
///     - The list is terminated with 0. If the list is NULL, default values
///       will be used.
///
/// @remarks
///   _Analogues_
///     - **clCreateSamplerWithProperties**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDesc`
///         + `NULL == phSampler`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT < pDesc->addressingMode`
///         + `::UR_SAMPLER_FILTER_MODE_LINEAR < pDesc->filterMode`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urSamplerCreate(
    ur_context_handle_t hContext,   ///< [in] handle of the context object
    const ur_sampler_desc_t *pDesc, ///< [in] pointer to the sampler description
    ur_sampler_handle_t
        *phSampler ///< [out] pointer to handle of sampler object created
    ) try {
    auto pfnCreate = ur_lib::context->urDdiTable.Sampler.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreate(hContext, pDesc, phSampler);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the sampler object handle. Increment its reference
///        count
///
/// @remarks
///   _Analogues_
///     - **clRetainSampler**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urSamplerRetain(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to get access
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Sampler.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hSampler);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the sampler's reference count and delete the sampler if the
///        reference count becomes zero.
///
/// @remarks
///   _Analogues_
///     - **clReleaseSampler**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urSamplerRelease(
    ur_sampler_handle_t
        hSampler ///< [in] handle of the sampler object to release
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Sampler.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hSampler);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a sampler object
///
/// @remarks
///   _Analogues_
///     - **clGetSamplerInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_SAMPLER_INFO_FILTER_MODE < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urSamplerGetInfo(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler object
    ur_sampler_info_t propName, ///< [in] name of the sampler property to query
    size_t
        propSize, ///< [in] size in bytes of the sampler property value provided
    void *
        pPropValue, ///< [out][typename(propName, propSize)][optional] value of the sampler
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] size in bytes returned in sampler property value
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Sampler.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hSampler, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return sampler native sampler handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability sampler extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeSampler`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler.
    ur_native_handle_t *
        phNativeSampler ///< [out] a pointer to the native handle of the sampler.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Sampler.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hSampler, phNativeSampler);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime sampler object from native sampler handle.
///
/// @details
///     - Creates runtime sampler handle from native driver sampler handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phSampler`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urSamplerCreateWithNativeHandle(
    ur_native_handle_t
        hNativeSampler, ///< [in][nocheck] the native handle of the sampler.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_sampler_native_properties_t *
        pProperties, ///< [in][optional] pointer to native sampler properties struct.
    ur_sampler_handle_t *
        phSampler ///< [out] pointer to the handle of the sampler object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Sampler.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativeSampler, hContext, pProperties,
                                     phSampler);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate host memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_host_desc_t.
///     - See also ::ur_usm_alloc_location_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::UR_DEVICE_INFO_USM_HOST_SUPPORT is false.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urUSMHostAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM host memory object
    ) try {
    auto pfnHostAlloc = ur_lib::context->urDdiTable.USM.pfnHostAlloc;
    if (nullptr == pfnHostAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnHostAlloc(hContext, pUSMDesc, pool, size, ppMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate device memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_device_desc_t.
///     - See also ::ur_usm_alloc_location_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::UR_DEVICE_INFO_USM_HOST_SUPPORT is false.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM device memory object
    ) try {
    auto pfnDeviceAlloc = ur_lib::context->urDdiTable.USM.pfnDeviceAlloc;
    if (nullptr == pfnDeviceAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnDeviceAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate shared memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_host_desc_t.
///     - See also ::ur_usm_device_desc_t.
///     - See also ::ur_usm_alloc_location_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If `UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT` and `UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT` are both false.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t *
        pUSMDesc, ///< [in][optional] Pointer to USM memory allocation descriptor.
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM shared memory object
    ) try {
    auto pfnSharedAlloc = ur_lib::context->urDdiTable.USM.pfnSharedAlloc;
    if (nullptr == pfnSharedAlloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSharedAlloc(hContext, hDevice, pUSMDesc, pool, size, ppMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Free the USM memory object
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
    ) try {
    auto pfnFree = ur_lib::context->urDdiTable.USM.pfnFree;
    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnFree(hContext, pMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get USM memory object allocation information
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_ALLOC_INFO_POOL < propName`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urUSMGetMemAllocInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const void *pMem,             ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t
        propName, ///< [in] the name of the USM allocation property to query
    size_t
        propSize, ///< [in] size in bytes of the USM allocation property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the USM
                    ///< allocation property
    size_t *
        pPropSizeRet ///< [out][optional] bytes returned in USM allocation property
    ) try {
    auto pfnGetMemAllocInfo =
        ur_lib::context->urDdiTable.USM.pfnGetMemAllocInfo;
    if (nullptr == pfnGetMemAllocInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetMemAllocInfo(hContext, pMem, propName, propSize, pPropValue,
                              pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create USM memory pool with desired properties.
///
/// @details
///     - UR can create multiple instances of the pool depending on allocation
///       requests.
///     - See also ::ur_usm_pool_limits_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPoolDesc`
///         + `NULL == ppPool`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_POOL_FLAGS_MASK & pPoolDesc->flags`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urUSMPoolCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_pool_desc_t *
        pPoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *ppPool ///< [out] pointer to USM memory pool
    ) try {
    auto pfnPoolCreate = ur_lib::context->urDdiTable.USM.pfnPoolCreate;
    if (nullptr == pfnPoolCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnPoolCreate(hContext, pPoolDesc, ppPool);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the pool handle. Increment its reference count
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == pPool`
ur_result_t UR_APICALL urUSMPoolRetain(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
    ) try {
    auto pfnPoolRetain = ur_lib::context->urDdiTable.USM.pfnPoolRetain;
    if (nullptr == pfnPoolRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnPoolRetain(pPool);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the pool's reference count and delete the pool if the
///        reference count becomes zero.
///
/// @details
///     - All allocation belonging to the pool must be freed prior to the the
///       reference count becoming zero.
///     - If the pool is deleted, this function returns all its reserved memory
///       to the driver.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == pPool`
ur_result_t UR_APICALL urUSMPoolRelease(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
    ) try {
    auto pfnPoolRelease = ur_lib::context->urDdiTable.USM.pfnPoolRelease;
    if (nullptr == pfnPoolRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnPoolRelease(pPool);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a USM memory pool
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPool`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_POOL_INFO_CONTEXT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urUSMPoolGetInfo(
    ur_usm_pool_handle_t hPool,  ///< [in] handle of the USM memory pool
    ur_usm_pool_info_t propName, ///< [in] name of the pool property to query
    size_t propSize, ///< [in] size in bytes of the pool property value provided
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the pool
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] size in bytes returned in pool property value
    ) try {
    auto pfnPoolGetInfo = ur_lib::context->urDdiTable.USM.pfnPoolGetInfo;
    if (nullptr == pfnPoolGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnPoolGetInfo(hPool, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about the minimum and recommended granularity of
///        physical and virtual memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    ur_device_handle_t
        hDevice, ///< [in][optional] is the device to get the granularity from, if the
    ///< device is null then the granularity is suitable for all devices in context.
    ur_virtual_mem_granularity_info_t
        propName, ///< [in] type of the info to query.
    size_t
        propSize, ///< [in] size in bytes of the memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
    ///< the info. If propSize is less than the real number of bytes needed to
    ///< return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName."
    ) try {
    auto pfnGranularityGetInfo =
        ur_lib::context->urDdiTable.VirtualMem.pfnGranularityGetInfo;
    if (nullptr == pfnGranularityGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGranularityGetInfo(hContext, hDevice, propName, propSize,
                                 pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Reserve a virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppStart`
ur_result_t UR_APICALL urVirtualMemReserve(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const void *
        pStart, ///< [in][optional] pointer to the start of the virtual memory region to
    ///< reserve, specifying a null value causes the implementation to select a
    ///< start address.
    size_t
        size, ///< [in] size in bytes of the virtual address range to reserve.
    void **
        ppStart ///< [out] pointer to the returned address at the start of reserved virtual
                ///< memory range.
    ) try {
    auto pfnReserve = ur_lib::context->urDdiTable.VirtualMem.pfnReserve;
    if (nullptr == pfnReserve) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnReserve(hContext, pStart, size, ppStart);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Free a virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
ur_result_t UR_APICALL urVirtualMemFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const void *
        pStart, ///< [in] pointer to the start of the virtual memory range to free.
    size_t size ///< [in] size in bytes of the virtual memory range to free.
    ) try {
    auto pfnFree = ur_lib::context->urDdiTable.VirtualMem.pfnFree;
    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnFree(hContext, pStart, size);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Map a virtual memory range to a physical memory handle.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hPhysicalMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_ACCESS_FLAGS_MASK & flags`
ur_result_t UR_APICALL urVirtualMemMap(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void
        *pStart, ///< [in] pointer to the start of the virtual memory range.
    size_t size, ///< [in] size in bytes of the virtual memory range to map.
    ur_physical_mem_handle_t
        hPhysicalMem, ///< [in] handle of the physical memory to map pStart to.
    size_t
        offset, ///< [in] offset in bytes into the physical memory to map pStart to.
    ur_virtual_mem_access_flags_t
        flags ///< [in] access flags for the physical memory mapping.
    ) try {
    auto pfnMap = ur_lib::context->urDdiTable.VirtualMem.pfnMap;
    if (nullptr == pfnMap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMap(hContext, pStart, size, hPhysicalMem, offset, flags);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Unmap a virtual memory range previously mapped in a context.
///
/// @details
///     - After a call to this function, the virtual memory range is left in a
///       state ready to be remapped.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
ur_result_t UR_APICALL urVirtualMemUnmap(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void *
        pStart, ///< [in] pointer to the start of the mapped virtual memory range
    size_t size ///< [in] size in bytes of the virtual memory range.
    ) try {
    auto pfnUnmap = ur_lib::context->urDdiTable.VirtualMem.pfnUnmap;
    if (nullptr == pfnUnmap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUnmap(hContext, pStart, size);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the access mode of a mapped virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_ACCESS_FLAGS_MASK & flags`
ur_result_t UR_APICALL urVirtualMemSetAccess(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void
        *pStart, ///< [in] pointer to the start of the virtual memory range.
    size_t size, ///< [in] size in bytes of the virtual memory range.
    ur_virtual_mem_access_flags_t
        flags ///< [in] access flags to set for the mapped virtual memory range.
    ) try {
    auto pfnSetAccess = ur_lib::context->urDdiTable.VirtualMem.pfnSetAccess;
    if (nullptr == pfnSetAccess) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetAccess(hContext, pStart, size, flags);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about a mapped virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_INFO_ACCESS_MODE < propName`
ur_result_t UR_APICALL urVirtualMemGetInfo(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void
        *pStart, ///< [in] pointer to the start of the virtual memory range.
    size_t size, ///< [in] size in bytes of the virtual memory range.
    ur_virtual_mem_info_t propName, ///< [in] type of the info to query.
    size_t
        propSize, ///< [in] size in bytes of the memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
    ///< the info. If propSize is less than the real number of bytes needed to
    ///< return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
    ///< returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName."
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.VirtualMem.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hContext, pStart, size, propName, propSize, pPropValue,
                      pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a physical memory handle that virtual memory can be mapped to.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_PHYSICAL_MEM_FLAGS_MASK & pProperties->flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPhysicalMem`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If size is not a multiple of ::UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM.
ur_result_t UR_APICALL urPhysicalMemCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    ur_device_handle_t hDevice,   ///< [in] handle of the device object.
    size_t
        size, ///< [in] size in bytes of physical memory to allocate, must be a multiple
              ///< of ::UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM.
    const ur_physical_mem_properties_t *
        pProperties, ///< [in][optional] pointer to physical memory creation properties.
    ur_physical_mem_handle_t *
        phPhysicalMem ///< [out] pointer to handle of physical memory object created.
    ) try {
    auto pfnCreate = ur_lib::context->urDdiTable.PhysicalMem.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreate(hContext, hDevice, size, pProperties, phPhysicalMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retain a physical memory handle, increment its reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPhysicalMem`
ur_result_t UR_APICALL urPhysicalMemRetain(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to retain.
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.PhysicalMem.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hPhysicalMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Release a physical memory handle, decrement its reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPhysicalMem`
ur_result_t UR_APICALL urPhysicalMemRelease(
    ur_physical_mem_handle_t
        hPhysicalMem ///< [in] handle of the physical memory object to release.
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.PhysicalMem.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hPhysicalMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a program object from input intermediate language.
///
/// @details
///     - The application may call this function from simultaneous threads.
///
/// @remarks
///   _Analogues_
///     - **clCreateProgramWithIL**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pIL`
///         + `NULL == phProgram`
///         + `NULL != pProperties && pProperties->count > 0 && NULL == pProperties->pMetadatas`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NULL != pProperties && NULL != pProperties->pMetadatas && pProperties->count == 0`
///         + `length == 0`
///     - ::UR_RESULT_ERROR_INVALID_BINARY
///         + If `pIL` is not a valid IL binary for devices in `hContext`.
///     - ::UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE
///         + If devices in `hContext` don't have the capability to compile an IL binary at runtime.
ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    const void *pIL,              ///< [in] pointer to IL binary.
    size_t length,                ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t *
        pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
    ) try {
    auto pfnCreateWithIL = ur_lib::context->urDdiTable.Program.pfnCreateWithIL;
    if (nullptr == pfnCreateWithIL) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithIL(hContext, pIL, length, pProperties, phProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a program object from device native binary.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point, `phProgram` will
///       contain a binary of type ::UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT or
///       ::UR_PROGRAM_BINARY_TYPE_LIBRARY for `hDevice`.
///
/// @remarks
///   _Analogues_
///     - **clCreateProgramWithBinary**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pBinary`
///         + `NULL == phProgram`
///         + `NULL != pProperties && pProperties->count > 0 && NULL == pProperties->pMetadatas`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NULL != pProperties && NULL != pProperties->pMetadatas && pProperties->count == 0`
///     - ::UR_RESULT_ERROR_INVALID_NATIVE_BINARY
///         + If `pBinary` isn't a valid binary for `hDevice.`
ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    ur_device_handle_t
        hDevice,            ///< [in] handle to device associated with binary.
    size_t size,            ///< [in] size in bytes.
    const uint8_t *pBinary, ///< [in] pointer to binary.
    const ur_program_properties_t *
        pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of Program object created.
    ) try {
    auto pfnCreateWithBinary =
        ur_lib::context->urDdiTable.Program.pfnCreateWithBinary;
    if (nullptr == pfnCreateWithBinary) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithBinary(hContext, hDevice, size, pBinary, pProperties,
                               phProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one program, negates need for the
///        linking step.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point, the program passed
///       will contain a binary of the ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type
///       for each device in `hContext`.
///
/// @remarks
///   _Analogues_
///     - **clBuildProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred when building `hProgram`.
ur_result_t UR_APICALL urProgramBuild(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
    ) try {
    auto pfnBuild = ur_lib::context->urDdiTable.Program.pfnBuild;
    if (nullptr == pfnBuild) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnBuild(hContext, hProgram, pOptions);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point `hProgram` will
///       contain a binary of the ::UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT type
///       for each device in `hContext`.
///
/// @remarks
///   _Analogues_
///     - **clCompileProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred while compiling `hProgram`.
ur_result_t UR_APICALL urProgramCompile(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t
        hProgram, ///< [in][out] handle of the program to compile.
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
    ) try {
    auto pfnCompile = ur_lib::context->urDdiTable.Program.pfnCompile;
    if (nullptr == pfnCompile) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCompile(hContext, hProgram, pOptions);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point the program returned
///       in `phProgram` will contain a binary of the
///       ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type for each device in
///       `hContext`.
///
/// @remarks
///   _Analogues_
///     - **clLinkProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPrograms`
///         + `NULL == phProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If one of the programs in `phPrograms` isn't a valid program object.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
///     - ::UR_RESULT_ERROR_PROGRAM_LINK_FAILURE
///         + If an error occurred while linking `phPrograms`.
ur_result_t UR_APICALL urProgramLink(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    uint32_t count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *
        phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *
        pOptions, ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
    ) try {
    auto pfnLink = ur_lib::context->urDdiTable.Program.pfnLink;
    if (nullptr == pfnLink) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnLink(hContext, count, phPrograms, pOptions, phProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the Program object.
///
/// @details
///     - Get a reference to the Program object handle. Increment its reference
///       count
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clRetainProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
ur_result_t UR_APICALL urProgramRetain(
    ur_program_handle_t hProgram ///< [in] handle for the Program to retain
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Program.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Release Program.
///
/// @details
///     - Decrement reference count and destroy the Program if reference count
///       becomes zero.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clReleaseProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
ur_result_t UR_APICALL urProgramRelease(
    ur_program_handle_t hProgram ///< [in] handle for the Program to release
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Program.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves a device function pointer to a user-defined function.
///
/// @details
///     - Retrieves a pointer to the functions with the given name and defined
///       in the given program.
///     - ::UR_RESULT_ERROR_INVALID_FUNCTION_NAME is returned if the function
///       can not be obtained.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceFunctionPointerINTEL**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pFunctionName`
///         + `NULL == ppFunctionPointer`
ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t
        hDevice, ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t
        hProgram, ///< [in] handle of the program to search for function in.
    ///< The program must already be built to the specified device, or
    ///< otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char *
        pFunctionName, ///< [in] A null-terminates string denoting the mangled function name.
    void **
        ppFunctionPointer ///< [out] Returns the pointer to the function if it is found in the program.
    ) try {
    auto pfnGetFunctionPointer =
        ur_lib::context->urDdiTable.Program.pfnGetFunctionPointer;
    if (nullptr == pfnGetFunctionPointer) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetFunctionPointer(hDevice, hProgram, pFunctionName,
                                 ppFunctionPointer);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a Program object
///
/// @remarks
///   _Analogues_
///     - **clGetProgramInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_INFO_KERNEL_NAMES < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urProgramGetInfo(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    ur_program_info_t propName, ///< [in] name of the Program property to query
    size_t propSize,            ///< [in] the size of the Program property.
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] array of bytes of
                    ///< holding the program info property.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Program.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hProgram, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query build information about a Program object for a Device
///
/// @remarks
///   _Analogues_
///     - **clGetProgramBuildInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_BUILD_INFO_BINARY_TYPE < propName`
ur_result_t UR_APICALL urProgramGetBuildInfo(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    ur_device_handle_t hDevice,   ///< [in] handle of the Device object
    ur_program_build_info_t
        propName,    ///< [in] name of the Program build info to query
    size_t propSize, ///< [in] size of the Program build info property.
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] value of the Program
                    ///< build property.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE
    ///< error is returned and pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
    ) try {
    auto pfnGetBuildInfo = ur_lib::context->urDdiTable.Program.pfnGetBuildInfo;
    if (nullptr == pfnGetBuildInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetBuildInfo(hProgram, hDevice, propName, propSize, pPropValue,
                           pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set an array of specialization constants on a Program.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///     - `hProgram` must have been created with the ::urProgramCreateWithIL
///       entry point.
///     - Any spec constants set with this entry point will apply only to
///       subsequent calls to ::urProgramBuild or ::urProgramCompile.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSpecConstants`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    uint32_t count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *
        pSpecConstants ///< [in][range(0, count)] array of specialization constant value
                       ///< descriptions
    ) try {
    auto pfnSetSpecializationConstants =
        ur_lib::context->urDdiTable.Program.pfnSetSpecializationConstants;
    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetSpecializationConstants(hProgram, count, pSpecConstants);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return program native program handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability program extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeProgram`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ///< [in] handle of the program.
    ur_native_handle_t *
        phNativeProgram ///< [out] a pointer to the native handle of the program.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Program.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hProgram, phNativeProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime program object from native program handle.
///
/// @details
///     - Creates runtime program handle from native driver program handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phProgram`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t
        hNativeProgram, ///< [in][nocheck] the native handle of the program.
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    const ur_program_native_properties_t *
        pProperties, ///< [in][optional] pointer to native program properties struct.
    ur_program_handle_t *
        phProgram ///< [out] pointer to the handle of the program object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Program.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativeProgram, hContext, pProperties,
                                     phProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create kernel object from a program.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pKernelName`
///         + `NULL == phKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_NAME
///         + If `pKernelName` wasn't found in `hProgram`.
ur_result_t UR_APICALL urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to handle of kernel object created.
    ) try {
    auto pfnCreate = ur_lib::context->urDdiTable.Kernel.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreate(hProgram, pKernelName, phKernel);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set kernel argument to a value.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t argSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *pProperties, ///< [in][optional] pointer to value properties.
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
    ) try {
    auto pfnSetArgValue = ur_lib::context->urDdiTable.Kernel.pfnSetArgValue;
    if (nullptr == pfnSetArgValue) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetArgValue(hKernel, argIndex, argSize, pProperties, pArgValue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set kernel argument to a local buffer.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t
        argSize, ///< [in] size of the local buffer to be allocated by the runtime
    const ur_kernel_arg_local_properties_t
        *pProperties ///< [in][optional] pointer to local buffer properties.
    ) try {
    auto pfnSetArgLocal = ur_lib::context->urDdiTable.Kernel.pfnSetArgLocal;
    if (nullptr == pfnSetArgLocal) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetArgLocal(hKernel, argIndex, argSize, pProperties);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a Kernel object
///
/// @remarks
///   _Analogues_
///     - **clGetKernelInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_INFO_NUM_REGS < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urKernelGetInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_kernel_info_t propName,  ///< [in] name of the Kernel property to query
    size_t propSize,            ///< [in] the size of the Kernel property value.
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] array of bytes
                    ///< holding the kernel info property.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Kernel.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hKernel, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query work Group information about a Kernel object
///
/// @remarks
///   _Analogues_
///     - **clGetKernelWorkGroupInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE < propName`
ur_result_t UR_APICALL urKernelGetGroupInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice, ///< [in] handle of the Device object
    ur_kernel_group_info_t
        propName,    ///< [in] name of the work Group property to query
    size_t propSize, ///< [in] size of the Kernel Work Group property value
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] value of the Kernel
                    ///< Work Group property.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
    ) try {
    auto pfnGetGroupInfo = ur_lib::context->urDdiTable.Kernel.pfnGetGroupInfo;
    if (nullptr == pfnGetGroupInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetGroupInfo(hKernel, hDevice, propName, propSize, pPropValue,
                           pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query SubGroup information about a Kernel object
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL < propName`
ur_result_t UR_APICALL urKernelGetSubGroupInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice, ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t
        propName,    ///< [in] name of the SubGroup property to query
    size_t propSize, ///< [in] size of the Kernel SubGroup property value
    void *
        pPropValue, ///< [in,out][optional][typename(propName, propSize)] value of the Kernel
                    ///< SubGroup property.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of data being
                     ///< queried by propName.
    ) try {
    auto pfnGetSubGroupInfo =
        ur_lib::context->urDdiTable.Kernel.pfnGetSubGroupInfo;
    if (nullptr == pfnGetSubGroupInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetSubGroupInfo(hKernel, hDevice, propName, propSize, pPropValue,
                              pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the Kernel object.
///
/// @details
///     - Get a reference to the Kernel object handle. Increment its reference
///       count
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clRetainKernel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
ur_result_t UR_APICALL urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Kernel.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hKernel);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Release Kernel.
///
/// @details
///     - Decrement reference count and destroy the Kernel if reference count
///       becomes zero.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clReleaseKernel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
ur_result_t UR_APICALL urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Kernel.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hKernel);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a USM pointer as the argument value of a Kernel.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clSetKernelArgSVMPointer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *pProperties, ///< [in][optional] pointer to USM pointer properties.
    const void *
        pArgValue ///< [in][optional] USM pointer to memory location holding the argument
                  ///< value. If null then argument value is considered null.
    ) try {
    auto pfnSetArgPointer = ur_lib::context->urDdiTable.Kernel.pfnSetArgPointer;
    if (nullptr == pfnSetArgPointer) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetArgPointer(hKernel, argIndex, pProperties, pArgValue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set additional Kernel execution attributes.
///
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clSetKernelExecInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_EXEC_INFO_CACHE_CONFIG < propName`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPropValue`
ur_result_t UR_APICALL urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,     ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName, ///< [in] name of the execution attribute
    size_t propSize,                ///< [in] size in byte the attribute value
    const ur_kernel_exec_info_properties_t
        *pProperties, ///< [in][optional] pointer to execution info properties.
    const void *
        pPropValue ///< [in][typename(propName, propSize)] pointer to memory location holding
                   ///< the property value.
    ) try {
    auto pfnSetExecInfo = ur_lib::context->urDdiTable.Kernel.pfnSetExecInfo;
    if (nullptr == pfnSetExecInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetExecInfo(hKernel, propName, propSize, pProperties, pPropValue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Sampler object as the argument value of a Kernel.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_sampler_properties_t
        *pProperties, ///< [in][optional] pointer to sampler properties.
    ur_sampler_handle_t hArgValue ///< [in] handle of Sampler object.
    ) try {
    auto pfnSetArgSampler = ur_lib::context->urDdiTable.Kernel.pfnSetArgSampler;
    if (nullptr == pfnSetArgSampler) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetArgSampler(hKernel, argIndex, pProperties, hArgValue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Memory object as the argument value of a Kernel.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_MEM_FLAGS_MASK & pProperties->memoryAccess`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_mem_obj_properties_t
        *pProperties, ///< [in][optional] pointer to Memory object properties.
    ur_mem_handle_t hArgValue ///< [in][optional] handle of Memory object.
    ) try {
    auto pfnSetArgMemObj = ur_lib::context->urDdiTable.Kernel.pfnSetArgMemObj;
    if (nullptr == pfnSetArgMemObj) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set an array of specialization constants on a Kernel.
///
/// @details
///     - This entry point is optional, the application should query for support
///       with device query ::UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS
///       passed to ::urDeviceGetInfo.
///     - Adapters which are capable of setting specialization constants
///       immediately prior to ::urEnqueueKernelLaunch with low overhead should
///       implement this entry point.
///     - Otherwise, if setting specialization constants late requires
///       recompiling or linking a program, adapters should not implement this
///       entry point.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSpecConstants`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If ::UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS query is false
ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *
        pSpecConstants ///< [in] array of specialization constant value descriptions
    ) try {
    auto pfnSetSpecializationConstants =
        ur_lib::context->urDdiTable.Kernel.pfnSetSpecializationConstants;
    if (nullptr == pfnSetSpecializationConstants) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetSpecializationConstants(hKernel, count, pSpecConstants);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native kernel handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeKernel`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel.
    ur_native_handle_t
        *phNativeKernel ///< [out] a pointer to the native handle of the kernel.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Kernel.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hKernel, phNativeKernel);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime kernel object from native kernel handle.
///
/// @details
///     - Creates runtime kernel handle from native driver kernel handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phKernel`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t
        hNativeKernel, ///< [in][nocheck] the native handle of the kernel.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_program_handle_t
        hProgram, ///< [in] handle of the program associated with the kernel
    const ur_kernel_native_properties_t *
        pProperties, ///< [in][optional] pointer to native kernel properties struct
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to the handle of the kernel object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Kernel.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativeKernel, hContext, hProgram,
                                     pProperties, phKernel);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a command queue
///
/// @remarks
///   _Analogues_
///     - **clGetCommandQueueInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_QUEUE_INFO_EMPTY < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urQueueGetInfo(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_queue_info_t propName, ///< [in] name of the queue property to query
    size_t
        propSize, ///< [in] size in bytes of the queue property value provided
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the queue
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] size in bytes returned in queue property value
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Queue.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hQueue, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a command queue for a device in a context
///
/// @details
///     - See also ::ur_queue_index_properties_t.
///
/// @remarks
///   _Analogues_
///     - **clCreateCommandQueueWithProperties**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_QUEUE_FLAGS_MASK & pProperties->flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phQueue`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES
///         + `pProperties != NULL && pProperties->flags & UR_QUEUE_FLAG_PRIORITY_HIGH && pProperties->flags & UR_QUEUE_FLAG_PRIORITY_LOW`
///         + `pProperties != NULL && pProperties->flags & UR_QUEUE_FLAG_SUBMISSION_BATCHED && pProperties->flags & UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_queue_properties_t
        *pProperties, ///< [in][optional] pointer to queue creation properties.
    ur_queue_handle_t
        *phQueue ///< [out] pointer to handle of queue object created
    ) try {
    auto pfnCreate = ur_lib::context->urDdiTable.Queue.pfnCreate;
    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreate(hContext, hDevice, pProperties, phQueue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the command queue handle. Increment the command
///        queue's reference count
///
/// @details
///     - Useful in library function to retain access to the command queue after
///       the caller released the queue.
///
/// @remarks
///   _Analogues_
///     - **clRetainCommandQueue**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urQueueRetain(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to get access
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Queue.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hQueue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the command queue's reference count and delete the command
///        queue if the reference count becomes zero.
///
/// @details
///     - After the command queue reference count becomes zero and all queued
///       commands in the queue have finished, the queue is deleted.
///     - It also performs an implicit flush to issue all previously queued
///       commands in the queue.
///
/// @remarks
///   _Analogues_
///     - **clReleaseCommandQueue**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urQueueRelease(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to release
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Queue.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hQueue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return queue native queue handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability queue extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeQueue`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue.
    ur_queue_native_desc_t
        *pDesc, ///< [in][optional] pointer to native descriptor
    ur_native_handle_t
        *phNativeQueue ///< [out] a pointer to the native handle of the queue.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Queue.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hQueue, pDesc, phNativeQueue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime queue object from native queue handle.
///
/// @details
///     - Creates runtime queue handle from native driver queue handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phQueue`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t
        hNativeQueue, ///< [in][nocheck] the native handle of the queue.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_queue_native_properties_t *
        pProperties, ///< [in][optional] pointer to native queue properties struct
    ur_queue_handle_t
        *phQueue ///< [out] pointer to the handle of the queue object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Queue.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativeQueue, hContext, hDevice,
                                     pProperties, phQueue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Blocks until all previously issued commands to the command queue are
///        finished.
///
/// @details
///     - Blocks until all previously issued commands to the command queue are
///       issued and completed.
///     - ::urQueueFinish does not return until all enqueued commands have been
///       processed and finished.
///     - ::urQueueFinish acts as a synchronization point.
///
/// @remarks
///   _Analogues_
///     - **clFinish**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urQueueFinish(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be finished.
    ) try {
    auto pfnFinish = ur_lib::context->urDdiTable.Queue.pfnFinish;
    if (nullptr == pfnFinish) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnFinish(hQueue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Issues all previously enqueued commands in a command queue to the
///        device.
///
/// @details
///     - Guarantees that all enqueued commands will be issued to the
///       appropriate device.
///     - There is no guarantee that they will be completed after ::urQueueFlush
///       returns.
///
/// @remarks
///   _Analogues_
///     - **clFlush**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urQueueFlush(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be flushed.
    ) try {
    auto pfnFlush = ur_lib::context->urDdiTable.Queue.pfnFlush;
    if (nullptr == pfnFlush) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnFlush(hQueue);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get event object information
///
/// @remarks
///   _Analogues_
///     - **clGetEventInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EVENT_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urEventGetInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_event_info_t propName, ///< [in] the name of the event property to query
    size_t propSize, ///< [in] size in bytes of the event property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the event
                    ///< property
    size_t *pPropSizeRet ///< [out][optional] bytes returned in event property
    ) try {
    auto pfnGetInfo = ur_lib::context->urDdiTable.Event.pfnGetInfo;
    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfo(hEvent, propName, propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get profiling information for the command associated with an event
///        object
///
/// @remarks
///   _Analogues_
///     - **clGetEventProfilingInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROFILING_INFO_COMMAND_COMPLETE < propName`
///     - ::UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE
///         + If `hEvent`s associated queue was not created with `UR_QUEUE_FLAG_PROFILING_ENABLE`.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pPropValue && propSize == 0`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_profiling_info_t
        propName,    ///< [in] the name of the profiling property to query
    size_t propSize, ///< [in] size in bytes of the profiling property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the profiling
                    ///< property
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes returned in
                     ///< propValue
    ) try {
    auto pfnGetProfilingInfo =
        ur_lib::context->urDdiTable.Event.pfnGetProfilingInfo;
    if (nullptr == pfnGetProfilingInfo) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetProfilingInfo(hEvent, propName, propSize, pPropValue,
                               pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for a list of events to finish.
///
/// @remarks
///   _Analogues_
///     - **clWaitForEvent**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEventWaitList`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `numEvents == 0`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEventWait(
    uint32_t numEvents, ///< [in] number of events in the event list
    const ur_event_handle_t *
        phEventWaitList ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                        ///< completion
    ) try {
    auto pfnWait = ur_lib::context->urDdiTable.Event.pfnWait;
    if (nullptr == pfnWait) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnWait(numEvents, phEventWaitList);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to an event handle. Increment the event object's
///        reference count.
///
/// @remarks
///   _Analogues_
///     - **clRetainEvent**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urEventRetain(
    ur_event_handle_t hEvent ///< [in] handle of the event object
    ) try {
    auto pfnRetain = ur_lib::context->urDdiTable.Event.pfnRetain;
    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetain(hEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the event object's reference count and delete the event
///        object if the reference count becomes zero.
///
/// @remarks
///   _Analogues_
///     - **clReleaseEvent**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urEventRelease(
    ur_event_handle_t hEvent ///< [in] handle of the event object
    ) try {
    auto pfnRelease = ur_lib::context->urDdiTable.Event.pfnRelease;
    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRelease(hEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native event handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeEvent`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ///< [in] handle of the event.
    ur_native_handle_t
        *phNativeEvent ///< [out] a pointer to the native handle of the event.
    ) try {
    auto pfnGetNativeHandle =
        ur_lib::context->urDdiTable.Event.pfnGetNativeHandle;
    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetNativeHandle(hEvent, phNativeEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime event object from native event handle.
///
/// @details
///     - Creates runtime event handle from native driver event handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEvent`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t
        hNativeEvent, ///< [in][nocheck] the native handle of the event.
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_event_native_properties_t *
        pProperties, ///< [in][optional] pointer to native event properties struct
    ur_event_handle_t
        *phEvent ///< [out] pointer to the handle of the event object created.
    ) try {
    auto pfnCreateWithNativeHandle =
        ur_lib::context->urDdiTable.Event.pfnCreateWithNativeHandle;
    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateWithNativeHandle(hNativeEvent, hContext, pProperties,
                                     phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Register a user callback function for a specific command execution
///        status.
///
/// @details
///     - The registered callback function will be called when the execution
///       status of command associated with event changes to an execution status
///       equal to or past the status specified by command_exec_status.
///     - `execStatus` must not be `UR_EXECUTION_INFO_QUEUED` as this is the
///       initial state of all events.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXECUTION_INFO_QUEUED < execStatus`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnNotify`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + `execStatus == UR_EXECUTION_INFO_QUEUED`
ur_result_t UR_APICALL urEventSetCallback(
    ur_event_handle_t hEvent,       ///< [in] handle of the event object
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    ur_event_callback_t pfnNotify,  ///< [in] execution status of the event
    void *
        pUserData ///< [in][out][optional] pointer to data to be passed to callback.
    ) try {
    auto pfnSetCallback = ur_lib::context->urDdiTable.Event.pfnSetCallback;
    if (nullptr == pfnSetCallback) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSetCallback(hEvent, execStatus, pfnNotify, pUserData);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to execute a kernel
///
/// @remarks
///   _Analogues_
///     - **clEnqueueNDRangeKernel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGlobalWorkOffset`
///         + `NULL == pGlobalWorkSize`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue,   ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t
        workDim, ///< [in] number of dimensions, from 1 to 3, to specify the global and
                 ///< work-group work-items
    const size_t *
        pGlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< offset used to calculate the global ID of a work-item
    const size_t *
        pGlobalWorkSize, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< number of global work-items in workDim that will execute the kernel
    ///< function
    const size_t *
        pLocalWorkSize, ///< [in][optional] pointer to an array of workDim unsigned values that
    ///< specify the number of local work-items forming a work-group that will
    ///< execute the kernel function.
    ///< If nullptr, the runtime implementation will choose the work-group
    ///< size.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
    ) try {
    auto pfnKernelLaunch = ur_lib::context->urDdiTable.Enqueue.pfnKernelLaunch;
    if (nullptr == pfnKernelLaunch) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                           pGlobalWorkSize, pLocalWorkSize, numEventsInWaitList,
                           phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command which waits a list of events to complete before it
///        completes
///
/// @details
///     - If the event list is empty, it waits for all previously enqueued
///       commands to complete.
///     - It returns an event which can be waited on.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueMarkerWithWaitList**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue,     ///< [in] handle of the queue object
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnEventsWait = ur_lib::context->urDdiTable.Enqueue.pfnEventsWait;
    if (nullptr == pfnEventsWait) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnEventsWait(hQueue, numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a barrier command which waits a list of events to complete
///        before it completes
///
/// @details
///     - If the event list is empty, it waits for all previously enqueued
///       commands to complete.
///     - It blocks command execution - any following commands enqueued after it
///       do not execute until it completes.
///     - It returns an event which can be waited on.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueBarrierWithWaitList**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue,     ///< [in] handle of the queue object
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnEventsWaitWithBarrier =
        ur_lib::context->urDdiTable.Enqueue.pfnEventsWaitWithBarrier;
    if (nullptr == pfnEventsWaitWithBarrier) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                    phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read from a buffer object to host memory
///
/// @details
///     - Input parameter blockingRead indicates if the read is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueReadBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being read
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemBufferRead =
        ur_lib::context->urDdiTable.Enqueue.pfnMemBufferRead;
    if (nullptr == pfnMemBufferRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size, pDst,
                            numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write into a buffer object from host memory
///
/// @details
///     - Input parameter blockingWrite indicates if the write is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueWriteBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being written
    const void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemBufferWrite =
        ur_lib::context->urDdiTable.Enqueue.pfnMemBufferWrite;
    if (nullptr == pfnMemBufferWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size, pSrc,
                             numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read a 2D or 3D rectangular region from a buffer
///        object to host memory
///
/// @details
///     - Input parameter blockingRead indicates if the read is blocking or
///       non-blocking.
///     - The buffer and host 2D or 3D rectangular regions can have different
///       shapes.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueReadBufferRect**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.width == 0`
///         + `bufferRowPitch != 0 && bufferRowPitch < region.width`
///         + `hostRowPitch != 0 && hostRowPitch < region.width`
///         + `bufferSlicePitch != 0 && bufferSlicePitch < region.height * (bufferRowPitch != 0 ? bufferRowPitch : region.width)`
///         + `bufferSlicePitch != 0 && bufferSlicePitch % (bufferRowPitch != 0 ? bufferRowPitch : region.width) != 0`
///         + `hostSlicePitch != 0 && hostSlicePitch < region.height * (hostRowPitch != 0 ? hostRowPitch : region.width)`
///         + `hostSlicePitch != 0 && hostSlicePitch % (hostRowPitch != 0 ? hostRowPitch : region.width) != 0`
///         + If the combination of `bufferOrigin`, `region`, `bufferRowPitch`, and `bufferSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed by
                      ///< dst
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed by dst
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemBufferReadRect =
        ur_lib::context->urDdiTable.Enqueue.pfnMemBufferReadRect;
    if (nullptr == pfnMemBufferReadRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferReadRect(
        hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
        numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write a 2D or 3D rectangular region in a buffer
///        object from host memory
///
/// @details
///     - Input parameter blockingWrite indicates if the write is blocking or
///       non-blocking.
///     - The buffer and host 2D or 3D rectangular regions can have different
///       shapes.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueWriteBufferRect**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.width == 0`
///         + `bufferRowPitch != 0 && bufferRowPitch < region.width`
///         + `hostRowPitch != 0 && hostRowPitch < region.width`
///         + `bufferSlicePitch != 0 && bufferSlicePitch < region.height * (bufferRowPitch != 0 ? bufferRowPitch : region.width)`
///         + `bufferSlicePitch != 0 && bufferSlicePitch % (bufferRowPitch != 0 ? bufferRowPitch : region.width) != 0`
///         + `hostSlicePitch != 0 && hostSlicePitch < region.height * (hostRowPitch != 0 ? hostRowPitch : region.width)`
///         + `hostSlicePitch != 0 && hostSlicePitch % (hostRowPitch != 0 ? hostRowPitch : region.width) != 0`
///         + If the combination of `bufferOrigin`, `region`, `bufferRowPitch`, and `bufferSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being
                          ///< written
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed by
                      ///< src
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed by src
    void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemBufferWriteRect =
        ur_lib::context->urDdiTable.Enqueue.pfnMemBufferWriteRect;
    if (nullptr == pfnMemBufferWriteRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferWriteRect(
        hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
        numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy from a buffer object to another
///
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBufferSrc`
///         + `NULL == hBufferDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `srcOffset + size` results in an out-of-bounds access.
///         + If `dstOffset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBufferSrc, ///< [in][bounds(srcOffset, size)] handle of the src buffer object
    ur_mem_handle_t
        hBufferDst, ///< [in][bounds(dstOffset, size)] handle of the dest buffer object
    size_t srcOffset, ///< [in] offset into hBufferSrc to begin copying from
    size_t dstOffset, ///< [in] offset info hBufferDst to begin copying into
    size_t size,      ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemBufferCopy =
        ur_lib::context->urDdiTable.Enqueue.pfnMemBufferCopy;
    if (nullptr == pfnMemBufferCopy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset,
                            dstOffset, size, numEventsInWaitList,
                            phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy a 2D or 3D rectangular region from one
///        buffer object to another
///
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyBufferRect**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBufferSrc`
///         + `NULL == hBufferDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///         + `srcRowPitch != 0 && srcRowPitch < region.width`
///         + `dstRowPitch != 0 && dstRowPitch < region.width`
///         + `srcSlicePitch != 0 && srcSlicePitch < region.height * (srcRowPitch != 0 ? srcRowPitch : region.width)`
///         + `srcSlicePitch != 0 && srcSlicePitch % (srcRowPitch != 0 ? srcRowPitch : region.width) != 0`
///         + `dstSlicePitch != 0 && dstSlicePitch < region.height * (dstRowPitch != 0 ? dstRowPitch : region.width)`
///         + `dstSlicePitch != 0 && dstSlicePitch % (dstRowPitch != 0 ? dstRowPitch : region.width) != 0`
///         + If the combination of `srcOrigin`, `region`, `srcRowPitch`, and `srcSlicePitch` results in an out-of-bounds access.
///         + If the combination of `dstOrigin`, `region`, `dstRowPitch`, and `dstSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBufferSrc, ///< [in][bounds(srcOrigin, region)] handle of the source buffer object
    ur_mem_handle_t
        hBufferDst, ///< [in][bounds(dstOrigin, region)] handle of the dest buffer object
    ur_rect_offset_t srcOrigin, ///< [in] 3D offset in the source buffer
    ur_rect_offset_t dstOrigin, ///< [in] 3D offset in the destination buffer
    ur_rect_region_t
        region, ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t
        srcRowPitch, ///< [in] length of each row in bytes in the source buffer object
    size_t
        srcSlicePitch, ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t
        dstRowPitch, ///< [in] length of each row in bytes in the destination buffer object
    size_t
        dstSlicePitch, ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemBufferCopyRect =
        ur_lib::context->urDdiTable.Enqueue.pfnMemBufferCopyRect;
    if (nullptr == pfnMemBufferCopyRect) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferCopyRect(hQueue, hBufferSrc, hBufferDst, srcOrigin,
                                dstOrigin, region, srcRowPitch, srcSlicePitch,
                                dstRowPitch, dstSlicePitch, numEventsInWaitList,
                                phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill a buffer object with a pattern of a given
///        size
///
/// @remarks
///   _Analogues_
///     - **clEnqueueFillBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `patternSize == 0 || size == 0`
///         + `patternSize > size`
///         + `(patternSize & (patternSize - 1)) != 0`
///         + `size % patternSize != 0`
///         + `offset % patternSize != 0`
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    const void *pPattern, ///< [in] pointer to the fill pattern
    size_t patternSize,   ///< [in] size in bytes of the pattern
    size_t offset,        ///< [in] offset into the buffer
    size_t size, ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemBufferFill =
        ur_lib::context->urDdiTable.Enqueue.pfnMemBufferFill;
    if (nullptr == pfnMemBufferFill) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize, offset,
                            size, numEventsInWaitList, phEventWaitList,
                            phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read from an image or image array object to host
///        memory
///
/// @details
///     - Input parameter blockingRead indicates if the read is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueReadImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hImage, ///< [in][bounds(origin, region)] handle of the image object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t
        origin, ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t
        region, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                ///< image
    size_t rowPitch,   ///< [in] length of each row in bytes
    size_t slicePitch, ///< [in] length of each 2D slice of the 3D image
    void *pDst, ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemImageRead = ur_lib::context->urDdiTable.Enqueue.pfnMemImageRead;
    if (nullptr == pfnMemImageRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemImageRead(hQueue, hImage, blockingRead, origin, region,
                           rowPitch, slicePitch, pDst, numEventsInWaitList,
                           phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write an image or image array object from host
///        memory
///
/// @details
///     - Input parameter blockingWrite indicates if the write is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueWriteImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hImage, ///< [in][bounds(origin, region)] handle of the image object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t
        origin, ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t
        region, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                ///< image
    size_t rowPitch,   ///< [in] length of each row in bytes
    size_t slicePitch, ///< [in] length of each 2D slice of the 3D image
    void *pSrc, ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemImageWrite =
        ur_lib::context->urDdiTable.Enqueue.pfnMemImageWrite;
    if (nullptr == pfnMemImageWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemImageWrite(hQueue, hImage, blockingWrite, origin, region,
                            rowPitch, slicePitch, pSrc, numEventsInWaitList,
                            phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy from an image object to another
///
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImageSrc`
///         + `NULL == hImageDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hImageSrc, ///< [in][bounds(srcOrigin, region)] handle of the src image object
    ur_mem_handle_t
        hImageDst, ///< [in][bounds(dstOrigin, region)] handle of the dest image object
    ur_rect_offset_t
        srcOrigin, ///< [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
                   ///< image
    ur_rect_offset_t
        dstOrigin, ///< [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
                   ///< or 3D image
    ur_rect_region_t
        region, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                ///< image
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemImageCopy = ur_lib::context->urDdiTable.Enqueue.pfnMemImageCopy;
    if (nullptr == pfnMemImageCopy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemImageCopy(hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin,
                           region, numEventsInWaitList, phEventWaitList,
                           phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to map a region of the buffer object into the host
///        address space and return a pointer to the mapped region
///
/// @details
///     - Input parameter blockingMap indicates if the map is blocking or
///       non-blocking.
///     - Currently, no direct support in Level Zero. Implemented as a shared
///       allocation followed by copying on discrete GPU
///     - TODO: add a driver function in Level Zero?
///
/// @remarks
///   _Analogues_
///     - **clEnqueueMapBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MAP_FLAGS_MASK & mapFlags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppRetMap`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingMap, ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t mapFlags, ///< [in] flags for read, write, readwrite mapping
    size_t offset, ///< [in] offset in bytes of the buffer region being mapped
    size_t size,   ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent, ///< [out][optional] return an event object that identifies this particular
                 ///< command instance.
    void **ppRetMap ///< [out] return mapped pointer.  TODO: move it before
                    ///< numEventsInWaitList?
    ) try {
    auto pfnMemBufferMap = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferMap;
    if (nullptr == pfnMemBufferMap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags, offset, size,
                           numEventsInWaitList, phEventWaitList, phEvent,
                           ppRetMap);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to unmap a previously mapped region of a memory
///        object
///
/// @remarks
///   _Analogues_
///     - **clEnqueueUnmapMemObject**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMappedPtr`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hMem,         ///< [in] handle of the memory (buffer or image) object
    void *pMappedPtr, ///< [in] mapped host address
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnMemUnmap = ur_lib::context->urDdiTable.Enqueue.pfnMemUnmap;
    if (nullptr == pfnMemUnmap) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList,
                       phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `patternSize == 0 || size == 0`
///         + `patternSize > size`
///         + `(patternSize & (patternSize - 1)) != 0`
///         + `size % patternSize != 0`
///         + If `size` is higher than the allocation size of `ptr`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    void *pMem, ///< [in][bounds(0, size)] pointer to USM memory object
    size_t
        patternSize, ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                     ///< than or equal to width.
    const void
        *pPattern, ///< [in] pointer with the bytes of the pattern to set.
    size_t
        size, ///< [in] size in bytes to be set. Must be a multiple of patternSize.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnUSMFill = ur_lib::context->urDdiTable.Enqueue.pfnUSMFill;
    if (nullptr == pfnUSMFill) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUSMFill(hQueue, pMem, patternSize, pPattern, size,
                      numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy USM memory
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pSrc` or `pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    bool blocking,            ///< [in] blocking or non-blocking copy
    void *
        pDst, ///< [in][bounds(0, size)] pointer to the destination USM memory object
    const void *
        pSrc, ///< [in][bounds(0, size)] pointer to the source USM memory object
    size_t size,                  ///< [in] size in bytes to be copied
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnUSMMemcpy = ur_lib::context->urDdiTable.Enqueue.pfnUSMMemcpy;
    if (nullptr == pfnUSMMemcpy) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUSMMemcpy(hQueue, blocking, pDst, pSrc, size, numEventsInWaitList,
                        phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to prefetch USM memory
///
/// @details
///     - Prefetching may not be supported for all devices or allocation types.
///       If memory prefetching is not supported, the prefetch hint will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_MIGRATION_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    const void
        *pMem,   ///< [in][bounds(0, size)] pointer to the USM memory object
    size_t size, ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t flags, ///< [in] USM prefetch flags
    uint32_t numEventsInWaitList,   ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnUSMPrefetch = ur_lib::context->urDdiTable.Enqueue.pfnUSMPrefetch;
    if (nullptr == pfnUSMPrefetch) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUSMPrefetch(hQueue, pMem, size, flags, numEventsInWaitList,
                          phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to set USM memory advice
///
/// @details
///     - Not all memory advice hints may be supported for all devices or
///       allocation types. If a memory advice hint is not supported, it will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_ADVICE_FLAGS_MASK & advice`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueUSMAdvise(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    const void
        *pMem,   ///< [in][bounds(0, size)] pointer to the USM memory object
    size_t size, ///< [in] size in bytes to be advised
    ur_usm_advice_flags_t advice, ///< [in] USM memory advice
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnUSMAdvise = ur_lib::context->urDdiTable.Enqueue.pfnUSMAdvise;
    if (nullptr == pfnUSMAdvise) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUSMAdvise(hQueue, pMem, size, advice, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill 2D USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `pitch == 0`
///         + `pitch < width`
///         + `patternSize == 0`
///         + `patternSize > width * height`
///         + `patternSize != 0 && ((patternSize & (patternSize - 1)) != 0)`
///         + `width == 0`
///         + `height == 0`
///         + `width * height % patternSize != 0`
///         + If `pitch * height` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    void *
        pMem, ///< [in][bounds(0, pitch * height)] pointer to memory to be filled.
    size_t
        pitch, ///< [in] the total width of the destination memory including padding.
    size_t
        patternSize, ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                     ///< than or equal to width.
    const void
        *pPattern, ///< [in] pointer with the bytes of the pattern to set.
    size_t
        width, ///< [in] the width in bytes of each row to fill. Must be a multiple of
               ///< patternSize.
    size_t height,                ///< [in] the height of the columns to fill.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
    ) try {
    auto pfnUSMFill2D = ur_lib::context->urDdiTable.Enqueue.pfnUSMFill2D;
    if (nullptr == pfnUSMFill2D) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUSMFill2D(hQueue, pMem, pitch, patternSize, pPattern, width,
                        height, numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy 2D USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `srcPitch == 0`
///         + `dstPitch == 0`
///         + `srcPitch < width`
///         + `dstPitch < width`
///         + `height == 0`
///         + If `srcPitch * height` is higher than the allocation size of `pSrc`
///         + If `dstPitch * height` is higher than the allocation size of `pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    bool blocking, ///< [in] indicates if this operation should block the host.
    void *
        pDst, ///< [in][bounds(0, dstPitch * height)] pointer to memory where data will
              ///< be copied.
    size_t
        dstPitch, ///< [in] the total width of the source memory including padding.
    const void *
        pSrc, ///< [in][bounds(0, srcPitch * height)] pointer to memory to be copied.
    size_t
        srcPitch, ///< [in] the total width of the source memory including padding.
    size_t width,  ///< [in] the width in bytes of each row to be copied.
    size_t height, ///< [in] the height of columns to be copied.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
    ) try {
    auto pfnUSMMemcpy2D = ur_lib::context->urDdiTable.Enqueue.pfnUSMMemcpy2D;
    if (nullptr == pfnUSMMemcpy2D) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUSMMemcpy2D(hQueue, blocking, pDst, dstPitch, pSrc, srcPitch,
                          width, height, numEventsInWaitList, phEventWaitList,
                          phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write data from the host to device global
///        variable.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == name`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    ur_program_handle_t
        hProgram, ///< [in] handle of the program containing the device global variable.
    const char
        *name, ///< [in] the unique identifier for the device global variable.
    bool blockingWrite, ///< [in] indicates if this operation should block.
    size_t count,       ///< [in] the number of bytes to copy.
    size_t
        offset, ///< [in] the byte offset into the device global variable to start copying.
    const void *pSrc, ///< [in] pointer to where the data must be copied from.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
    ) try {
    auto pfnDeviceGlobalVariableWrite =
        ur_lib::context->urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite;
    if (nullptr == pfnDeviceGlobalVariableWrite) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnDeviceGlobalVariableWrite(
        hQueue, hProgram, name, blockingWrite, count, offset, pSrc,
        numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read data from a device global variable to the
///        host.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == name`
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue to submit to.
    ur_program_handle_t
        hProgram, ///< [in] handle of the program containing the device global variable.
    const char
        *name, ///< [in] the unique identifier for the device global variable.
    bool blockingRead, ///< [in] indicates if this operation should block.
    size_t count,      ///< [in] the number of bytes to copy.
    size_t
        offset, ///< [in] the byte offset into the device global variable to start copying.
    void *pDst, ///< [in] pointer to where the data must be copied to.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
    ) try {
    auto pfnDeviceGlobalVariableRead =
        ur_lib::context->urDdiTable.Enqueue.pfnDeviceGlobalVariableRead;
    if (nullptr == pfnDeviceGlobalVariableRead) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnDeviceGlobalVariableRead(hQueue, hProgram, name, blockingRead,
                                       count, offset, pDst, numEventsInWaitList,
                                       phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read from a pipe to the host.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pipe_symbol`
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t
        hQueue, ///< [in] a valid host command-queue in which the read command
    ///< will be queued. hQueue and hProgram must be created with the same
    ///< UR context.
    ur_program_handle_t
        hProgram, ///< [in] a program object with a successfully built executable.
    const char *
        pipe_symbol, ///< [in] the name of the program scope pipe global variable.
    bool
        blocking, ///< [in] indicate if the read operation is blocking or non-blocking.
    void *
        pDst, ///< [in] a pointer to buffer in host memory that will hold resulting data
              ///< from pipe.
    size_t size, ///< [in] size of the memory region to read, in bytes.
    uint32_t numEventsInWaitList, ///< [in] number of events in the wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the host pipe read.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait event.
    ur_event_handle_t *
        phEvent ///< [out][optional] returns an event object that identifies this read
                ///< command
    ///< and can be used to query or queue a wait for this command to complete.
    ) try {
    auto pfnReadHostPipe = ur_lib::context->urDdiTable.Enqueue.pfnReadHostPipe;
    if (nullptr == pfnReadHostPipe) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnReadHostPipe(hQueue, hProgram, pipe_symbol, blocking, pDst, size,
                           numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write data from the host to a pipe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pipe_symbol`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t
        hQueue, ///< [in] a valid host command-queue in which the write command
    ///< will be queued. hQueue and hProgram must be created with the same
    ///< UR context.
    ur_program_handle_t
        hProgram, ///< [in] a program object with a successfully built executable.
    const char *
        pipe_symbol, ///< [in] the name of the program scope pipe global variable.
    bool
        blocking, ///< [in] indicate if the read and write operations are blocking or
                  ///< non-blocking.
    void *
        pSrc, ///< [in] a pointer to buffer in host memory that holds data to be written
              ///< to the host pipe.
    size_t size, ///< [in] size of the memory region to read or write, in bytes.
    uint32_t numEventsInWaitList, ///< [in] number of events in the wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the host pipe write.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait event.
    ur_event_handle_t *
        phEvent ///< [out][optional] returns an event object that identifies this write command
    ///< and can be used to query or queue a wait for this command to complete.
    ) try {
    auto pfnWriteHostPipe =
        ur_lib::context->urDdiTable.Enqueue.pfnWriteHostPipe;
    if (nullptr == pfnWriteHostPipe) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnWriteHostPipe(hQueue, hProgram, pipe_symbol, blocking, pSrc, size,
                            numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate pitched memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_host_desc_t.
///     - See also ::ur_usm_device_desc_t.
///
/// @remarks
///   _Analogues_
///     - **cuMemAllocPitch**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///         + `NULL == pResultPitch`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `widthInBytes == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If `UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT` and `UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT` are both false.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urUSMPitchedAllocExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t *
        pUSMDesc, ///< [in][optional] Pointer to USM memory allocation descriptor.
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        widthInBytes, ///< [in] width in bytes of the USM memory object to be allocated
    size_t height, ///< [in] height of the USM memory object to be allocated
    size_t
        elementSizeBytes, ///< [in] size in bytes of an element in the allocation
    void **ppMem,         ///< [out] pointer to USM shared memory object
    size_t *pResultPitch  ///< [out] pitch of the allocation
    ) try {
    auto pfnPitchedAllocExp =
        ur_lib::context->urDdiTable.USMExp.pfnPitchedAllocExp;
    if (nullptr == pfnPitchedAllocExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnPitchedAllocExp(hContext, hDevice, pUSMDesc, pool, widthInBytes,
                              height, elementSizeBytes, ppMem, pResultPitch);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy bindless unsampled image handles
///
/// @remarks
///   _Analogues_
///     - **cuSurfObjectDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t
        hImage ///< [in] pointer to handle of image object to destroy
    ) try {
    auto pfnUnsampledImageHandleDestroyExp =
        ur_lib::context->urDdiTable.BindlessImagesExp
            .pfnUnsampledImageHandleDestroyExp;
    if (nullptr == pfnUnsampledImageHandleDestroyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUnsampledImageHandleDestroyExp(hContext, hDevice, hImage);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy bindless sampled image handles
///
/// @remarks
///   _Analogues_
///     - **cuTexObjectDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t
        hImage ///< [in] pointer to handle of image object to destroy
    ) try {
    auto pfnSampledImageHandleDestroyExp =
        ur_lib::context->urDdiTable.BindlessImagesExp
            .pfnSampledImageHandleDestroyExp;
    if (nullptr == pfnSampledImageHandleDestroyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSampledImageHandleDestroyExp(hContext, hDevice, hImage);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Allocate memory for bindless images
///
/// @remarks
///   _Analogues_
///     - **cuArray3DCreate**
///     - **cuMipmappedArrayCreate**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_exp_image_mem_handle_t
        *phImageMem ///< [out] pointer to handle of image memory allocated
    ) try {
    auto pfnImageAllocateExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnImageAllocateExp;
    if (nullptr == pfnImageAllocateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImageAllocateExp(hContext, hDevice, pImageFormat, pImageDesc,
                               phImageMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Free memory for bindless images
///
/// @remarks
///   _Analogues_
///     - **cuArrayDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem ///< [in] handle of image memory to be freed
    ) try {
    auto pfnImageFreeExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnImageFreeExp;
    if (nullptr == pfnImageFreeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImageFreeExp(hContext, hDevice, hImageMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a bindless unsampled image handle
///
/// @remarks
///   _Analogues_
///     - **cuSurfObjectCreate**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///         + `NULL == phImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem, ///< [in] handle to memory from which to create the image
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_mem_handle_t *phMem, ///< [out] pointer to handle of image object created
    ur_exp_image_handle_t
        *phImage ///< [out] pointer to handle of image object created
    ) try {
    auto pfnUnsampledImageCreateExp =
        ur_lib::context->urDdiTable.BindlessImagesExp
            .pfnUnsampledImageCreateExp;
    if (nullptr == pfnUnsampledImageCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUnsampledImageCreateExp(hContext, hDevice, hImageMem,
                                      pImageFormat, pImageDesc, phMem, phImage);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a bindless sampled image handle
///
/// @remarks
///   _Analogues_
///     - **cuTexObjectCreate**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///         + `NULL == phImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem, ///< [in] handle to memory from which to create the image
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_sampler_handle_t hSampler,      ///< [in] sampler to be used
    ur_mem_handle_t *phMem, ///< [out] pointer to handle of image object created
    ur_exp_image_handle_t
        *phImage ///< [out] pointer to handle of image object created
    ) try {
    auto pfnSampledImageCreateExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnSampledImageCreateExp;
    if (nullptr == pfnSampledImageCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSampledImageCreateExp(hContext, hDevice, hImageMem, pImageFormat,
                                    pImageDesc, hSampler, phMem, phImage);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Copy image data Host to Device or Device to Host
///
/// @remarks
///   _Analogues_
///     - **cuMemcpyHtoAAsync**
///     - **cuMemcpyAtoHAsync**
///     - **cuMemcpy2DAsync**
///     - **cuMemcpy3DAsync**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_IMAGE_COPY_FLAGS_MASK & imageCopyFlags`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    void *pDst,               ///< [in] location the data will be copied to
    void *pSrc,               ///< [in] location the data will be copied from
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_exp_image_copy_flags_t
        imageCopyFlags, ///< [in] flags describing copy direction e.g. H2D or D2H
    ur_rect_offset_t
        srcOffset, ///< [in] defines the (x,y,z) source offset in pixels in the 1D, 2D, or 3D
                   ///< image
    ur_rect_offset_t
        dstOffset, ///< [in] defines the (x,y,z) destination offset in pixels in the 1D, 2D,
                   ///< or 3D image
    ur_rect_region_t
        copyExtent, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                    ///< region to copy
    ur_rect_region_t
        hostExtent, ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                    ///< region on the host
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnImageCopyExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnImageCopyExp;
    if (nullptr == pfnImageCopyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImageCopyExp(hQueue, pDst, pSrc, pImageFormat, pImageDesc,
                           imageCopyFlags, srcOffset, dstOffset, copyExtent,
                           hostExtent, numEventsInWaitList, phEventWaitList,
                           phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query an image memory handle for specific properties
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_IMAGE_INFO_DEPTH < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    ur_exp_image_mem_handle_t hImageMem, ///< [in] handle to the image memory
    ur_image_info_t propName,            ///< [in] queried info name
    void *pPropValue,    ///< [out][optional] returned query value
    size_t *pPropSizeRet ///< [out][optional] returned query value size
    ) try {
    auto pfnImageGetInfoExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnImageGetInfoExp;
    if (nullptr == pfnImageGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImageGetInfoExp(hImageMem, propName, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve individual image from mipmap
///
/// @remarks
///   _Analogues_
///     - **cuMipmappedArrayGetLevel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_mem_handle_t
        hImageMem,        ///< [in] memory handle to the mipmap image
    uint32_t mipmapLevel, ///< [in] requested level of the mipmap
    ur_exp_image_mem_handle_t
        *phImageMem ///< [out] returning memory handle to the individual image
    ) try {
    auto pfnMipmapGetLevelExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnMipmapGetLevelExp;
    if (nullptr == pfnMipmapGetLevelExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMipmapGetLevelExp(hContext, hDevice, hImageMem, mipmapLevel,
                                phImageMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Free mipmap memory for bindless images
///
/// @remarks
///   _Analogues_
///     - **cuMipmappedArrayDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hMem ///< [in] handle of image memory to be freed
    ) try {
    auto pfnMipmapFreeExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnMipmapFreeExp;
    if (nullptr == pfnMipmapFreeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMipmapFreeExp(hContext, hDevice, hMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Import external memory in the form of a file descriptor
///
/// @remarks
///   _Analogues_
///     - **cuImportExternalMemory**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pInteropMemDesc`
///         + `NULL == phInteropMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
ur_result_t UR_APICALL urBindlessImagesImportOpaqueFDExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    size_t size,                  ///< [in] size of the external memory
    ur_exp_interop_mem_desc_t
        *pInteropMemDesc, ///< [in] the interop memory descriptor
    ur_exp_interop_mem_handle_t
        *phInteropMem ///< [out] interop memory handle to the external memory
    ) try {
    auto pfnImportOpaqueFDExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnImportOpaqueFDExp;
    if (nullptr == pfnImportOpaqueFDExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImportOpaqueFDExp(hContext, hDevice, size, pInteropMemDesc,
                                phInteropMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Map an interop memory handle to an image memory handle
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hInteropMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_image_format_t
        *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc, ///< [in] pointer to image description
    ur_exp_interop_mem_handle_t
        hInteropMem, ///< [in] interop memory handle to the external memory
    ur_exp_image_mem_handle_t *
        phImageMem ///< [out] image memory handle to the externally allocated memory
    ) try {
    auto pfnMapExternalArrayExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnMapExternalArrayExp;
    if (nullptr == pfnMapExternalArrayExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnMapExternalArrayExp(hContext, hDevice, pImageFormat, pImageDesc,
                                  hInteropMem, phImageMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy interop memory
///
/// @remarks
///   _Analogues_
///     - **cuDestroyExternalMemory**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hInteropMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesReleaseInteropExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_mem_handle_t
        hInteropMem ///< [in] handle of interop memory to be freed
    ) try {
    auto pfnReleaseInteropExp =
        ur_lib::context->urDdiTable.BindlessImagesExp.pfnReleaseInteropExp;
    if (nullptr == pfnReleaseInteropExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnReleaseInteropExp(hContext, hDevice, hInteropMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Import an external semaphore in the form of a file descriptor
///
/// @remarks
///   _Analogues_
///     - **cuImportExternalSemaphore**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pInteropSemaphoreDesc`
///         + `NULL == phInteropSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesImportExternalSemaphoreOpaqueFDExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_semaphore_desc_t
        *pInteropSemaphoreDesc, ///< [in] the interop semaphore descriptor
    ur_exp_interop_semaphore_handle_t *
        phInteropSemaphore ///< [out] interop semaphore handle to the external semaphore
    ) try {
    auto pfnImportExternalSemaphoreOpaqueFDExp =
        ur_lib::context->urDdiTable.BindlessImagesExp
            .pfnImportExternalSemaphoreOpaqueFDExp;
    if (nullptr == pfnImportExternalSemaphoreOpaqueFDExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImportExternalSemaphoreOpaqueFDExp(
        hContext, hDevice, pInteropSemaphoreDesc, phInteropSemaphore);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy the external semaphore handle
///
/// @remarks
///   _Analogues_
///     - **cuDestroyExternalSemaphore**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hInteropSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesDestroyExternalSemaphoreExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_interop_semaphore_handle_t
        hInteropSemaphore ///< [in] handle of interop semaphore to be destroyed
    ) try {
    auto pfnDestroyExternalSemaphoreExp =
        ur_lib::context->urDdiTable.BindlessImagesExp
            .pfnDestroyExternalSemaphoreExp;
    if (nullptr == pfnDestroyExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnDestroyExternalSemaphoreExp(hContext, hDevice, hInteropSemaphore);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Instruct the queue with a non-blocking wait on an external semaphore
///
/// @remarks
///   _Analogues_
///     - **cuWaitExternalSemaphoresAsync**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_exp_interop_semaphore_handle_t
        hSemaphore,               ///< [in] interop semaphore handle
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnWaitExternalSemaphoreExp =
        ur_lib::context->urDdiTable.BindlessImagesExp
            .pfnWaitExternalSemaphoreExp;
    if (nullptr == pfnWaitExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnWaitExternalSemaphoreExp(hQueue, hSemaphore, numEventsInWaitList,
                                       phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Instruct the queue to signal the external semaphore handle once all
///        previous commands have completed execution
///
/// @remarks
///   _Analogues_
///     - **cuSignalExternalSemaphoresAsync**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_exp_interop_semaphore_handle_t
        hSemaphore,               ///< [in] interop semaphore handle
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
    ///< previously enqueued commands
    ///< must be complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
    ) try {
    auto pfnSignalExternalSemaphoreExp =
        ur_lib::context->urDdiTable.BindlessImagesExp
            .pfnSignalExternalSemaphoreExp;
    if (nullptr == pfnSignalExternalSemaphoreExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSignalExternalSemaphoreExp(
        hQueue, hSemaphore, numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a Command-Buffer object
///
/// @details
///     - Create a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ///< [in] Handle of the context object.
    ur_device_handle_t hDevice,   ///< [in] Handle of the device object.
    const ur_exp_command_buffer_desc_t
        *pCommandBufferDesc, ///< [in][optional] command-buffer descriptor.
    ur_exp_command_buffer_handle_t
        *phCommandBuffer ///< [out] Pointer to command-Buffer handle.
    ) try {
    auto pfnCreateExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnCreateExp;
    if (nullptr == pfnCreateExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCreateExp(hContext, hDevice, pCommandBufferDesc, phCommandBuffer);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Increment the command-buffer object's reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urCommandBufferRetainExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
    ) try {
    auto pfnRetainExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnRetainExp;
    if (nullptr == pfnRetainExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetainExp(hCommandBuffer);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the command-buffer object's reference count and delete the
///        command-buffer object if the reference count becomes zero.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urCommandBufferReleaseExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
    ) try {
    auto pfnReleaseExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnReleaseExp;
    if (nullptr == pfnReleaseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnReleaseExp(hCommandBuffer);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Stop recording on a command-buffer object such that no more commands
///        can be appended and make it ready to enqueue.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer ///< [in] Handle of the command-buffer object.
    ) try {
    auto pfnFinalizeExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnFinalizeExp;
    if (nullptr == pfnFinalizeExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnFinalizeExp(hCommandBuffer);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a kernel execution command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGlobalWorkOffset`
///         + `NULL == pGlobalWorkSize`
///         + `NULL == pLocalWorkSize`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,         ///< [in] Handle of the command-buffer object.
    ur_kernel_handle_t hKernel, ///< [in] Kernel to append.
    uint32_t workDim,           ///< [in] Dimension of the kernel execution.
    const size_t
        *pGlobalWorkOffset, ///< [in] Offset to use when executing kernel.
    const size_t *
        pGlobalWorkSize, ///< [in] Global work size to use when executing kernel.
    const size_t
        *pLocalWorkSize, ///< [in] Local work size to use when executing kernel.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint, ///< [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_command_handle_t
        *phCommand ///< [out][optional] Handle to this command.
    ) try {
    auto pfnAppendKernelLaunchExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendKernelLaunchExp;
    if (nullptr == pfnAppendKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendKernelLaunchExp(hCommandBuffer, hKernel, workDim,
                                    pGlobalWorkOffset, pGlobalWorkSize,
                                    pLocalWorkSize, numSyncPointsInWaitList,
                                    pSyncPointWaitList, pSyncPoint, phCommand);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM memcpy command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pSrc` or `pDst`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] Handle of the command-buffer object.
    void *pDst,         ///< [in] Location the data will be copied to.
    const void *pSrc,   ///< [in] The data to be copied.
    size_t size,        ///< [in] The number of bytes to copy.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
    ) try {
    auto pfnAppendUSMMemcpyExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendUSMMemcpyExp;
    if (nullptr == pfnAppendUSMMemcpyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendUSMMemcpyExp(hCommandBuffer, pDst, pSrc, size,
                                 numSyncPointsInWaitList, pSyncPointWaitList,
                                 pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM fill command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMemory`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `patternSize == 0 || size == 0`
///         + `patternSize > size`
///         + `(patternSize & (patternSize - 1)) != 0`
///         + `size % patternSize != 0`
///         + If `size` is higher than the allocation size of `ptr`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,   ///< [in] handle of the command-buffer object.
    void *pMemory,        ///< [in] pointer to USM allocated memory to fill.
    const void *pPattern, ///< [in] pointer to the fill pattern.
    size_t patternSize,   ///< [in] size in bytes of the pattern.
    size_t
        size, ///< [in] fill size in bytes, must be a multiple of patternSize.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
    ) try {
    auto pfnAppendUSMFillExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendUSMFillExp;
    if (nullptr == pfnAppendUSMFillExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendUSMFillExp(hCommandBuffer, pMemory, pPattern, patternSize,
                               size, numSyncPointsInWaitList,
                               pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory copy command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hSrcMem`
///         + `NULL == hDstMem`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hSrcMem, ///< [in] The data to be copied.
    ur_mem_handle_t hDstMem, ///< [in] The location the data will be copied to.
    size_t srcOffset,        ///< [in] Offset into the source memory.
    size_t dstOffset,        ///< [in] Offset into the destination memory
    size_t size,             ///< [in] The number of bytes to be copied.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
    ) try {
    auto pfnAppendMemBufferCopyExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendMemBufferCopyExp;
    if (nullptr == pfnAppendMemBufferCopyExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendMemBufferCopyExp(
        hCommandBuffer, hSrcMem, hDstMem, srcOffset, dstOffset, size,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory write command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    size_t offset,           ///< [in] Offset in bytes in the buffer object.
    size_t size,             ///< [in] Size in bytes of data being written.
    const void *
        pSrc, ///< [in] Pointer to host memory where data is to be written from.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
    ) try {
    auto pfnAppendMemBufferWriteExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendMemBufferWriteExp;
    if (nullptr == pfnAppendMemBufferWriteExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendMemBufferWriteExp(hCommandBuffer, hBuffer, offset, size,
                                      pSrc, numSyncPointsInWaitList,
                                      pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory read command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    size_t offset,           ///< [in] Offset in bytes in the buffer object.
    size_t size,             ///< [in] Size in bytes of data being written.
    void *pDst, ///< [in] Pointer to host memory where data is to be written to.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
    ) try {
    auto pfnAppendMemBufferReadExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendMemBufferReadExp;
    if (nullptr == pfnAppendMemBufferReadExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendMemBufferReadExp(hCommandBuffer, hBuffer, offset, size,
                                     pDst, numSyncPointsInWaitList,
                                     pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a rectangular memory copy command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hSrcMem`
///         + `NULL == hDstMem`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hSrcMem, ///< [in] The data to be copied.
    ur_mem_handle_t hDstMem, ///< [in] The location the data will be copied to.
    ur_rect_offset_t
        srcOrigin, ///< [in] Origin for the region of data to be copied from the source.
    ur_rect_offset_t
        dstOrigin, ///< [in] Origin for the region of data to be copied to in the destination.
    ur_rect_region_t
        region, ///< [in] The extents describing the region to be copied.
    size_t srcRowPitch,   ///< [in] Row pitch of the source memory.
    size_t srcSlicePitch, ///< [in] Slice pitch of the source memory.
    size_t dstRowPitch,   ///< [in] Row pitch of the destination memory.
    size_t dstSlicePitch, ///< [in] Slice pitch of the destination memory.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
    ) try {
    auto pfnAppendMemBufferCopyRectExp =
        ur_lib::context->urDdiTable.CommandBufferExp
            .pfnAppendMemBufferCopyRectExp;
    if (nullptr == pfnAppendMemBufferCopyRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendMemBufferCopyRectExp(
        hCommandBuffer, hSrcMem, hDstMem, srcOrigin, dstOrigin, region,
        srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a rectangular memory write command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region.
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t
        bufferRowPitch, ///< [in] Length of each row in bytes in the buffer object.
    size_t
        bufferSlicePitch, ///< [in] Length of each 2D slice in bytes in the buffer object being
                          ///< written.
    size_t
        hostRowPitch, ///< [in] Length of each row in bytes in the host memory region pointed to
                      ///< by pSrc.
    size_t
        hostSlicePitch, ///< [in] Length of each 2D slice in bytes in the host memory region
                        ///< pointed to by pSrc.
    void *
        pSrc, ///< [in] Pointer to host memory where data is to be written from.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
    ) try {
    auto pfnAppendMemBufferWriteRectExp =
        ur_lib::context->urDdiTable.CommandBufferExp
            .pfnAppendMemBufferWriteRectExp;
    if (nullptr == pfnAppendMemBufferWriteRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendMemBufferWriteRectExp(
        hCommandBuffer, hBuffer, bufferOffset, hostOffset, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a rectangular memory read command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] Handle of the buffer object.
    ur_rect_offset_t bufferOffset, ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,   ///< [in] 3D offset in the host region.
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t
        bufferRowPitch, ///< [in] Length of each row in bytes in the buffer object.
    size_t
        bufferSlicePitch, ///< [in] Length of each 2D slice in bytes in the buffer object being read.
    size_t
        hostRowPitch, ///< [in] Length of each row in bytes in the host memory region pointed to
                      ///< by pDst.
    size_t
        hostSlicePitch, ///< [in] Length of each 2D slice in bytes in the host memory region
                        ///< pointed to by pDst.
    void *pDst, ///< [in] Pointer to host memory where data is to be read into.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] Sync point associated with this command.
    ) try {
    auto pfnAppendMemBufferReadRectExp =
        ur_lib::context->urDdiTable.CommandBufferExp
            .pfnAppendMemBufferReadRectExp;
    if (nullptr == pfnAppendMemBufferReadRectExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendMemBufferReadRectExp(
        hCommandBuffer, hBuffer, bufferOffset, hostOffset, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory fill command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,      ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hBuffer, ///< [in] handle of the buffer object.
    const void *pPattern,    ///< [in] pointer to the fill pattern.
    size_t patternSize,      ///< [in] size in bytes of the pattern.
    size_t offset,           ///< [in] offset into the buffer.
    size_t
        size, ///< [in] fill size in bytes, must be a multiple of patternSize.
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
    ) try {
    auto pfnAppendMemBufferFillExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendMemBufferFillExp;
    if (nullptr == pfnAppendMemBufferFillExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendMemBufferFillExp(
        hCommandBuffer, hBuffer, pPattern, patternSize, offset, size,
        numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM Prefetch command to a command-buffer object.
///
/// @details
///     - Prefetching may not be supported for all devices or allocation types.
///       If memory prefetching is not supported, the prefetch hint will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_MIGRATION_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMemory`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,  ///< [in] handle of the command-buffer object.
    const void *pMemory, ///< [in] pointer to USM allocated memory to prefetch.
    size_t size,         ///< [in] size in bytes to be fetched.
    ur_usm_migration_flags_t flags, ///< [in] USM prefetch flags
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
    ) try {
    auto pfnAppendUSMPrefetchExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendUSMPrefetchExp;
    if (nullptr == pfnAppendUSMPrefetchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendUSMPrefetchExp(hCommandBuffer, pMemory, size, flags,
                                   numSyncPointsInWaitList, pSyncPointWaitList,
                                   pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM Advise command to a command-buffer object.
///
/// @details
///     - Not all memory advice hints may be supported for all devices or
///       allocation types. If a memory advice hint is not supported, it will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_ADVICE_FLAGS_MASK & advice`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMemory`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer,           ///< [in] handle of the command-buffer object.
    const void *pMemory,          ///< [in] pointer to the USM memory object.
    size_t size,                  ///< [in] size in bytes to be advised.
    ur_usm_advice_flags_t advice, ///< [in] USM memory advice
    uint32_t
        numSyncPointsInWaitList, ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *
        pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *
        pSyncPoint ///< [out][optional] sync point associated with this command.
    ) try {
    auto pfnAppendUSMAdviseExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnAppendUSMAdviseExp;
    if (nullptr == pfnAppendUSMAdviseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnAppendUSMAdviseExp(hCommandBuffer, pMemory, size, advice,
                                 numSyncPointsInWaitList, pSyncPointWaitList,
                                 pSyncPoint);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Submit a command-buffer for execution on a queue.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] Handle of the command-buffer object.
    ur_queue_handle_t
        hQueue, ///< [in] The queue to submit this command-buffer for execution.
    uint32_t numEventsInWaitList, ///< [in] Size of the event wait list.
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the command-buffer execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating no wait events.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command-buffer execution instance.
    ) try {
    auto pfnEnqueueExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnEnqueueExp;
    if (nullptr == pfnEnqueueExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnEnqueueExp(hCommandBuffer, hQueue, numEventsInWaitList,
                         phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Increment the command object's reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urCommandBufferRetainCommandExp(
    ur_exp_command_buffer_command_handle_t
        hCommand ///< [in] Handle of the command-buffer command.
    ) try {
    auto pfnRetainCommandExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnRetainCommandExp;
    if (nullptr == pfnRetainCommandExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnRetainCommandExp(hCommand);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the command object's reference count and delete the command
///        object if the reference count becomes zero.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urCommandBufferReleaseCommandExp(
    ur_exp_command_buffer_command_handle_t
        hCommand ///< [in] Handle of the command-buffer command.
    ) try {
    auto pfnReleaseCommandExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnReleaseCommandExp;
    if (nullptr == pfnReleaseCommandExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnReleaseCommandExp(hCommand);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Update a kernel launch command in a finalized command-buffer.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pUpdateKernelLaunch`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If update functionality is not supported by the device.
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::ur_exp_command_buffer_desc_t::isUpdatable was not set to true on creation of the command buffer `hCommand` belongs to.
///         + If the command-buffer `hCommand` belongs to has not been finalized.
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_command_handle_t
        hCommand, ///< [in] Handle of the command-buffer kernel command to update.
    const ur_exp_command_buffer_update_kernel_launch_desc_t *
        pUpdateKernelLaunch ///< [in] Struct defining how the kernel command is to be updated.
    ) try {
    auto pfnUpdateKernelLaunchExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnUpdateKernelLaunchExp;
    if (nullptr == pfnUpdateKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnUpdateKernelLaunchExp(hCommand, pUpdateKernelLaunch);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get command-buffer object information.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urCommandBufferGetInfoExp(
    ur_exp_command_buffer_handle_t
        hCommandBuffer, ///< [in] handle of the command-buffer object
    ur_exp_command_buffer_info_t
        propName, ///< [in] the name of the command-buffer property to query
    size_t
        propSize, ///< [in] size in bytes of the command-buffer property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the
                    ///< command-buffer property
    size_t *
        pPropSizeRet ///< [out][optional] bytes returned in command-buffer property
    ) try {
    auto pfnGetInfoExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnGetInfoExp;
    if (nullptr == pfnGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnGetInfoExp(hCommandBuffer, propName, propSize, pPropValue,
                         pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get command-buffer object information.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_COMMAND_BUFFER_COMMAND_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL urCommandBufferCommandGetInfoExp(
    ur_exp_command_buffer_command_handle_t
        hCommand, ///< [in] handle of the command-buffer command object
    ur_exp_command_buffer_command_info_t
        propName, ///< [in] the name of the command-buffer command property to query
    size_t
        propSize, ///< [in] size in bytes of the command-buffer command property value
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] value of the
                    ///< command-buffer command property
    size_t *
        pPropSizeRet ///< [out][optional] bytes returned in command-buffer command property
    ) try {
    auto pfnCommandGetInfoExp =
        ur_lib::context->urDdiTable.CommandBufferExp.pfnCommandGetInfoExp;
    if (nullptr == pfnCommandGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCommandGetInfoExp(hCommand, propName, propSize, pPropValue,
                                pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to execute a cooperative kernel
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGlobalWorkOffset`
///         + `NULL == pGlobalWorkSize`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t hQueue,   ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t
        workDim, ///< [in] number of dimensions, from 1 to 3, to specify the global and
                 ///< work-group work-items
    const size_t *
        pGlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< offset used to calculate the global ID of a work-item
    const size_t *
        pGlobalWorkSize, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< number of global work-items in workDim that will execute the kernel
    ///< function
    const size_t *
        pLocalWorkSize, ///< [in][optional] pointer to an array of workDim unsigned values that
    ///< specify the number of local work-items forming a work-group that will
    ///< execute the kernel function.
    ///< If nullptr, the runtime implementation will choose the work-group
    ///< size.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
    ) try {
    auto pfnCooperativeKernelLaunchExp =
        ur_lib::context->urDdiTable.EnqueueExp.pfnCooperativeKernelLaunchExp;
    if (nullptr == pfnCooperativeKernelLaunchExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCooperativeKernelLaunchExp(
        hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
        pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Query the maximum number of work groups for a cooperative kernel
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGroupCountRet`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
ur_result_t UR_APICALL urKernelSuggestMaxCooperativeGroupCountExp(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    size_t
        localWorkSize, ///< [in] number of local work-items that will form a work-group when the
                       ///< kernel is launched
    size_t
        dynamicSharedMemorySize, ///< [in] size of dynamic shared memory, for each work-group, in bytes,
    ///< that will be used when the kernel is launched
    uint32_t *pGroupCountRet ///< [out] pointer to maximum number of groups
    ) try {
    auto pfnSuggestMaxCooperativeGroupCountExp =
        ur_lib::context->urDdiTable.KernelExp
            .pfnSuggestMaxCooperativeGroupCountExp;
    if (nullptr == pfnSuggestMaxCooperativeGroupCountExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnSuggestMaxCooperativeGroupCountExp(
        hKernel, localWorkSize, dynamicSharedMemorySize, pGroupCountRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one program, negates need for the
///        linking step.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point, the program passed
///       will contain a binary of the ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type
///       for each device in `phDevices`.
///
/// @remarks
///   _Analogues_
///     - **clBuildProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred when building `hProgram`.
ur_result_t UR_APICALL urProgramBuildExp(
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
    ) try {
    auto pfnBuildExp = ur_lib::context->urDdiTable.ProgramExp.pfnBuildExp;
    if (nullptr == pfnBuildExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnBuildExp(hProgram, numDevices, phDevices, pOptions);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point `hProgram` will
///       contain a binary of the ::UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT type
///       for each device in `phDevices`.
///
/// @remarks
///   _Analogues_
///     - **clCompileProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred while compiling `hProgram`.
ur_result_t UR_APICALL urProgramCompileExp(
    ur_program_handle_t
        hProgram,        ///< [in][out] handle of the program to compile.
    uint32_t numDevices, ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
    ) try {
    auto pfnCompileExp = ur_lib::context->urDdiTable.ProgramExp.pfnCompileExp;
    if (nullptr == pfnCompileExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnCompileExp(hProgram, numDevices, phDevices, pOptions);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point the program returned
///       in `phProgram` will contain a binary of the
///       ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type for each device in
///       `phDevices`.
///
/// @remarks
///   _Analogues_
///     - **clLinkProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///         + `NULL == phPrograms`
///         + `NULL == phProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If one of the programs in `phPrograms` isn't a valid program object.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
///     - ::UR_RESULT_ERROR_PROGRAM_LINK_FAILURE
///         + If an error occurred while linking `phPrograms`.
ur_result_t UR_APICALL urProgramLinkExp(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    uint32_t count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *
        phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *
        pOptions, ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
    ) try {
    auto pfnLinkExp = ur_lib::context->urDdiTable.ProgramExp.pfnLinkExp;
    if (nullptr == pfnLinkExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnLinkExp(hContext, numDevices, phDevices, count, phPrograms,
                      pOptions, phProgram);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Import memory into USM
///
/// @details
///     - Import memory into USM
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
ur_result_t UR_APICALL urUSMImportExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem,                   ///< [in] pointer to host memory object
    size_t size ///< [in] size in bytes of the host memory object to be imported
    ) try {
    auto pfnImportExp = ur_lib::context->urDdiTable.USMExp.pfnImportExp;
    if (nullptr == pfnImportExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnImportExp(hContext, pMem, size);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Release memory from USM
///
/// @details
///     - Release memory from USM
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
ur_result_t UR_APICALL urUSMReleaseExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to host memory object
    ) try {
    auto pfnReleaseExp = ur_lib::context->urDdiTable.USMExp.pfnReleaseExp;
    if (nullptr == pfnReleaseExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnReleaseExp(hContext, pMem);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enable access to peer device memory
///
/// @details
///     - Enables the command device to access and write device memory
///       allocations located on the peer device, provided that a P2P link
///       between the two devices is available.
///     - When Peer Access is successfully enabled, P2P memory accesses are
///       guaranteed to be allowed on the peer device until
///       ::urUsmP2PDisablePeerAccessExp is called.
///     - Note that the function operands may, but aren't guaranteed to, commute
///       for a given adapter: the peer device is not guaranteed to have access
///       to device memory allocations located on the command device.
///     - It is not guaranteed that the commutation relations of the function
///       arguments are identical for peer access and peer copies: For example,
///       for a given adapter the peer device may be able to copy data from the
///       command device, but not access and write the same data on the command
///       device.
///     - Consult the appropriate adapter driver documentation for details of
///       adapter specific behavior and native error codes that may be returned.
///
/// @remarks
///   _Analogues_
///     - **cuCtxEnablePeerAccess**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == commandDevice`
///         + `NULL == peerDevice`
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
ur_result_t UR_APICALL urUsmP2PEnablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
    ) try {
    auto pfnEnablePeerAccessExp =
        ur_lib::context->urDdiTable.UsmP2PExp.pfnEnablePeerAccessExp;
    if (nullptr == pfnEnablePeerAccessExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnEnablePeerAccessExp(commandDevice, peerDevice);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Disable access to peer device memory
///
/// @details
///     - Disables the ability of the command device to access and write device
///       memory allocations located on the peer device, provided that a P2P
///       link between the two devices was enabled prior to the call.
///     - Note that the function operands may, but aren't guaranteed to, commute
///       for a given adapter. If, prior to the function call, the peer device
///       had access to device memory allocations on the command device, it is
///       not guaranteed to still have such access following the function
///       return.
///     - It is not guaranteed that the commutation relations of the function
///       arguments are identical for peer access and peer copies: For example
///       for a given adapter, if, prior to the call, the peer device had access
///       to device memory allocations on the command device, the peer device
///       may still, following the function call, be able to copy data from the
///       command device, but not access and write the same data on the command
///       device.
///     - Consult the appropriate adapter driver documentation for details of
///       adapter specific behavior and native error codes that may be returned.
///
/// @remarks
///   _Analogues_
///     - **cuCtxDisablePeerAccess**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == commandDevice`
///         + `NULL == peerDevice`
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
ur_result_t UR_APICALL urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t
        commandDevice,            ///< [in] handle of the command device object
    ur_device_handle_t peerDevice ///< [in] handle of the peer device object
    ) try {
    auto pfnDisablePeerAccessExp =
        ur_lib::context->urDdiTable.UsmP2PExp.pfnDisablePeerAccessExp;
    if (nullptr == pfnDisablePeerAccessExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnDisablePeerAccessExp(commandDevice, peerDevice);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Disable access to peer device memory
///
/// @details
///     - Queries the peer access capabilities from the command device to the
///       peer device according to the query `propName`.
///
/// @remarks
///   _Analogues_
///     - **cuDeviceGetP2PAttribute**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == commandDevice`
///         + `NULL == peerDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
ur_result_t UR_APICALL urUsmP2PPeerAccessGetInfoExp(
    ur_device_handle_t
        commandDevice,             ///< [in] handle of the command device object
    ur_device_handle_t peerDevice, ///< [in] handle of the peer device object
    ur_exp_peer_info_t propName,   ///< [in] type of the info to retrieve
    size_t propSize, ///< [in] the number of bytes pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is not equal to or greater than the real number of bytes
    ///< needed to return the info
    ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
    ) try {
    auto pfnPeerAccessGetInfoExp =
        ur_lib::context->urDdiTable.UsmP2PExp.pfnPeerAccessGetInfoExp;
    if (nullptr == pfnPeerAccessGetInfoExp) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }

    return pfnPeerAccessGetInfoExp(commandDevice, peerDevice, propName,
                                   propSize, pPropValue, pPropSizeRet);
} catch (...) {
    return exceptionToResult(std::current_exception());
}

} // extern "C"
