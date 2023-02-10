/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_api.h
 * @version v0.6-r0
 *
 */
#ifndef UR_API_H_INCLUDED
#define UR_API_H_INCLUDED
#if defined(__cplusplus)
#pragma once
#endif

// standard headers
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Unified Runtime API common types
#if !defined(__GNUC__)
#pragma region common
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef UR_MAKE_VERSION
/// @brief Generates generic 'oneAPI' API versions
#define UR_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))
#endif // UR_MAKE_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_MAJOR_VERSION
/// @brief Extracts 'oneAPI' API major version
#define UR_MAJOR_VERSION(_ver) (_ver >> 16)
#endif // UR_MAJOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_MINOR_VERSION
/// @brief Extracts 'oneAPI' API minor version
#define UR_MINOR_VERSION(_ver) (_ver & 0x0000ffff)
#endif // UR_MINOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_APICALL
#if defined(_WIN32)
/// @brief Calling convention for all API functions
#define UR_APICALL __cdecl
#else
#define UR_APICALL
#endif // defined(_WIN32)
#endif // UR_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_APIEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define UR_APIEXPORT __declspec(dllexport)
#else
#define UR_APIEXPORT
#endif // defined(_WIN32)
#endif // UR_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DLLEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define UR_DLLEXPORT __declspec(dllexport)
#endif // defined(_WIN32)
#endif // UR_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DLLEXPORT
#if __GNUC__ >= 4
/// @brief GCC-specific dllexport storage-class attribute
#define UR_DLLEXPORT __attribute__((visibility("default")))
#else
#define UR_DLLEXPORT
#endif // __GNUC__ >= 4
#endif // UR_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief compiler-independent type
typedef uint8_t ur_bool_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a platform instance
typedef struct ur_platform_handle_t_ *ur_platform_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of platform's device object
typedef struct ur_device_handle_t_ *ur_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of context object
typedef struct ur_context_handle_t_ *ur_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of event object
typedef struct ur_event_handle_t_ *ur_event_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of Program object
typedef struct ur_program_handle_t_ *ur_program_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of program's Kernel object
typedef struct ur_kernel_handle_t_ *ur_kernel_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a queue object
typedef struct ur_queue_handle_t_ *ur_queue_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a native object
typedef struct ur_native_handle_t_ *ur_native_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a Sampler object
typedef struct ur_sampler_handle_t_ *ur_sampler_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of memory object which can either be buffer or image
typedef struct ur_mem_handle_t_ *ur_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_BIT
/// @brief Generic macro for enumerator bit masks
#define UR_BIT(_i) (1 << _i)
#endif // UR_BIT

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes
typedef enum ur_result_t {
    UR_RESULT_SUCCESS = 0,                                ///< Success
    UR_RESULT_ERROR_INVALID_OPERATION = 1,                ///< Invalid operation
    UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES = 2,         ///< Invalid queue properties
    UR_RESULT_ERROR_INVALID_QUEUE = 3,                    ///< Invalid queue
    UR_RESULT_ERROR_INVALID_VALUE = 4,                    ///< Invalid Value
    UR_RESULT_ERROR_INVALID_CONTEXT = 5,                  ///< Invalid context
    UR_RESULT_ERROR_INVALID_PLATFORM = 6,                 ///< Invalid platform
    UR_RESULT_ERROR_INVALID_BINARY = 7,                   ///< Invalid binary
    UR_RESULT_ERROR_INVALID_PROGRAM = 8,                  ///< Invalid program
    UR_RESULT_ERROR_INVALID_SAMPLER = 9,                  ///< Invalid sampler
    UR_RESULT_ERROR_INVALID_BUFFER_SIZE = 10,             ///< Invalid buffer size
    UR_RESULT_ERROR_INVALID_MEM_OBJECT = 11,              ///< Invalid memory object
    UR_RESULT_ERROR_INVALID_EVENT = 12,                   ///< Invalid event
    UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST = 13,         ///< Returned when the event wait list or the events in the wait list are
                                                          ///< invalid.
    UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET = 14,    ///< Misaligned sub buffer offset
    UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE = 15,         ///< Invalid work group size
    UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE = 16,          ///< Compiler not available
    UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE = 17,    ///< Profiling info not available
    UR_RESULT_ERROR_DEVICE_NOT_FOUND = 18,                ///< Device not found
    UR_RESULT_ERROR_INVALID_DEVICE = 19,                  ///< Invalid device
    UR_RESULT_ERROR_DEVICE_LOST = 20,                     ///< Device hung, reset, was removed, or driver update occurred
    UR_RESULT_ERROR_DEVICE_REQUIRES_RESET = 21,           ///< Device requires a reset
    UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 22,       ///< Device currently in low power state
    UR_RESULT_ERROR_DEVICE_PARTITION_FAILED = 23,         ///< Device paritioning failed
    UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT = 24,  ///< Invalid counts provided with ::UR_DEVICE_PARTITION_BY_COUNTS
    UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE = 25,          ///< Invalid work item size
    UR_RESULT_ERROR_INVALID_WORK_DIMENSION = 26,          ///< Invalid work dimension
    UR_RESULT_ERROR_INVALID_KERNEL_ARGS = 27,             ///< Invalid kernel args
    UR_RESULT_ERROR_INVALID_KERNEL = 28,                  ///< Invalid kernel
    UR_RESULT_ERROR_INVALID_KERNEL_NAME = 29,             ///< [Validation] kernel name is not found in the program
    UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 30,   ///< [Validation] kernel argument index is not valid for kernel
    UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 31,    ///< [Validation] kernel argument size does not match kernel
    UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 32,  ///< [Validation] value of kernel attribute is not valid for the kernel or
                                                          ///< device
    UR_RESULT_ERROR_INVALID_IMAGE_SIZE = 33,              ///< Invalid image size
    UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR = 34, ///< Invalid image format descriptor
    UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED = 35,      ///< Image format not supported
    UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE = 36,   ///< Memory object allocation failure
    UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE = 37,      ///< Program object parameter is invalid.
    UR_RESULT_ERROR_UNINITIALIZED = 38,                   ///< [Validation] driver is not initialized
    UR_RESULT_ERROR_OUT_OF_HOST_MEMORY = 39,              ///< Insufficient host memory to satisfy call
    UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 40,            ///< Insufficient device memory to satisfy call
    UR_RESULT_ERROR_OUT_OF_RESOURCES = 41,                ///< Out of resources
    UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE = 42,           ///< Error occurred when building program, see build log for details
    UR_RESULT_ERROR_PROGRAM_LINK_FAILURE = 43,            ///< Error occurred when linking programs, see build log for details
    UR_RESULT_ERROR_UNSUPPORTED_VERSION = 44,             ///< [Validation] generic error code for unsupported versions
    UR_RESULT_ERROR_UNSUPPORTED_FEATURE = 45,             ///< [Validation] generic error code for unsupported features
    UR_RESULT_ERROR_INVALID_ARGUMENT = 46,                ///< [Validation] generic error code for invalid arguments
    UR_RESULT_ERROR_INVALID_NULL_HANDLE = 47,             ///< [Validation] handle argument is not valid
    UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 48,            ///< [Validation] object pointed to by handle still in-use by device
    UR_RESULT_ERROR_INVALID_NULL_POINTER = 49,            ///< [Validation] pointer argument may not be nullptr
    UR_RESULT_ERROR_INVALID_SIZE = 50,                    ///< [Validation] invalid size or dimensions (e.g., must not be zero, or is
                                                          ///< out of bounds)
    UR_RESULT_ERROR_UNSUPPORTED_SIZE = 51,                ///< [Validation] size argument is not supported by the device (e.g., too
                                                          ///< large)
    UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 52,           ///< [Validation] alignment argument is not supported by the device (e.g.,
                                                          ///< too small)
    UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 53,  ///< [Validation] synchronization object in invalid state
    UR_RESULT_ERROR_INVALID_ENUMERATION = 54,             ///< [Validation] enumerator argument is not valid
    UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 55,         ///< [Validation] enumerator argument is not supported by the device
    UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 56,        ///< [Validation] image format is not supported by the device
    UR_RESULT_ERROR_INVALID_NATIVE_BINARY = 57,           ///< [Validation] native binary is not supported by the device
    UR_RESULT_ERROR_INVALID_GLOBAL_NAME = 58,             ///< [Validation] global variable is not found in the program
    UR_RESULT_ERROR_INVALID_FUNCTION_NAME = 59,           ///< [Validation] function name is not found in the program
    UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 60,    ///< [Validation] group size dimension is not valid for the kernel or
                                                          ///< device
    UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 61,  ///< [Validation] global width dimension is not valid for the kernel or
                                                          ///< device
    UR_RESULT_ERROR_PROGRAM_UNLINKED = 62,                ///< [Validation] compiled program or program with imports needs to be
                                                          ///< linked before kernels can be created from it.
    UR_RESULT_ERROR_OVERLAPPING_REGIONS = 63,             ///< [Validation] copy operations do not support overlapping regions of
                                                          ///< memory
    UR_RESULT_ERROR_INVALID_HOST_PTR = 64,                ///< Invalid host pointer
    UR_RESULT_ERROR_INVALID_USM_SIZE = 65,                ///< Invalid USM size
    UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE = 66,       ///< Objection allocation failure
    UR_RESULT_ERROR_ADAPTER_SPECIFIC = 67,                ///< An adapter specific warning/error has been reported and can be
                                                          ///< retrieved via the urGetLastResult entry point.
    UR_RESULT_ERROR_UNKNOWN = 0x7ffffffe,                 ///< Unknown or internal error
    /// @cond
    UR_RESULT_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum ur_structure_type_t {
    UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES = 0,               ///< ::ur_context_properties_t
    UR_STRUCTURE_TYPE_IMAGE_DESC = 1,                       ///< ::ur_image_desc_t
    UR_STRUCTURE_TYPE_BUFFER_PROPERTIES = 2,                ///< ::ur_buffer_properties_t
    UR_STRUCTURE_TYPE_BUFFER_REGION = 3,                    ///< ::ur_buffer_region_t
    UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES = 4,        ///< ::ur_buffer_channel_properties_t
    UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES = 5, ///< ::ur_buffer_alloc_location_properties_t
    UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES = 6,               ///< ::ur_program_properties_t
    UR_STRUCTURE_TYPE_USM_DESC = 7,                         ///< ::ur_usm_desc_t
    UR_STRUCTURE_TYPE_USM_POOL_DESC = 8,                    ///< ::ur_usm_pool_desc_t
    UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC = 9,             ///< ::ur_usm_pool_limits_desc_t
    /// @cond
    UR_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct ur_base_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure

} ur_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct ur_base_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure

} ur_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3D offset argument passed to buffer rect operations
typedef struct ur_rect_offset_t {
    uint64_t x; ///< [in] x offset (bytes)
    uint64_t y; ///< [in] y offset (scalar)
    uint64_t z; ///< [in] z offset (scalar)

} ur_rect_offset_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3D region argument passed to buffer rect operations
typedef struct ur_rect_region_t {
    uint64_t width;  ///< [in] width (bytes)
    uint64_t height; ///< [in] height (scalar)
    uint64_t depth;  ///< [in] scalar (scalar)

} ur_rect_region_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Runtime
#if !defined(__GNUC__)
#pragma region runtime
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device initialization flags
typedef uint32_t ur_device_init_flags_t;
typedef enum ur_device_init_flag_t {
    UR_DEVICE_INIT_FLAG_GPU = UR_BIT(0), ///< initialize GPU device drivers
    /// @cond
    UR_DEVICE_INIT_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_init_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Initialize the 'oneAPI' driver(s)
///
/// @details
///     - The application must call this function before calling any other
///       function.
///     - If this function is not called then all other functions will return
///       ::UR_RESULT_ERROR_UNINITIALIZED.
///     - Only one instance of each driver will be initialized per process.
///     - The application may call this function multiple times with different
///       flags or environment variables enabled.
///     - The application must call this function after forking new processes.
///       Each forked process must call this function.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe for scenarios
///       where multiple libraries may initialize the driver(s) simultaneously.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < device_flags`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urInit(
    ur_device_init_flags_t device_flags ///< [in] device initialization flags.
                                        ///< must be 0 (default) or a combination of ::ur_device_init_flag_t.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Tear down the 'oneAPI' instance and release all its resources
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pParams`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urTearDown(
    void *pParams ///< [in] pointer to tear down parameters
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Platform
#if !defined(__GNUC__)
#pragma region platform
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all available platforms
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
///     - ::UR_RESULT_ERROR_INVALID_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(
    uint32_t NumEntries,               ///< [in] the number of platforms to be added to phPlatforms.
                                       ///< If phPlatforms is not NULL, then NumEntries should be greater than
                                       ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
                                       ///< will be returned.
    ur_platform_handle_t *phPlatforms, ///< [out][optional][range(0, NumEntries)] array of handle of platforms.
                                       ///< If NumEntries is less than the number of platforms available, then
                                       ///< ::urPlatformGet shall only retrieve that number of platforms.
    uint32_t *pNumPlatforms            ///< [out][optional] returns the total number of platforms available.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported platform info
typedef enum ur_platform_info_t {
    UR_PLATFORM_INFO_NAME = 1,        ///< [char*] The string denoting name of the platform. The size of the info
                                      ///< needs to be dynamically queried.
    UR_PLATFORM_INFO_VENDOR_NAME = 2, ///< [char*] The string denoting name of the vendor of the platform. The
                                      ///< size of the info needs to be dynamically queried.
    UR_PLATFORM_INFO_VERSION = 3,     ///< [char*] The string denoting the version of the platform. The size of
                                      ///< the info needs to be dynamically queried.
    UR_PLATFORM_INFO_EXTENSIONS = 4,  ///< [char*] The string denoting extensions supported by the platform. The
                                      ///< size of the info needs to be dynamically queried.
    UR_PLATFORM_INFO_PROFILE = 5,     ///< [char*] The string denoting profile of the platform. The size of the
                                      ///< info needs to be dynamically queried.
    /// @cond
    UR_PLATFORM_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_platform_info_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PLATFORM_INFO_PROFILE < PlatformInfoType`
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(
    ur_platform_handle_t hPlatform,      ///< [in] handle of the platform
    ur_platform_info_t PlatformInfoType, ///< [in] type of the info to retrieve
    size_t Size,                         ///< [in] the number of bytes pointed to by pPlatformInfo.
    void *pPlatformInfo,                 ///< [out][optional] array of bytes holding the info.
                                         ///< If Size is not equal to or greater to the real number of bytes needed
                                         ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                                         ///< returned and pPlatformInfo is not used.
    size_t *pSizeRet                     ///< [out][optional] pointer to the actual number of bytes being queried by pPlatformInfo.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported API versions
///
/// @details
///     - API versions contain major and minor attributes, use
///       ::UR_MAJOR_VERSION and ::UR_MINOR_VERSION
typedef enum ur_api_version_t {
    UR_API_VERSION_0_9 = UR_MAKE_VERSION(0, 9),     ///< version 0.9
    UR_API_VERSION_CURRENT = UR_MAKE_VERSION(0, 9), ///< latest known version
    /// @cond
    UR_API_VERSION_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_api_version_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDriver`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pVersion`
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetApiVersion(
    ur_platform_handle_t hDriver, ///< [in] handle of the platform
    ur_api_version_t *pVersion    ///< [out] api version
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativePlatform`
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform,      ///< [in] handle of the platform.
    ur_native_handle_t *phNativePlatform ///< [out] a pointer to the native handle of the platform.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativePlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPlatform`
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform, ///< [in] the native handle of the platform.
    ur_platform_handle_t *phPlatform    ///< [out] pointer to the handle of the platform object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve string representation of the underlying adapter specific
///        result reported by the the last API that returned
///        UR_RESULT_ADAPTER_SPECIFIC. Allows for an adapter independent way to
///        return an adapter specific result.
///
/// @details
///     - The string returned via the ppMessage is a NULL terminated C style
///       string.
///     - The string returned via the ppMessage is thread local.
///     - The entry point will return UR_RESULT_SUCCESS if the result being
///       reported is to be considered a warning. Any other result code returned
///       indicates that the adapter specific result is an error.
///     - The memory in the string returned via the ppMessage is owned by the
///       adapter.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMessage`
UR_APIEXPORT ur_result_t UR_APICALL
urGetLastResult(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    const char **ppMessage          ///< [out] pointer to a string containing adapter specific result in string
                                    ///< representation.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Device
#if !defined(__GNUC__)
#pragma region device
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum ur_device_type_t {
    UR_DEVICE_TYPE_DEFAULT = 1, ///< The default device type as preferred by the runtime
    UR_DEVICE_TYPE_ALL = 2,     ///< Devices of all types
    UR_DEVICE_TYPE_GPU = 3,     ///< Graphics Processing Unit
    UR_DEVICE_TYPE_CPU = 4,     ///< Central Processing Unit
    UR_DEVICE_TYPE_FPGA = 5,    ///< Field Programmable Gate Array
    UR_DEVICE_TYPE_MCA = 6,     ///< Memory Copy Accelerator
    UR_DEVICE_TYPE_VPU = 7,     ///< Vision Processing Unit
    /// @cond
    UR_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_type_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_TYPE_VPU < DeviceType`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGet(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    ur_device_type_t DeviceType,    ///< [in] the type of the devices.
    uint32_t NumEntries,            ///< [in] the number of devices to be added to phDevices.
                                    ///< If phDevices in not NULL then NumEntries should be greater than zero,
                                    ///< otherwise ::UR_RESULT_ERROR_INVALID_VALUE,
                                    ///< will be returned.
    ur_device_handle_t *phDevices,  ///< [out][optional][range(0, NumEntries)] array of handle of devices.
                                    ///< If NumEntries is less than the number of devices available, then
                                    ///< platform shall only retrieve that number of devices.
    uint32_t *pNumDevices           ///< [out][optional] pointer to the number of devices.
                                    ///< pNumDevices will be updated with the total number of devices available.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device info
typedef enum ur_device_info_t {
    UR_DEVICE_INFO_TYPE = 0,                                    ///< ::ur_device_type_t: type of the device
    UR_DEVICE_INFO_VENDOR_ID = 1,                               ///< uint32_t: vendor Id of the device
    UR_DEVICE_INFO_DEVICE_ID = 2,                               ///< uint32_t: Id of the device
    UR_DEVICE_INFO_MAX_COMPUTE_UNITS = 3,                       ///< uint32_t: the number of compute units
    UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS = 4,                ///< uint32_t: max work item dimensions
    UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES = 5,                     ///< size_t[]: return an array of max work item sizes
    UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE = 6,                     ///< size_t: max work group size
    UR_DEVICE_INFO_SINGLE_FP_CONFIG = 7,                        ///< Return a bit field of ::ur_fp_capability_flags_t: single precision
                                                                ///< floating point capability
    UR_DEVICE_INFO_HALF_FP_CONFIG = 8,                          ///< Return a bit field of ::ur_fp_capability_flags_t: half precision
                                                                ///< floating point capability
    UR_DEVICE_INFO_DOUBLE_FP_CONFIG = 9,                        ///< Return a bit field of ::ur_fp_capability_flags_t: double precision
                                                                ///< floating point capability
    UR_DEVICE_INFO_QUEUE_PROPERTIES = 10,                       ///< Return a bit field of ::ur_queue_flags_t: command queue properties
                                                                ///< supported by the device
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR = 11,            ///< uint32_t: preferred vector width for char
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT = 12,           ///< uint32_t: preferred vector width for short
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT = 13,             ///< uint32_t: preferred vector width for int
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG = 14,            ///< uint32_t: preferred vector width for long
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT = 15,           ///< uint32_t: preferred vector width for float
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE = 16,          ///< uint32_t: preferred vector width for double
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF = 17,            ///< uint32_t: preferred vector width for half float
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR = 18,               ///< uint32_t: native vector width for char
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT = 19,              ///< uint32_t: native vector width for short
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT = 20,                ///< uint32_t: native vector width for int
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG = 21,               ///< uint32_t: native vector width for long
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT = 22,              ///< uint32_t: native vector width for float
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE = 23,             ///< uint32_t: native vector width for double
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF = 24,               ///< uint32_t: native vector width for half float
    UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY = 25,                    ///< uint32_t: max clock frequency in MHz
    UR_DEVICE_INFO_MEMORY_CLOCK_RATE = 26,                      ///< uint32_t: memory clock frequency in MHz
    UR_DEVICE_INFO_ADDRESS_BITS = 27,                           ///< uint32_t: address bits
    UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE = 28,                     ///< uint64_t: max memory allocation size
    UR_DEVICE_INFO_IMAGE_SUPPORTED = 29,                        ///< bool: images are supported
    UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS = 30,                    ///< uint32_t: max number of image objects arguments of a kernel declared
                                                                ///< with the read_only qualifier
    UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS = 31,                   ///< uint32_t: max number of image objects arguments of a kernel declared
                                                                ///< with the write_only qualifier
    UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS = 32,              ///< uint32_t: max number of image objects arguments of a kernel declared
                                                                ///< with the read_write qualifier
    UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH = 33,                      ///< size_t: max width of Image2D object
    UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT = 34,                     ///< size_t: max heigh of Image2D object
    UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH = 35,                      ///< size_t: max width of Image3D object
    UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT = 36,                     ///< size_t: max height of Image3D object
    UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH = 37,                      ///< size_t: max depth of Image3D object
    UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE = 38,                  ///< size_t: max image buffer size
    UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE = 39,                   ///< size_t: max image array size
    UR_DEVICE_INFO_MAX_SAMPLERS = 40,                           ///< uint32_t: max number of samplers that can be used in a kernel
    UR_DEVICE_INFO_MAX_PARAMETER_SIZE = 41,                     ///< size_t: max size in bytes of all arguments passed to a kernel
    UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN = 42,                    ///< uint32_t: memory base address alignment
    UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE = 43,                  ///< ::ur_device_mem_cache_type_t: global memory cache type
    UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE = 44,              ///< uint32_t: global memory cache line size in bytes
    UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE = 45,                  ///< uint64_t: size of global memory cache in bytes
    UR_DEVICE_INFO_GLOBAL_MEM_SIZE = 46,                        ///< uint64_t: size of global memory in bytes
    UR_DEVICE_INFO_GLOBAL_MEM_FREE = 47,                        ///< uint64_t: size of global memory which is free in bytes
    UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE = 48,               ///< uint64_t: max constant buffer size in bytes
    UR_DEVICE_INFO_MAX_CONSTANT_ARGS = 49,                      ///< uint32_t: max number of __const declared arguments in a kernel
    UR_DEVICE_INFO_LOCAL_MEM_TYPE = 50,                         ///< ::ur_device_local_mem_type_t: local memory type
    UR_DEVICE_INFO_LOCAL_MEM_SIZE = 51,                         ///< uint64_t: local memory size in bytes
    UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT = 52,               ///< bool: support error correction to global and local memory
    UR_DEVICE_INFO_HOST_UNIFIED_MEMORY = 53,                    ///< bool: unified host device memory
    UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION = 54,             ///< size_t: profiling timer resolution in nanoseconds
    UR_DEVICE_INFO_ENDIAN_LITTLE = 55,                          ///< bool: little endian byte order
    UR_DEVICE_INFO_AVAILABLE = 56,                              ///< bool: device is available
    UR_DEVICE_INFO_COMPILER_AVAILABLE = 57,                     ///< bool: device compiler is available
    UR_DEVICE_INFO_LINKER_AVAILABLE = 58,                       ///< bool: device linker is available
    UR_DEVICE_INFO_EXECUTION_CAPABILITIES = 59,                 ///< ::ur_device_exec_capability_flags_t: device kernel execution
                                                                ///< capability bit-field
    UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES = 60,             ///< ::ur_queue_flags_t: device command queue property bit-field
    UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES = 61,               ///< ::ur_queue_flags_t: host queue property bit-field
    UR_DEVICE_INFO_BUILT_IN_KERNELS = 62,                       ///< char[]: a semi-colon separated list of built-in kernels
    UR_DEVICE_INFO_PLATFORM = 63,                               ///< ::ur_platform_handle_t: the platform associated with the device
    UR_DEVICE_INFO_REFERENCE_COUNT = 64,                        ///< [uint32_t] Reference count of the device object.
                                                                ///< The reference count returned should be considered immediately stale.
                                                                ///< It is unsuitable for general use in applications. This feature is
                                                                ///< provided for identifying memory leaks.
    UR_DEVICE_INFO_IL_VERSION = 65,                             ///< char[]: IL version
    UR_DEVICE_INFO_NAME = 66,                                   ///< char[]: Device name
    UR_DEVICE_INFO_VENDOR = 67,                                 ///< char[]: Device vendor
    UR_DEVICE_INFO_DRIVER_VERSION = 68,                         ///< char[]: Driver version
    UR_DEVICE_INFO_PROFILE = 69,                                ///< char[]: Device profile
    UR_DEVICE_INFO_VERSION = 70,                                ///< char[]: Device version
    UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION = 71,                ///< char[]: Version of backend runtime
    UR_DEVICE_INFO_EXTENSIONS = 72,                             ///< char[]: Return a space separated list of extension names
    UR_DEVICE_INFO_PRINTF_BUFFER_SIZE = 73,                     ///< size_t: Maximum size in bytes of internal printf buffer
    UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC = 74,            ///< bool: prefer user synchronization when sharing object with other API
    UR_DEVICE_INFO_PARENT_DEVICE = 75,                          ///< ::ur_device_handle_t: return parent device handle
    UR_DEVICE_INFO_PARTITION_PROPERTIES = 76,                   ///< ::ur_device_partition_property_t[]: Returns the list of partition
                                                                ///< types supported by the device
    UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES = 77,              ///< uint32_t: maximum number of sub-devices when the device is partitioned
    UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN = 78,              ///< uint32_t: return a bit-field of affinity domain
                                                                ///< ::ur_device_affinity_domain_flags_t
    UR_DEVICE_INFO_PARTITION_TYPE = 79,                         ///< ::ur_device_partition_property_t[]: return a list of
                                                                ///< ::ur_device_partition_property_t for properties specified in
                                                                ///< ::urDevicePartition
    UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS = 80,                     ///< uint32_t: max number of sub groups
    UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 81, ///< bool: support sub group independent forward progress
    UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL = 82,                  ///< uint32_t[]: return an array of sub group sizes supported on Intel
                                                                ///< device
    UR_DEVICE_INFO_USM_HOST_SUPPORT = 83,                       ///< bool: support USM host memory access
    UR_DEVICE_INFO_USM_DEVICE_SUPPORT = 84,                     ///< bool: support USM device memory access
    UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT = 85,              ///< bool: support USM single device shared memory access
    UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT = 86,               ///< bool: support USM cross device shared memory access
    UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT = 87,              ///< bool: support USM system wide shared memory access
    UR_DEVICE_INFO_UUID = 88,                                   ///< char[]: return device UUID
    UR_DEVICE_INFO_PCI_ADDRESS = 89,                            ///< char[]: return device PCI address
    UR_DEVICE_INFO_GPU_EU_COUNT = 90,                           ///< uint32_t: return Intel GPU EU count
    UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH = 91,                      ///< uint32_t: return Intel GPU EU SIMD width
    UR_DEVICE_INFO_GPU_EU_SLICES = 92,                          ///< uint32_t: return Intel GPU number of slices
    UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE = 93,                ///< uint32_t: return Intel GPU number of subslices per slice
    UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH = 94,                   ///< uint32_t: return max memory bandwidth in Mb/s
    UR_DEVICE_INFO_IMAGE_SRGB = 95,                             ///< bool: image is SRGB
    UR_DEVICE_INFO_ATOMIC_64 = 96,                              ///< bool: support 64 bit atomics
    UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES = 97,       ///< ::ur_memory_order_capability_flags_t: return a bit-field of atomic
                                                                ///< memory order capabilities
    UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES = 98,       ///< ::ur_memory_scope_capability_flags_t: return a bit-field of atomic
                                                                ///< memory scope capabilities
    UR_DEVICE_INFO_BFLOAT16 = 99,                               ///< bool: support for bfloat16
    UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES = 100,             ///< uint32_t: Returns 1 if the device doesn't have a notion of a
                                                                ///< queue index. Otherwise, returns the number of queue indices that are
                                                                ///< available for this device.
    UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS = 101,   ///< `bool`: support the ::urKernelSetSpecializationConstants entry point
    /// @cond
    UR_DEVICE_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_info_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS < infoType`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGetInfo(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    ur_device_info_t infoType,  ///< [in] type of the info to retrieve
    size_t propSize,            ///< [in] the number of bytes pointed to by pDeviceInfo.
    void *pDeviceInfo,          ///< [out][optional] array of bytes holding the info.
                                ///< If propSize is not equal to or greater than the real number of bytes
                                ///< needed to return the info
                                ///< then the ::UR_RESULT_ERROR_INVALID_VALUE error is returned and
                                ///< pDeviceInfo is not used.
    size_t *pPropSizeRet        ///< [out][optional] pointer to the actual size in bytes of the queried infoType.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes a reference of the device handle indicating it's in use until
///        paired ::urDeviceRelease is called
///
/// @details
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRetain(
    ur_device_handle_t hDevice ///< [in] handle of the device to get a reference of.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Releases the device handle reference indicating end of its usage
///
/// @details
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRelease(
    ur_device_handle_t hDevice ///< [in] handle of the device to release.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Device partition property type
typedef intptr_t ur_device_partition_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Partition Properties
typedef enum ur_device_partition_t {
    UR_DEVICE_PARTITION_EQUALLY = 0x1086,            ///< Partition Equally
    UR_DEVICE_PARTITION_BY_COUNTS = 0x1087,          ///< Partition by counts
    UR_DEVICE_PARTITION_BY_COUNTS_LIST_END = 0x0,    ///< End of by counts list
    UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088, ///< Partition by affinity domain
    UR_DEVICE_PARTITION_BY_CSLICE = 0x1089,          ///< Partition by c-slice
    /// @cond
    UR_DEVICE_PARTITION_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_partition_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pProperties`
///     - ::UR_RESULT_ERROR_DEVICE_PARTITION_FAILED
///     - ::UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT
UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(
    ur_device_handle_t hDevice,                        ///< [in] handle of the device to partition.
    const ur_device_partition_property_t *pProperties, ///< [in] null-terminated array of <$_device_partition_t enum, value> pairs.
    uint32_t NumDevices,                               ///< [in] the number of sub-devices.
    ur_device_handle_t *phSubDevices,                  ///< [out][optional][range(0, NumDevices)] array of handle of devices.
                                                       ///< If NumDevices is less than the number of sub-devices available, then
                                                       ///< the function shall only retrieve that number of sub-devices.
    uint32_t *pNumDevicesRet                           ///< [out][optional] pointer to the number of sub-devices the device can be
                                                       ///< partitioned into according to the partitioning property.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppBinaries`
///         + `NULL == pSelectedBinary`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceSelectBinary(
    ur_device_handle_t hDevice, ///< [in] handle of the device to select binary for.
    const uint8_t **ppBinaries, ///< [in] the array of binaries to select from.
    uint32_t NumBinaries,       ///< [in] the number of binaries passed in ppBinaries.
                                ///< Must greater than or equal to zero otherwise
                                ///< ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t *pSelectedBinary   ///< [out] the index of the selected binary in the input array of binaries.
                                ///< If a suitable binary was not found the function returns ${X}_INVALID_BINARY.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief FP capabilities
typedef uint32_t ur_fp_capability_flags_t;
typedef enum ur_fp_capability_flag_t {
    UR_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT = UR_BIT(0), ///< Support correctly rounded divide and sqrt
    UR_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST = UR_BIT(1),              ///< Support round to nearest
    UR_FP_CAPABILITY_FLAG_ROUND_TO_ZERO = UR_BIT(2),                 ///< Support round to zero
    UR_FP_CAPABILITY_FLAG_ROUND_TO_INF = UR_BIT(3),                  ///< Support round to infinity
    UR_FP_CAPABILITY_FLAG_INF_NAN = UR_BIT(4),                       ///< Support INF to NAN
    UR_FP_CAPABILITY_FLAG_DENORM = UR_BIT(5),                        ///< Support denorm
    UR_FP_CAPABILITY_FLAG_FMA = UR_BIT(6),                           ///< Support FMA
    /// @cond
    UR_FP_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_fp_capability_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device memory cache type
typedef enum ur_device_mem_cache_type_t {
    UR_DEVICE_MEM_CACHE_TYPE_NONE = 0,             ///< Has none cache
    UR_DEVICE_MEM_CACHE_TYPE_READ_ONLY_CACHE = 1,  ///< Has read only cache
    UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE = 2, ///< Has read write cache
    /// @cond
    UR_DEVICE_MEM_CACHE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_mem_cache_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device local memory type
typedef enum ur_device_local_mem_type_t {
    UR_DEVICE_LOCAL_MEM_TYPE_NONE = 0,   ///< No local memory support
    UR_DEVICE_LOCAL_MEM_TYPE_LOCAL = 1,  ///< Dedicated local memory
    UR_DEVICE_LOCAL_MEM_TYPE_GLOBAL = 2, ///< Global memory
    /// @cond
    UR_DEVICE_LOCAL_MEM_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_local_mem_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device kernel execution capability
typedef uint32_t ur_device_exec_capability_flags_t;
typedef enum ur_device_exec_capability_flag_t {
    UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL = UR_BIT(0),        ///< Support kernel execution
    UR_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL = UR_BIT(1), ///< Support native kernel execution
    /// @cond
    UR_DEVICE_EXEC_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_exec_capability_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device affinity domain
typedef uint32_t ur_device_affinity_domain_flags_t;
typedef enum ur_device_affinity_domain_flag_t {
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA = UR_BIT(0),               ///< By NUMA
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE = UR_BIT(1), ///< BY next partitionable
    /// @cond
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_affinity_domain_flag_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGetNativeHandle(
    ur_device_handle_t hDevice,        ///< [in] handle of the device.
    ur_native_handle_t *phNativeDevice ///< [out] a pointer to the native handle of the device.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeDevice`
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice, ///< [in] the native handle of the device.
    ur_platform_handle_t hPlatform,   ///< [in] handle of the platform instance
    ur_device_handle_t *phDevice      ///< [out] pointer to the handle of the device object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief static
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    uint64_t *pDeviceTimestamp, ///< [out][optional] pointer to the Device's global timestamp that
                                ///< correlates with the Host's global timestamp value
    uint64_t *pHostTimestamp    ///< [out][optional] pointer to the Host's global timestamp that
                                ///< correlates with the Device's global timestamp value
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory order capabilities
typedef uint32_t ur_memory_order_capability_flags_t;
typedef enum ur_memory_order_capability_flag_t {
    UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED = UR_BIT(0), ///< Relaxed memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE = UR_BIT(1), ///< Acquire memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE = UR_BIT(2), ///< Release memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL = UR_BIT(3), ///< Acquire/release memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST = UR_BIT(4), ///< Sequentially consistent memory ordering
    /// @cond
    UR_MEMORY_ORDER_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_memory_order_capability_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory scope capabilities
typedef uint32_t ur_memory_scope_capability_flags_t;
typedef enum ur_memory_scope_capability_flag_t {
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM = UR_BIT(0),  ///< Work item scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP = UR_BIT(1),  ///< Sub group scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP = UR_BIT(2), ///< Work group scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE = UR_BIT(3),     ///< Device scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM = UR_BIT(4),     ///< System scope
    /// @cond
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_memory_scope_capability_flag_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Context
#if !defined(__GNUC__)
#pragma region context
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Context property type
typedef uint32_t ur_context_flags_t;
typedef enum ur_context_flag_t {
    UR_CONTEXT_FLAG_TBD = UR_BIT(0), ///< reserved for future use
    /// @cond
    UR_CONTEXT_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_context_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Context creation properties
typedef struct ur_context_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    ur_context_flags_t flags;  ///< [in] context creation flags.

} ur_context_properties_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///         + `NULL == phContext`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urContextCreate(
    uint32_t DeviceCount,                       ///< [in] the number of devices given in phDevices
    const ur_device_handle_t *phDevices,        ///< [in][range(0, DeviceCount)] array of handle of devices.
    const ur_context_properties_t *pProperties, ///< [in][optional] pointer to context creation properties.
    ur_context_handle_t *phContext              ///< [out] pointer to handle of context object created
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(
    ur_context_handle_t hContext ///< [in] handle of the context to get a reference of.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported context info
typedef enum ur_context_info_t {
    UR_CONTEXT_INFO_NUM_DEVICES = 0,          ///< [uint32_t] The number of the devices in the context
    UR_CONTEXT_INFO_DEVICES = 1,              ///< [::ur_context_handle_t...] The array of the device handles in the
                                              ///< context
    UR_CONTEXT_INFO_REFERENCE_COUNT = 2,      ///< [uint32_t] Reference count of the context object.
                                              ///< The reference count returned should be considered immediately stale.
                                              ///< It is unsuitable for general use in applications. This feature is
                                              ///< provided for identifying memory leaks.
    UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT = 3, ///< [bool] to indicate if the ::urEnqueueUSMMemcpy2D entrypoint is
                                              ///< supported.
    UR_CONTEXT_INFO_USM_FILL2D_SUPPORT = 4,   ///< [bool] to indicate if the ::urEnqueueUSMFill2D entrypoint is
                                              ///< supported.
    /// @cond
    UR_CONTEXT_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_context_info_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
UR_APIEXPORT ur_result_t UR_APICALL
urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_CONTEXT_INFO_USM_FILL2D_SUPPORT < ContextInfoType`
UR_APIEXPORT ur_result_t UR_APICALL
urContextGetInfo(
    ur_context_handle_t hContext,      ///< [in] handle of the context
    ur_context_info_t ContextInfoType, ///< [in] type of the info to retrieve
    size_t propSize,                   ///< [in] the number of bytes of memory pointed to by pContextInfo.
    void *pContextInfo,                ///< [out][optional] array of bytes holding the info.
                                       ///< if propSize is not equal to or greater than the real number of bytes
                                       ///< needed to return
                                       ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                       ///< pContextInfo is not used.
    size_t *pPropSizeRet               ///< [out][optional] pointer to the actual size in bytes of data queried by ContextInfoType.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeContext`
UR_APIEXPORT ur_result_t UR_APICALL
urContextGetNativeHandle(
    ur_context_handle_t hContext,       ///< [in] handle of the context.
    ur_native_handle_t *phNativeContext ///< [out] a pointer to the native handle of the context.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phContext`
UR_APIEXPORT ur_result_t UR_APICALL
urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext, ///< [in] the native handle of the context.
    ur_context_handle_t *phContext     ///< [out] pointer to the handle of the context object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Context's extended deleter callback function with user data.
typedef void (*ur_context_extended_deleter_t)(
    void *pUserData ///< [in][out] pointer to data to be passed to callback
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnDeleter`
UR_APIEXPORT ur_result_t UR_APICALL
urContextSetExtendedDeleter(
    ur_context_handle_t hContext,             ///< [in] handle of the context.
    ur_context_extended_deleter_t pfnDeleter, ///< [in] Function pointer to extended deleter.
    void *pUserData                           ///< [in][out][optional] pointer to data to be passed to callback.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region memory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Memory flags
typedef uint32_t ur_mem_flags_t;
typedef enum ur_mem_flag_t {
    UR_MEM_FLAG_READ_WRITE = UR_BIT(0),              ///< The memory object will be read and written by a kernel. This is the
                                                     ///< default
    UR_MEM_FLAG_WRITE_ONLY = UR_BIT(1),              ///< The memory object will be written but not read by a kernel
    UR_MEM_FLAG_READ_ONLY = UR_BIT(2),               ///< The memory object is a read-only inside a kernel
    UR_MEM_FLAG_USE_HOST_POINTER = UR_BIT(3),        ///< Use memory pointed by a host pointer parameter as the storage bits for
                                                     ///< the memory object
    UR_MEM_FLAG_ALLOC_HOST_POINTER = UR_BIT(4),      ///< Allocate memory object from host accessible memory
    UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER = UR_BIT(5), ///< Allocate memory and copy the data from host pointer pointed memory
    /// @cond
    UR_MEM_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_mem_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory types
typedef enum ur_mem_type_t {
    UR_MEM_TYPE_BUFFER = 0,         ///< Buffer object
    UR_MEM_TYPE_IMAGE2D = 1,        ///< 2D image object
    UR_MEM_TYPE_IMAGE3D = 2,        ///< 3D image object
    UR_MEM_TYPE_IMAGE2D_ARRAY = 3,  ///< 2D image array object
    UR_MEM_TYPE_IMAGE1D = 4,        ///< 1D image object
    UR_MEM_TYPE_IMAGE1D_ARRAY = 5,  ///< 1D image array object
    UR_MEM_TYPE_IMAGE1D_BUFFER = 6, ///< 1D image buffer object
    /// @cond
    UR_MEM_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_mem_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory Information type
typedef enum ur_mem_info_t {
    UR_MEM_INFO_SIZE = 0,    ///< size_t: actual size of of memory object in bytes
    UR_MEM_INFO_CONTEXT = 1, ///< ::ur_context_handle_t: context in which the memory object was created
    /// @cond
    UR_MEM_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_mem_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image channel order info: number of channels and the channel layout
typedef enum ur_image_channel_order_t {
    UR_IMAGE_CHANNEL_ORDER_A = 0,         ///< channel order A
    UR_IMAGE_CHANNEL_ORDER_R = 1,         ///< channel order R
    UR_IMAGE_CHANNEL_ORDER_RG = 2,        ///< channel order RG
    UR_IMAGE_CHANNEL_ORDER_RA = 3,        ///< channel order RA
    UR_IMAGE_CHANNEL_ORDER_RGB = 4,       ///< channel order RGB
    UR_IMAGE_CHANNEL_ORDER_RGBA = 5,      ///< channel order RGBA
    UR_IMAGE_CHANNEL_ORDER_BGRA = 6,      ///< channel order BGRA
    UR_IMAGE_CHANNEL_ORDER_ARGB = 7,      ///< channel order ARGB
    UR_IMAGE_CHANNEL_ORDER_INTENSITY = 8, ///< channel order intensity
    UR_IMAGE_CHANNEL_ORDER_LUMINANCE = 9, ///< channel order luminance
    UR_IMAGE_CHANNEL_ORDER_RX = 10,       ///< channel order Rx
    UR_IMAGE_CHANNEL_ORDER_RGX = 11,      ///< channel order RGx
    UR_IMAGE_CHANNEL_ORDER_RGBX = 12,     ///< channel order RGBx
    UR_IMAGE_CHANNEL_ORDER_SRGBA = 13,    ///< channel order sRGBA
    /// @cond
    UR_IMAGE_CHANNEL_ORDER_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_image_channel_order_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image channel type info: describe the size of the channel data type
typedef enum ur_image_channel_type_t {
    UR_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0,      ///< channel type snorm int8
    UR_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1,     ///< channel type snorm int16
    UR_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2,      ///< channel type unorm int8
    UR_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3,     ///< channel type unorm int16
    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = 4, ///< channel type unorm short 565
    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = 5, ///< channel type unorm short 555
    UR_IMAGE_CHANNEL_TYPE_INT_101010 = 6,      ///< channel type int 101010
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 7,     ///< channel type signed int8
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 8,    ///< channel type signed int16
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 9,    ///< channel type signed int32
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 10,  ///< channel type unsigned int8
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 11, ///< channel type unsigned int16
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 12, ///< channel type unsigned int32
    UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 13,     ///< channel type half float
    UR_IMAGE_CHANNEL_TYPE_FLOAT = 14,          ///< channel type float
    /// @cond
    UR_IMAGE_CHANNEL_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_image_channel_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image information types
typedef enum ur_image_info_t {
    UR_IMAGE_INFO_FORMAT = 0,       ///< ::ur_image_format_t: image format
    UR_IMAGE_INFO_ELEMENT_SIZE = 1, ///< size_t: element size
    UR_IMAGE_INFO_ROW_PITCH = 2,    ///< size_t: row pitch
    UR_IMAGE_INFO_SLICE_PITCH = 3,  ///< size_t: slice pitch
    UR_IMAGE_INFO_WIDTH = 4,        ///< size_t: image width
    UR_IMAGE_INFO_HEIGHT = 5,       ///< size_t: image height
    UR_IMAGE_INFO_DEPTH = 6,        ///< size_t: image depth
    /// @cond
    UR_IMAGE_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_image_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image format including channel layout and data type
typedef struct ur_image_format_t {
    ur_image_channel_order_t channelOrder; ///< [in] image channel order
    ur_image_channel_type_t channelType;   ///< [in] image channel type

} ur_image_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image descriptor type.
typedef struct ur_image_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_IMAGE_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    ur_mem_type_t type;        ///< [in] memory object type
    size_t width;              ///< [in] image width
    size_t height;             ///< [in] image height
    size_t depth;              ///< [in] image depth
    size_t arraySize;          ///< [in] image array size
    size_t rowPitch;           ///< [in] image row pitch
    size_t slicePitch;         ///< [in] image slice pitch
    uint32_t numMipLevel;      ///< [in] number of MIP levels
    uint32_t numSamples;       ///< [in] number of samples

} ur_image_desc_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
///         + `::UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///         + `pHost == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pHost != NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urMemImageCreate(
    ur_context_handle_t hContext,          ///< [in] handle of the context object
    ur_mem_flags_t flags,                  ///< [in] allocation and usage information flags
    const ur_image_format_t *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,     ///< [in] pointer to image description
    void *pHost,                           ///< [in][optional] pointer to the buffer data
    ur_mem_handle_t *phMem                 ///< [out] pointer to handle of image object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer creation properties
typedef struct ur_buffer_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_BUFFER_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    void *pHost;               ///< [in][optional] pointer to the buffer data

} ur_buffer_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer memory channel creation properties
///
/// @details
///     - Specify these properties in ::urMemBufferCreate via
///       ::ur_buffer_properties_t as part of a `pNext` chain.
///
/// @remarks
///   _Analogues_
///     - cl_intel_mem_channel_property
typedef struct ur_buffer_channel_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    uint32_t channel;          ///< [in] Identifies the channel/region to which the buffer should be mapped.

} ur_buffer_channel_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer allocation location creation properties
///
/// @details
///     - Specify these properties in ::urMemBufferCreate via
///       ::ur_buffer_properties_t as part of a `pNext` chain.
///
/// @remarks
///   _Analogues_
///     - cl_intel_mem_alloc_buffer_location
typedef struct ur_buffer_alloc_location_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    uint32_t location;         ///< [in] Identifies the ID of global memory partition to which the memory
                               ///< should be allocated.

} ur_buffer_alloc_location_properties_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phBuffer`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_BUFFER_SIZE
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///         + `pProperties == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pProperties->pHost == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pProperties->pHost != NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urMemBufferCreate(
    ur_context_handle_t hContext,              ///< [in] handle of the context object
    ur_mem_flags_t flags,                      ///< [in] allocation and usage information flags
    size_t size,                               ///< [in] size in bytes of the memory object to be allocated
    const ur_buffer_properties_t *pProperties, ///< [in][optional] pointer to buffer creation properties
    ur_mem_handle_t *phBuffer                  ///< [out] pointer to handle of the memory buffer created
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer region type, used to describe a sub buffer
typedef struct ur_buffer_region_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_BUFFER_REGION
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    size_t origin;             ///< [in] buffer origin offset
    size_t size;               ///< [in] size of the buffer region

} ur_buffer_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer creation type
typedef enum ur_buffer_create_type_t {
    UR_BUFFER_CREATE_TYPE_REGION = 0, ///< buffer create type is region
    /// @cond
    UR_BUFFER_CREATE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_buffer_create_type_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
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
UR_APIEXPORT ur_result_t UR_APICALL
urMemBufferPartition(
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t flags,                     ///< [in] allocation and usage information flags
    ur_buffer_create_type_t bufferCreateType, ///< [in] buffer creation type
    const ur_buffer_region_t *pRegion,        ///< [in] pointer to buffer create region information
    ur_mem_handle_t *phMem                    ///< [out] pointer to the handle of sub buffer created
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeMem`
UR_APIEXPORT ur_result_t UR_APICALL
urMemGetNativeHandle(
    ur_mem_handle_t hMem,           ///< [in] handle of the mem.
    ur_native_handle_t *phNativeMem ///< [out] a pointer to the native handle of the mem.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime mem object from native mem handle.
///
/// @details
///     - Creates runtime mem handle from native driver mem handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeMem`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phMem`
UR_APIEXPORT ur_result_t UR_APICALL
urMemCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ///< [in] the native handle of the mem.
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_mem_handle_t *phMem         ///< [out] pointer to the handle of the mem object created.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_INFO_CONTEXT < MemInfoType`
UR_APIEXPORT ur_result_t UR_APICALL
urMemGetInfo(
    ur_mem_handle_t hMemory,   ///< [in] handle to the memory object being queried.
    ur_mem_info_t MemInfoType, ///< [in] type of the info to retrieve.
    size_t propSize,           ///< [in] the number of bytes of memory pointed to by pMemInfo.
    void *pMemInfo,            ///< [out][optional] array of bytes holding the info.
                               ///< If propSize is less than the real number of bytes needed to return
                               ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                               ///< pMemInfo is not used.
    size_t *pPropSizeRet       ///< [out][optional] pointer to the actual size in bytes of data queried by pMemInfo.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_IMAGE_INFO_DEPTH < ImgInfoType`
UR_APIEXPORT ur_result_t UR_APICALL
urMemImageGetInfo(
    ur_mem_handle_t hMemory,     ///< [in] handle to the image object being queried.
    ur_image_info_t ImgInfoType, ///< [in] type of image info to retrieve.
    size_t propSize,             ///< [in] the number of bytes of memory pointer to by pImgInfo.
    void *pImgInfo,              ///< [out][optional] array of bytes holding the info.
                                 ///< If propSize is less than the real number of bytes needed to return
                                 ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                 ///< pImgInfo is not used.
    size_t *pPropSizeRet         ///< [out][optional] pointer to the actual size in bytes of data queried by pImgInfo.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region sampler
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Get sample object information
typedef enum ur_sampler_info_t {
    UR_SAMPLER_INFO_REFERENCE_COUNT = 0,   ///< [uint32_t] Reference count of the sampler object.
                                           ///< The reference count returned should be considered immediately stale.
                                           ///< It is unsuitable for general use in applications. This feature is
                                           ///< provided for identifying memory leaks.
    UR_SAMPLER_INFO_CONTEXT = 1,           ///< Sampler context info
    UR_SAMPLER_INFO_NORMALIZED_COORDS = 2, ///< Sampler normalized coordindate setting
    UR_SAMPLER_INFO_ADDRESSING_MODE = 3,   ///< Sampler addressing mode setting
    UR_SAMPLER_INFO_FILTER_MODE = 4,       ///< Sampler filter mode setting
    UR_SAMPLER_INFO_MIP_FILTER_MODE = 5,   ///< Sampler MIP filter mode setting
    UR_SAMPLER_INFO_LOD_MIN = 6,           ///< Sampler LOD Min value
    UR_SAMPLER_INFO_LOD_MAX = 7,           ///< Sampler LOD Max value
    /// @cond
    UR_SAMPLER_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_sampler_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler properties
typedef enum ur_sampler_properties_t {
    UR_SAMPLER_PROPERTIES_NORMALIZED_COORDS = 0, ///< Sampler normalized coordinates
    UR_SAMPLER_PROPERTIES_ADDRESSING_MODE = 1,   ///< Sampler addressing mode
    UR_SAMPLER_PROPERTIES_FILTER_MODE = 2,       ///< Sampler filter mode
    /// @cond
    UR_SAMPLER_PROPERTIES_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_sampler_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler Properties type
typedef intptr_t ur_sampler_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler addressing mode
typedef enum ur_sampler_addressing_mode_t {
    UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = 0, ///< Mirrored Repeat
    UR_SAMPLER_ADDRESSING_MODE_REPEAT = 1,          ///< Repeat
    UR_SAMPLER_ADDRESSING_MODE_CLAMP = 2,           ///< Clamp
    UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = 3,   ///< Clamp to edge
    UR_SAMPLER_ADDRESSING_MODE_NONE = 4,            ///< None
    /// @cond
    UR_SAMPLER_ADDRESSING_MODE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_sampler_addressing_mode_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pProps`
///         + `NULL == phSampler`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerCreate(
    ur_context_handle_t hContext,        ///< [in] handle of the context object
    const ur_sampler_property_t *pProps, ///< [in] specifies a list of sampler property names and their
                                         ///< corresponding values.
    ur_sampler_handle_t *phSampler       ///< [out] pointer to handle of sampler object created
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRetain(
    ur_sampler_handle_t hSampler ///< [in] handle of the sampler object to get access
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRelease(
    ur_sampler_handle_t hSampler ///< [in] handle of the sampler object to release
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_SAMPLER_INFO_LOD_MAX < propName`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPropValue`
///         + `NULL == pPropSizeRet`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetInfo(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler object
    ur_sampler_info_t propName,   ///< [in] name of the sampler property to query
    size_t propValueSize,         ///< [in] size in bytes of the sampler property value provided
    void *pPropValue,             ///< [out] value of the sampler property
    size_t *pPropSizeRet          ///< [out] size in bytes returned in sampler property value
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeSampler`
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler,       ///< [in] handle of the sampler.
    ur_native_handle_t *phNativeSampler ///< [out] a pointer to the native handle of the sampler.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeSampler`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phSampler`
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerCreateWithNativeHandle(
    ur_native_handle_t hNativeSampler, ///< [in] the native handle of the sampler.
    ur_context_handle_t hContext,      ///< [in] handle of the context object
    ur_sampler_handle_t *phSampler     ///< [out] pointer to the handle of the sampler object created.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region usm
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory property flags
typedef uint32_t ur_usm_mem_flags_t;
typedef enum ur_usm_mem_flag_t {
    UR_USM_MEM_FLAG_BIAS_CACHED = UR_BIT(0),              ///< Allocation should be cached
    UR_USM_MEM_FLAG_BIAS_UNCACHED = UR_BIT(1),            ///< Allocation should not be cached
    UR_USM_MEM_FLAG_WRITE_COMBINED = UR_BIT(2),           ///< Memory should be allocated write-combined (WC)
    UR_USM_MEM_FLAG_INITIAL_PLACEMENT_DEVICE = UR_BIT(3), ///< Optimize shared allocation for first access on the device
    UR_USM_MEM_FLAG_INITIAL_PLACEMENT_HOST = UR_BIT(4),   ///< Optimize shared allocation for first access on the host
    UR_USM_MEM_FLAG_DEVICE_READ_ONLY = UR_BIT(5),         ///< Memory is only possibly modified from the host, but read-only in all
                                                          ///< device code
    /// @cond
    UR_USM_MEM_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_mem_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory property flags
typedef uint32_t ur_usm_pool_flags_t;
typedef enum ur_usm_pool_flag_t {
    UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK = UR_BIT(0), ///< All coarse-grain allocations (allocations from the driver) will be
                                                        ///< zero-initialized.
    /// @cond
    UR_USM_POOL_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_pool_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocation type
typedef enum ur_usm_type_t {
    UR_USM_TYPE_UNKNOWN = 0, ///< Unknown USM type
    UR_USM_TYPE_HOST = 1,    ///< Host USM type
    UR_USM_TYPE_DEVICE = 2,  ///< Device USM type
    UR_USM_TYPE_SHARED = 3,  ///< Shared USM type
    /// @cond
    UR_USM_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory allocation information type
typedef enum ur_usm_alloc_info_t {
    UR_USM_ALLOC_INFO_TYPE = 0,     ///< Memory allocation type info
    UR_USM_ALLOC_INFO_BASE_PTR = 1, ///< Memory allocation base pointer info
    UR_USM_ALLOC_INFO_SIZE = 2,     ///< Memory allocation size info
    UR_USM_ALLOC_INFO_DEVICE = 3,   ///< Memory allocation device info
    /// @cond
    UR_USM_ALLOC_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_alloc_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory advice
typedef enum ur_mem_advice_t {
    UR_MEM_ADVICE_DEFAULT = 0,                  ///< The USM memory advice is default
    UR_MEM_ADVICE_SET_READ_MOSTLY = 1,          ///< Hint that memory will be read from frequently and written to rarely
    UR_MEM_ADVICE_CLEAR_READ_MOSTLY = 2,        ///< Removes the affect of ::::UR_MEM_ADVICE_SET_READ_MOSTLY
    UR_MEM_ADVICE_SET_PREFERRED_LOCATION = 3,   ///< Hint that the preferred memory location is the specified device
    UR_MEM_ADVICE_CLEAR_PREFERRED_LOCATION = 4, ///< Removes the affect of ::::UR_MEM_ADVICE_SET_PREFERRED_LOCATION
    UR_MEM_ADVICE_SET_NON_ATOMIC_MOSTLY = 5,    ///< Hints that memory will mostly be accessed non-atomically
    UR_MEM_ADVICE_CLEAR_NON_ATOMIC_MOSTLY = 6,  ///< Removes the affect of ::::UR_MEM_ADVICE_SET_NON_ATOMIC_MOSTLY
    UR_MEM_ADVICE_BIAS_CACHED = 7,              ///< Hints that memory should be cached
    UR_MEM_ADVICE_BIAS_UNCACHED = 8,            ///< Hints that memory should be not be cached
    /// @cond
    UR_MEM_ADVICE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_mem_advice_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of USM pool
typedef struct ur_usm_pool_handle_t_ *ur_usm_pool_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocation descriptor type
typedef struct ur_usm_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    ur_usm_mem_flags_t flags;  ///< [in] memory allocation flags
    ur_mem_advice_t hints;     ///< [in] Memory advice hints

} ur_usm_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM pool descriptor type
typedef struct ur_usm_pool_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_POOL_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    ur_usm_pool_flags_t flags; ///< [in] memory allocation flags

} ur_usm_pool_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM pool limits descriptor type
typedef struct ur_usm_pool_limits_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    size_t maxPoolSize;        ///< [in] Maximum size of a memory pool
    size_t maxPoolableSize;    ///< [in] Allocations up to this limit will be subject to pooling
    size_t capacity;           ///< [in] When pooling, each bucket will hold a max of 4 unfreed slabs
    size_t slabMinSize;        ///< [in] Minimum allocation size that will be requested from the driver

} ur_usm_pool_limits_desc_t;

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
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::UR_DEVICE_INFO_USM_HOST_SUPPORT is false.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `align != 0 && ((align & (align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_desc_t *pUSMDesc,      ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,    ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                  ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,               ///< [in] alignment of the USM memory object
                                  ///< Must be zero or a power of 2.
                                  ///< Must be equal to or smaller than the size of the largest data type
                                  ///< supported by `hDevice`.
    void **ppMem                  ///< [out] pointer to USM host memory object
);

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
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::UR_DEVICE_INFO_USM_HOST_SUPPORT is false.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `align != 0 && ((align & (align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_usm_desc_t *pUSMDesc,      ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,    ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                  ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,               ///< [in] alignment of the USM memory object
                                  ///< Must be zero or a power of 2.
                                  ///< Must be equal to or smaller than the size of the largest data type
                                  ///< supported by `hDevice`.
    void **ppMem                  ///< [out] pointer to USM device memory object
);

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
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `align != 0 && ((align & (align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If `UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT` and `UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT` are both false.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_usm_desc_t *pUSMDesc,      ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,    ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                  ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,               ///< [in] alignment of the USM memory object.
                                  ///< Must be zero or a power of 2.
                                  ///< Must be equal to or smaller than the size of the largest data type
                                  ///< supported by `hDevice`.
    void **ppMem                  ///< [out] pointer to USM shared memory object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Free the USM memory object
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get USM memory object allocation information
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_ALLOC_INFO_DEVICE < propName`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const void *pMem,             ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t propName, ///< [in] the name of the USM allocation property to query
    size_t propValueSize,         ///< [in] size in bytes of the USM allocation property value
    void *pPropValue,             ///< [out][optional] value of the USM allocation property
    size_t *pPropValueSizeRet     ///< [out][optional] bytes returned in USM allocation property
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create USM memory pool with desired properties.
///
/// @details
///     - UR can create multiple instances of the pool depending on allocation
///       requests.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPoolDesc`
///         + `NULL == ppPool`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < pPoolDesc->flags`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolCreate(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_usm_pool_desc_t *pPoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *ppPool   ///< [out] pointer to USM memory pool
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy USM memory pool
///
/// @details
///     - All allocation belonging to the pool should be freed before calling
///       this function.
///     - This functions returns all memory reserved by the pool to the driver.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == pPool`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolDestroy(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_usm_pool_handle_t pPool    ///< [in] pointer to USM memory pool
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Program
#if !defined(__GNUC__)
#pragma region program
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Program metadata property type.
typedef enum ur_program_metadata_type_t {
    UR_PROGRAM_METADATA_TYPE_UINT32 = 0,     ///< type is a 32-bit integer.
    UR_PROGRAM_METADATA_TYPE_UINT64 = 1,     ///< type is a 64-bit integer.
    UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY = 2, ///< type is a byte array.
    UR_PROGRAM_METADATA_TYPE_STRING = 3,     ///< type is a null-terminated string.
    /// @cond
    UR_PROGRAM_METADATA_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_metadata_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program metadata value union.
typedef union ur_program_metadata_value_t {
    uint32_t data32; ///< [in] inline storage for the 32-bit data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_UINT32.
    uint64_t data64; ///< [in] inline storage for the 64-bit data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_UINT64.
    char *pString;   ///< [in] pointer to null-terminated string data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_STRING.
    void *pData;     ///< [in] pointer to binary data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY.

} ur_program_metadata_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program metadata property.
typedef struct ur_program_metadata_t {
    char *pName;                       ///< [in] null-terminated metadata name.
    ur_program_metadata_type_t type;   ///< [in] the type of metadata value.
    size_t size;                       ///< [in] size in bytes of the data pointed to by value.pData, or 0 when
                                       ///< value size is less than 64-bits and is stored directly in value.data.
    ur_program_metadata_value_t value; ///< [in] the metadata value storage.

} ur_program_metadata_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program creation properties.
typedef struct ur_program_properties_t {
    ur_structure_type_t stype;               ///< [in] type of this structure, must be
                                             ///< ::UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES
    void *pNext;                             ///< [in,out][optional] pointer to extension-specific structure
    uint32_t count;                          ///< [in] the number of entries in pMetadatas, if count is greater than
                                             ///< zero then pMetadatas must not be null.
    const ur_program_metadata_t *pMetadatas; ///< [in][optional][range(0,count)] pointer to array of metadata entries.

} ur_program_properties_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pIL`
///         + `NULL == phProgram`
///         + `NULL != pProperties && pProperties->count > 0 && NULL == pProperties->pMetadatas`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NULL != pProperties && NULL != pProperties->pMetadatas && pProperties->count == 0`
///     - ::UR_RESULT_ERROR_INVALID_BINARY
///         + If `pIL` is not a valid IL binary for devices in `hContext`.
///     - ::UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE
///         + If devices in `hContext` don't have the capability to compile an IL binary at runtime.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(
    ur_context_handle_t hContext,               ///< [in] handle of the context instance
    const void *pIL,                            ///< [in] pointer to IL binary.
    size_t length,                              ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t *pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t *phProgram              ///< [out] pointer to handle of program object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a program object from device native binary.
///
/// @details
///     - The application may call this function from simultaneous threads.
///
/// @remarks
///   _Analogues_
///     - **clCreateProgramWithBinary**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
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
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithBinary(
    ur_context_handle_t hContext,               ///< [in] handle of the context instance
    ur_device_handle_t hDevice,                 ///< [in] handle to device associated with binary.
    size_t size,                                ///< [in] size in bytes.
    const uint8_t *pBinary,                     ///< [in] pointer to binary.
    const ur_program_properties_t *pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t *phProgram              ///< [out] pointer to handle of Program object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one program, negates need for the
///        linking step.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a succesful call to this entry point, the program passed
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred when building `hProgram`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramBuild(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    const char *pOptions          ///< [in][optional] pointer to build options null-terminated string.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a succesful call to this entry point `hProgram` will contain
///       a binary of the ::UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT type for each
///       device in `hContext`.
///
/// @remarks
///   _Analogues_
///     - **clCompileProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred while compiling `hProgram`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t hProgram, ///< [in][out] handle of the program to compile.
    const char *pOptions          ///< [in][optional] pointer to build options null-terminated string.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a succesful call to this entry point the program returned in
///       `phProgram` will contain a binary of the
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPrograms`
///         + `NULL == phProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If one of the programs in `phPrograms` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_LINK_FAILURE
///         + If an error occurred while linking `phPrograms`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(
    ur_context_handle_t hContext,          ///< [in] handle of the context instance.
    uint32_t count,                        ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *pOptions,                  ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t *phProgram         ///< [out] pointer to handle of program object created.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(
    ur_program_handle_t hProgram ///< [in] handle for the Program to retain
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(
    ur_program_handle_t hProgram ///< [in] handle for the Program to release
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pFunctionName`
///         + `NULL == ppFunctionPointer`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetFunctionPointer(
    ur_device_handle_t hDevice,   ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t hProgram, ///< [in] handle of the program to search for function in.
                                  ///< The program must already be built to the specified device, or
                                  ///< otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char *pFunctionName,    ///< [in] A null-terminates string denoting the mangled function name.
    void **ppFunctionPointer      ///< [out] Returns the pointer to the function if it is found in the program.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Program object information
typedef enum ur_program_info_t {
    UR_PROGRAM_INFO_REFERENCE_COUNT = 0, ///< [uint32_t] Reference count of the program object.
                                         ///< The reference count returned should be considered immediately stale.
                                         ///< It is unsuitable for general use in applications. This feature is
                                         ///< provided for identifying memory leaks.
    UR_PROGRAM_INFO_CONTEXT = 1,         ///< Program context info.
    UR_PROGRAM_INFO_NUM_DEVICES = 2,     ///< Return number of devices associated with Program.
    UR_PROGRAM_INFO_DEVICES = 3,         ///< Return list of devices associated with Program, return type uint32_t*.
    UR_PROGRAM_INFO_SOURCE = 4,          ///< Return program source associated with Program, return type char*.
    UR_PROGRAM_INFO_BINARY_SIZES = 5,    ///< Return program binary sizes for each device, return type size_t*.
    UR_PROGRAM_INFO_BINARIES = 6,        ///< Return program binaries for all devices for this Program, return type
                                         ///< uchar*.
    UR_PROGRAM_INFO_NUM_KERNELS = 7,     ///< Number of kernels in Program, return type size_t.
    UR_PROGRAM_INFO_KERNEL_NAMES = 8,    ///< Return a null-terminated, semi-colon separated list of kernel names in
                                         ///< Program, return type char*.
    /// @cond
    UR_PROGRAM_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_info_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_INFO_KERNEL_NAMES < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    ur_program_info_t propName,   ///< [in] name of the Program property to query
    size_t propSize,              ///< [in] the size of the Program property.
    void *pProgramInfo,           ///< [in,out][optional] array of bytes of holding the program info property.
                                  ///< If propSize is not equal to or greater than the real number of bytes
                                  ///< needed to return
                                  ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                  ///< pProgramInfo is not used.
    size_t *pPropSizeRet          ///< [out][optional] pointer to the actual size in bytes of data copied to pProgramInfo.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Program object build status
typedef enum ur_program_build_status_t {
    UR_PROGRAM_BUILD_STATUS_NONE = 0,        ///< Program build status none
    UR_PROGRAM_BUILD_STATUS_ERROR = 1,       ///< Program build error
    UR_PROGRAM_BUILD_STATUS_SUCCESS = 2,     ///< Program build success
    UR_PROGRAM_BUILD_STATUS_IN_PROGRESS = 3, ///< Program build in progress
    /// @cond
    UR_PROGRAM_BUILD_STATUS_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_build_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program object binary type
typedef enum ur_program_binary_type_t {
    UR_PROGRAM_BINARY_TYPE_NONE = 0,            ///< No program binary is associated with device
    UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 1, ///< Program binary is compiled object
    UR_PROGRAM_BINARY_TYPE_LIBRARY = 2,         ///< Program binary is library object
    UR_PROGRAM_BINARY_TYPE_EXECUTABLE = 3,      ///< Program binary is executable
    /// @cond
    UR_PROGRAM_BINARY_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_binary_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Program object build information
typedef enum ur_program_build_info_t {
    UR_PROGRAM_BUILD_INFO_STATUS = 0,      ///< Program build status, return type ::ur_program_build_status_t.
    UR_PROGRAM_BUILD_INFO_OPTIONS = 1,     ///< Null-terminated options string specified by last build, compile or
                                           ///< link operation performed on the program. Return type char*.
    UR_PROGRAM_BUILD_INFO_LOG = 2,         ///< Null-terminated program build log, return type char*.
    UR_PROGRAM_BUILD_INFO_BINARY_TYPE = 3, ///< Program binary type, return type ::ur_program_binary_type_t.
    /// @cond
    UR_PROGRAM_BUILD_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_build_info_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_BUILD_INFO_BINARY_TYPE < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(
    ur_program_handle_t hProgram,     ///< [in] handle of the Program object
    ur_device_handle_t hDevice,       ///< [in] handle of the Device object
    ur_program_build_info_t propName, ///< [in] name of the Program build info to query
    size_t propSize,                  ///< [in] size of the Program build info property.
    void *pPropValue,                 ///< [in,out][optional] value of the Program build property.
                                      ///< If propSize is not equal to or greater than the real number of bytes
                                      ///< needed to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE
                                      ///< error is returned and pKernelInfo is not used.
    size_t *pPropSizeRet              ///< [out][optional] pointer to the actual size in bytes of data being
                                      ///< queried by propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Specialization constant information
typedef struct ur_specialization_constant_info_t {
    uint32_t id;        ///< [in] specialization constant Id
    size_t size;        ///< [in] size of the specialization constant value
    const void *pValue; ///< [in] pointer to the specialization constant value bytes

} ur_specialization_constant_info_t;

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
/// @details
///     - `hProgram` must have been created with the ::urProgramCreateWithIL
///       entry point.
///     - Any spec constants set with this entry point will apply only to
///       subsequent calls to ::urProgramBuild or ::urProgramCompile.
///
/// @remarks
///   _Analogues_
///     - **clSetProgramSpecializationConstant**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSpecConstants`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `count == 0`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram,                           ///< [in] handle of the Program object
    uint32_t count,                                         ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *pSpecConstants ///< [in][range(0, count)] array of specialization constant value
                                                            ///< descriptions
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeProgram`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetNativeHandle(
    ur_program_handle_t hProgram,       ///< [in] handle of the program.
    ur_native_handle_t *phNativeProgram ///< [out] a pointer to the native handle of the program.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeProgram`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phProgram`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ///< [in] the native handle of the program.
    ur_context_handle_t hContext,      ///< [in] handle of the context instance
    ur_program_handle_t *phProgram     ///< [out] pointer to the handle of the program object created.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Program
#if !defined(__GNUC__)
#pragma region kernel
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Create kernel object from a program.
///
/// @details
///     - Multiple calls to this function will return identical device handles,
///       in the same order.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pKernelName`
///         + `NULL == phKernel`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t *phKernel  ///< [out] pointer to handle of kernel object created.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    size_t argSize,             ///< [in] size of argument type
    const void *pArgValue       ///< [in] argument value represented as matching arg type.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    size_t argSize              ///< [in] size of the local buffer to be allocated by the runtime
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel object information
typedef enum ur_kernel_info_t {
    UR_KERNEL_INFO_FUNCTION_NAME = 0,   ///< Return null-terminated kernel function name, return type char*.
    UR_KERNEL_INFO_NUM_ARGS = 1,        ///< Return Kernel number of arguments.
    UR_KERNEL_INFO_REFERENCE_COUNT = 2, ///< [uint32_t] Reference count of the kernel object.
                                        ///< The reference count returned should be considered immediately stale.
                                        ///< It is unsuitable for general use in applications. This feature is
                                        ///< provided for identifying memory leaks.
    UR_KERNEL_INFO_CONTEXT = 3,         ///< Return Context object associated with Kernel.
    UR_KERNEL_INFO_PROGRAM = 4,         ///< Return Program object associated with Kernel.
    UR_KERNEL_INFO_ATTRIBUTES = 5,      ///< Return null-terminated kernel attributes string, return type char*.
    /// @cond
    UR_KERNEL_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel Work Group information
typedef enum ur_kernel_group_info_t {
    UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE = 0,                   ///< Return Work Group maximum global size, return type size_t[3]
    UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE = 1,                    ///< Return maximum Work Group size, return type size_t
    UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE = 2,            ///< Return Work Group size required by the source code, such as
                                                                 ///< __attribute__((required_work_group_size(X,Y,Z)), return type size_t[3]
    UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE = 3,                     ///< Return local memory required by the Kernel, return type size_t
    UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 4, ///< Return preferred multiple of Work Group size for launch, return type
                                                                 ///< size_t
    UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE = 5,                   ///< Return minimum amount of private memory in bytes used by each work
                                                                 ///< item in the Kernel, return type size_t
    /// @cond
    UR_KERNEL_GROUP_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_group_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel SubGroup information
typedef enum ur_kernel_sub_group_info_t {
    UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE = 0,     ///< Return maximum SubGroup size, return type uint32_t
    UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS = 1,     ///< Return maximum number of SubGroup, return type uint32_t
    UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS = 2, ///< Return number of SubGroup required by the source code, return type
                                                         ///< uint32_t
    UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL = 3,   ///< Return SubGroup size required by Intel, return type uint32_t
    /// @cond
    UR_KERNEL_SUB_GROUP_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_sub_group_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set additional Kernel execution information
typedef enum ur_kernel_exec_info_t {
    UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS = 0, ///< Kernel might access data through USM pointer, type bool_t*
    UR_KERNEL_EXEC_INFO_USM_PTRS = 1,            ///< Provide an explicit list of USM pointers that the kernel will access,
                                                 ///< type void*[].
    /// @cond
    UR_KERNEL_EXEC_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_exec_info_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_INFO_ATTRIBUTES < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_kernel_info_t propName,  ///< [in] name of the Kernel property to query
    size_t propSize,            ///< [in] the size of the Kernel property value.
    void *pKernelInfo,          ///< [in,out][optional] array of bytes holding the kernel info property.
                                ///< If propSize is not equal to or greater than the real number of bytes
                                ///< needed to return
                                ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                ///< pKernelInfo is not used.
    size_t *pPropSizeRet        ///< [out][optional] pointer to the actual size in bytes of data being
                                ///< queried by propName.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(
    ur_kernel_handle_t hKernel,      ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,      ///< [in] handle of the Device object
    ur_kernel_group_info_t propName, ///< [in] name of the work Group property to query
    size_t propSize,                 ///< [in] size of the Kernel Work Group property value
    void *pPropValue,                ///< [in,out][optional][range(0, propSize)] value of the Kernel Work Group
                                     ///< property.
    size_t *pPropSizeRet             ///< [out][optional] pointer to the actual size in bytes of data being
                                     ///< queried by propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Query SubGroup information about a Kernel object
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetSubGroupInfo(
    ur_kernel_handle_t hKernel,          ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,          ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t propName, ///< [in] name of the SubGroup property to query
    size_t propSize,                     ///< [in] size of the Kernel SubGroup property value
    void *pPropValue,                    ///< [in,out][range(0, propSize)][optional] value of the Kernel SubGroup
                                         ///< property.
    size_t *pPropSizeRet                 ///< [out][optional] pointer to the actual size in bytes of data being
                                         ///< queried by propName.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    const void *pArgValue       ///< [in][optional] SVM pointer to memory location holding the argument
                                ///< value. If null then argument value is considered null.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_EXEC_INFO_USM_PTRS < propName`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPropValue`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,     ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName, ///< [in] name of the execution attribute
    size_t propSize,                ///< [in] size in byte the attribute value
    const void *pPropValue          ///< [in][range(0, propSize)] pointer to memory location holding the
                                    ///< property value.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgSampler(
    ur_kernel_handle_t hKernel,   ///< [in] handle of the kernel object
    uint32_t argIndex,            ///< [in] argument index in range [0, num args - 1]
    ur_sampler_handle_t hArgValue ///< [in] handle of Sampler object.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex,          ///< [in] argument index in range [0, num args - 1]
    ur_mem_handle_t hArgValue   ///< [in][optional] handle of Memory object.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSpecConstants`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `count == 0`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If ::UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS query is false
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel,                             ///< [in] handle of the kernel object
    uint32_t count,                                         ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *pSpecConstants ///< [in] array of specialization constant value descriptions
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeKernel`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel,        ///< [in] handle of the kernel.
    ur_native_handle_t *phNativeKernel ///< [out] a pointer to the native handle of the kernel.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeKernel`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phKernel`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel, ///< [in] the native handle of the kernel.
    ur_context_handle_t hContext,     ///< [in] handle of the context object
    ur_kernel_handle_t *phKernel      ///< [out] pointer to the handle of the kernel object created.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region queue
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Query queue info
typedef enum ur_queue_info_t {
    UR_QUEUE_INFO_CONTEXT = 0,         ///< ::ur_queue_handle_t: context associated with this queue.
    UR_QUEUE_INFO_DEVICE = 1,          ///< ::ur_device_handle_t: device associated with this queue.
    UR_QUEUE_INFO_DEVICE_DEFAULT = 2,  ///< ::ur_queue_handle_t: the current default queue of the underlying
                                       ///< device.
    UR_QUEUE_INFO_PROPERTIES = 3,      ///< ::ur_queue_flags_t: the properties associated with
                                       ///< ::UR_QUEUE_PROPERTIES_FLAGS.
    UR_QUEUE_INFO_REFERENCE_COUNT = 4, ///< [uint32_t] Reference count of the queue object.
                                       ///< The reference count returned should be considered immediately stale.
                                       ///< It is unsuitable for general use in applications. This feature is
                                       ///< provided for identifying memory leaks.
    UR_QUEUE_INFO_SIZE = 5,            ///< uint32_t: The size of the queue
    /// @cond
    UR_QUEUE_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_queue_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue property flags
typedef uint32_t ur_queue_flags_t;
typedef enum ur_queue_flag_t {
    UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE = UR_BIT(0), ///< Enable/disable out of order execution
    UR_QUEUE_FLAG_PROFILING_ENABLE = UR_BIT(1),              ///< Enable/disable profiling
    UR_QUEUE_FLAG_ON_DEVICE = UR_BIT(2),                     ///< Is a device queue
    UR_QUEUE_FLAG_ON_DEVICE_DEFAULT = UR_BIT(3),             ///< Is the default queue for a device
    UR_QUEUE_FLAG_DISCARD_EVENTS = UR_BIT(4),                ///< Events will be discarded
    UR_QUEUE_FLAG_PRIORITY_LOW = UR_BIT(5),                  ///< Low priority queue
    UR_QUEUE_FLAG_PRIORITY_HIGH = UR_BIT(6),                 ///< High priority queue
    /// @cond
    UR_QUEUE_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_queue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue property type
typedef intptr_t ur_queue_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue Properties
typedef enum ur_queue_properties_t {
    UR_QUEUE_PROPERTIES_FLAGS = -1,         ///< [::ur_queue_flags_t]: the bitfield of queue flags
    UR_QUEUE_PROPERTIES_COMPUTE_INDEX = -2, ///< [uint32_t]: the queue index
    /// @cond
    UR_QUEUE_PROPERTIES_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_queue_properties_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_QUEUE_INFO_SIZE < propName`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetInfo(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_queue_info_t propName, ///< [in] name of the queue property to query
    size_t propValueSize,     ///< [in] size in bytes of the queue property value provided
    void *pPropValue,         ///< [out][optional] value of the queue property
    size_t *pPropSizeRet      ///< [out][optional] size in bytes returned in queue property value
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a command queue for a device in a context
///
/// @remarks
///   _Analogues_
///     - **clCreateCommandQueueWithProperties**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phQueue`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreate(
    ur_context_handle_t hContext,      ///< [in] handle of the context object
    ur_device_handle_t hDevice,        ///< [in] handle of the device object
    const ur_queue_property_t *pProps, ///< [in][optional] specifies a list of queue properties and their
                                       ///< corresponding values.
                                       ///< Each property name is immediately followed by the corresponding
                                       ///< desired value.
                                       ///< The list is terminated with a 0.
                                       ///< If a property value is not specified, then its default value will be used.
    ur_queue_handle_t *phQueue         ///< [out] pointer to handle of queue object created
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueRetain(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to get access
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueRelease(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to release
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeQueue`
UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetNativeHandle(
    ur_queue_handle_t hQueue,         ///< [in] handle of the queue.
    ur_native_handle_t *phNativeQueue ///< [out] a pointer to the native handle of the queue.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeQueue`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phQueue`
UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ///< [in] the native handle of the queue.
    ur_context_handle_t hContext,    ///< [in] handle of the context object
    ur_queue_handle_t *phQueue       ///< [out] pointer to the handle of the queue object created.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urQueueFinish(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be finished.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urQueueFlush(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be flushed.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region event
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Command type
typedef enum ur_command_t {
    UR_COMMAND_KERNEL_LAUNCH = 0,                 ///< Event created by ::urEnqueueKernelLaunch
    UR_COMMAND_EVENTS_WAIT = 1,                   ///< Event created by ::urEnqueueEventsWait
    UR_COMMAND_EVENTS_WAIT_WITH_BARRIER = 2,      ///< Event created by ::urEnqueueEventsWaitWithBarrier
    UR_COMMAND_MEM_BUFFER_READ = 3,               ///< Event created by ::urEnqueueMemBufferRead
    UR_COMMAND_MEM_BUFFER_WRITE = 4,              ///< Event created by ::urEnqueueMemBufferWrite
    UR_COMMAND_MEM_BUFFER_READ_RECT = 5,          ///< Event created by ::urEnqueueMemBufferReadRect
    UR_COMMAND_MEM_BUFFER_WRITE_RECT = 6,         ///< Event created by ::urEnqueueMemBufferWriteRect
    UR_COMMAND_MEM_BUFFER_COPY = 7,               ///< Event created by ::urEnqueueMemBufferCopy
    UR_COMMAND_MEM_BUFFER_COPY_RECT = 8,          ///< Event created by ::urEnqueueMemBufferCopyRect
    UR_COMMAND_MEM_BUFFER_FILL = 9,               ///< Event created by ::urEnqueueMemBufferFill
    UR_COMMAND_MEM_IMAGE_READ = 10,               ///< Event created by ::urEnqueueMemImageRead
    UR_COMMAND_MEM_IMAGE_WRITE = 11,              ///< Event created by ::urEnqueueMemImageWrite
    UR_COMMAND_MEM_IMAGE_COPY = 12,               ///< Event created by ::urEnqueueMemImageCopy
    UR_COMMAND_MEM_BUFFER_MAP = 14,               ///< Event created by ::urEnqueueMemBufferMap
    UR_COMMAND_MEM_UNMAP = 16,                    ///< Event created by ::urEnqueueMemUnmap
    UR_COMMAND_USM_FILL = 17,                     ///< Event created by ::urEnqueueUSMFill
    UR_COMMAND_USM_MEMCPY = 18,                   ///< Event created by ::urEnqueueUSMMemcpy
    UR_COMMAND_USM_PREFETCH = 19,                 ///< Event created by ::urEnqueueUSMPrefetch
    UR_COMMAND_USM_MEM_ADVISE = 20,               ///< Event created by ::urEnqueueUSMMemAdvise
    UR_COMMAND_USM_FILL_2D = 21,                  ///< Event created by ::urEnqueueUSMFill2D
    UR_COMMAND_USM_MEMCPY_2D = 22,                ///< Event created by ::urEnqueueUSMMemcpy2D
    UR_COMMAND_DEVICE_GLOBAL_VARIABLE_WRITE = 23, ///< Event created by ::urEnqueueDeviceGlobalVariableWrite
    UR_COMMAND_DEVICE_GLOBAL_VARIABLE_READ = 24,  ///< Event created by ::urEnqueueDeviceGlobalVariableRead
    /// @cond
    UR_COMMAND_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_command_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event Status
typedef enum ur_event_status_t {
    UR_EVENT_STATUS_COMPLETE = 0,  ///< Command is complete
    UR_EVENT_STATUS_RUNNING = 1,   ///< Command is running
    UR_EVENT_STATUS_SUBMITTED = 2, ///< Command is submitted
    UR_EVENT_STATUS_QUEUED = 3,    ///< Command is queued
    /// @cond
    UR_EVENT_STATUS_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_event_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event query information type
typedef enum ur_event_info_t {
    UR_EVENT_INFO_COMMAND_QUEUE = 0,            ///< [::ur_queue_handle_t] Command queue information of an event object
    UR_EVENT_INFO_CONTEXT = 1,                  ///< [::ur_context_handle_t] Context information of an event object
    UR_EVENT_INFO_COMMAND_TYPE = 2,             ///< [::ur_command_t] Command type information of an event object
    UR_EVENT_INFO_COMMAND_EXECUTION_STATUS = 3, ///< [::ur_event_status_t] Command execution status of an event object
    UR_EVENT_INFO_REFERENCE_COUNT = 4,          ///< [uint32_t] Reference count of the event object.
                                                ///< The reference count returned should be considered immediately stale.
                                                ///< It is unsuitable for general use in applications. This feature is
                                                ///< provided for identifying memory leaks.
    /// @cond
    UR_EVENT_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_event_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profiling query information type
typedef enum ur_profiling_info_t {
    UR_PROFILING_INFO_COMMAND_QUEUED = 0, ///< A 64-bit value of current device counter in nanoseconds when the event
                                          ///< is enqueued
    UR_PROFILING_INFO_COMMAND_SUBMIT = 1, ///< A 64-bit value of current device counter in nanoseconds when the event
                                          ///< is submitted
    UR_PROFILING_INFO_COMMAND_START = 2,  ///< A 64-bit value of current device counter in nanoseconds when the event
                                          ///< starts execution
    UR_PROFILING_INFO_COMMAND_END = 3,    ///< A 64-bit value of current device counter in nanoseconds when the event
                                          ///< has finished execution
    /// @cond
    UR_PROFILING_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_profiling_info_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EVENT_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventGetInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_event_info_t propName, ///< [in] the name of the event property to query
    size_t propValueSize,     ///< [in] size in bytes of the event property value
    void *pPropValue,         ///< [out][optional] value of the event property
    size_t *pPropValueSizeRet ///< [out][optional] bytes returned in event property
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROFILING_INFO_COMMAND_END < propName`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventGetProfilingInfo(
    ur_event_handle_t hEvent,     ///< [in] handle of the event object
    ur_profiling_info_t propName, ///< [in] the name of the profiling property to query
    size_t propValueSize,         ///< [in] size in bytes of the profiling property value
    void *pPropValue,             ///< [out][optional] value of the profiling property
    size_t *pPropValueSizeRet     ///< [out][optional] pointer to the actual size in bytes returned in
                                  ///< propValue
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEventWaitList`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(
    uint32_t numEvents,                      ///< [in] number of events in the event list
    const ur_event_handle_t *phEventWaitList ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                                             ///< completion
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventRetain(
    ur_event_handle_t hEvent ///< [in] handle of the event object
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventRelease(
    ur_event_handle_t hEvent ///< [in] handle of the event object
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeEvent`
UR_APIEXPORT ur_result_t UR_APICALL
urEventGetNativeHandle(
    ur_event_handle_t hEvent,         ///< [in] handle of the event.
    ur_native_handle_t *phNativeEvent ///< [out] a pointer to the native handle of the event.
);

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeEvent`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEvent`
UR_APIEXPORT ur_result_t UR_APICALL
urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ///< [in] the native handle of the event.
    ur_context_handle_t hContext,    ///< [in] handle of the context object
    ur_event_handle_t *phEvent       ///< [out] pointer to the handle of the event object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Event states for all events.
typedef enum ur_execution_info_t {
    UR_EXECUTION_INFO_EXECUTION_INFO_COMPLETE = 0,  ///< Indicates that the event has completed.
    UR_EXECUTION_INFO_EXECUTION_INFO_RUNNING = 1,   ///< Indicates that the device has started processing this event.
    UR_EXECUTION_INFO_EXECUTION_INFO_SUBMITTED = 2, ///< Indicates that the event has been submitted by the host to the device.
    UR_EXECUTION_INFO_EXECUTION_INFO_QUEUED = 3,    ///< Indicates that the event has been queued, this is the initial state of
                                                    ///< events.
    /// @cond
    UR_EXECUTION_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_execution_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event callback function that can be registered by the application.
typedef void (*ur_event_callback_t)(
    ur_event_handle_t hEvent,       ///< [in] handle to event
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    void *pUserData                 ///< [in][out] pointer to data to be passed to callback
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Register a user callback function for a specific command execution
///        status.
///
/// @details
///     - The registered callback function will be called when the execution
///       status of command associated with event changes to an execution status
///       equal to or past the status specified by command_exec_status.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXECUTION_INFO_EXECUTION_INFO_QUEUED < execStatus`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnNotify`
UR_APIEXPORT ur_result_t UR_APICALL
urEventSetCallback(
    ur_event_handle_t hEvent,       ///< [in] handle of the event object
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    ur_event_callback_t pfnNotify,  ///< [in] execution status of the event
    void *pUserData                 ///< [in][out][optional] pointer to data to be passed to callback.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel'oneAPI' Unified Rutime function registry
#if !defined(__GNUC__)
#pragma region registry
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Defines unique stable identifiers for all functions
typedef enum ur_function_t {
    UR_FUNCTION_CONTEXT_CREATE = 1,                        ///< Enumerator for ::urContextCreate
    UR_FUNCTION_CONTEXT_RETAIN = 2,                        ///< Enumerator for ::urContextRetain
    UR_FUNCTION_CONTEXT_RELEASE = 3,                       ///< Enumerator for ::urContextRelease
    UR_FUNCTION_CONTEXT_GET_INFO = 4,                      ///< Enumerator for ::urContextGetInfo
    UR_FUNCTION_CONTEXT_GET_NATIVE_HANDLE = 5,             ///< Enumerator for ::urContextGetNativeHandle
    UR_FUNCTION_CONTEXT_CREATE_WITH_NATIVE_HANDLE = 6,     ///< Enumerator for ::urContextCreateWithNativeHandle
    UR_FUNCTION_CONTEXT_SET_EXTENDED_DELETER = 7,          ///< Enumerator for ::urContextSetExtendedDeleter
    UR_FUNCTION_DEVICE_GET = 8,                            ///< Enumerator for ::urDeviceGet
    UR_FUNCTION_DEVICE_GET_INFO = 9,                       ///< Enumerator for ::urDeviceGetInfo
    UR_FUNCTION_DEVICE_RETAIN = 10,                        ///< Enumerator for ::urDeviceRetain
    UR_FUNCTION_DEVICE_RELEASE = 11,                       ///< Enumerator for ::urDeviceRelease
    UR_FUNCTION_DEVICE_PARTITION = 12,                     ///< Enumerator for ::urDevicePartition
    UR_FUNCTION_DEVICE_SELECT_BINARY = 13,                 ///< Enumerator for ::urDeviceSelectBinary
    UR_FUNCTION_DEVICE_GET_NATIVE_HANDLE = 14,             ///< Enumerator for ::urDeviceGetNativeHandle
    UR_FUNCTION_DEVICE_CREATE_WITH_NATIVE_HANDLE = 15,     ///< Enumerator for ::urDeviceCreateWithNativeHandle
    UR_FUNCTION_DEVICE_GET_GLOBAL_TIMESTAMPS = 16,         ///< Enumerator for ::urDeviceGetGlobalTimestamps
    UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH = 17,                ///< Enumerator for ::urEnqueueKernelLaunch
    UR_FUNCTION_ENQUEUE_EVENTS_WAIT = 18,                  ///< Enumerator for ::urEnqueueEventsWait
    UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER = 19,     ///< Enumerator for ::urEnqueueEventsWaitWithBarrier
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ = 20,              ///< Enumerator for ::urEnqueueMemBufferRead
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE = 21,             ///< Enumerator for ::urEnqueueMemBufferWrite
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ_RECT = 22,         ///< Enumerator for ::urEnqueueMemBufferReadRect
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE_RECT = 23,        ///< Enumerator for ::urEnqueueMemBufferWriteRect
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY = 24,              ///< Enumerator for ::urEnqueueMemBufferCopy
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY_RECT = 25,         ///< Enumerator for ::urEnqueueMemBufferCopyRect
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_FILL = 26,              ///< Enumerator for ::urEnqueueMemBufferFill
    UR_FUNCTION_ENQUEUE_MEM_IMAGE_READ = 27,               ///< Enumerator for ::urEnqueueMemImageRead
    UR_FUNCTION_ENQUEUE_MEM_IMAGE_WRITE = 28,              ///< Enumerator for ::urEnqueueMemImageWrite
    UR_FUNCTION_ENQUEUE_MEM_IMAGE_COPY = 29,               ///< Enumerator for ::urEnqueueMemImageCopy
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_MAP = 30,               ///< Enumerator for ::urEnqueueMemBufferMap
    UR_FUNCTION_ENQUEUE_MEM_UNMAP = 31,                    ///< Enumerator for ::urEnqueueMemUnmap
    UR_FUNCTION_ENQUEUE_USM_FILL = 32,                     ///< Enumerator for ::urEnqueueUSMFill
    UR_FUNCTION_ENQUEUE_USM_MEMCPY = 33,                   ///< Enumerator for ::urEnqueueUSMMemcpy
    UR_FUNCTION_ENQUEUE_USM_PREFETCH = 34,                 ///< Enumerator for ::urEnqueueUSMPrefetch
    UR_FUNCTION_ENQUEUE_USM_MEM_ADVISE = 35,               ///< Enumerator for ::urEnqueueUSMMemAdvise
    UR_FUNCTION_ENQUEUE_USM_FILL2_D = 36,                  ///< Enumerator for ::urEnqueueUSMFill2D
    UR_FUNCTION_ENQUEUE_USM_MEMCPY2_D = 37,                ///< Enumerator for ::urEnqueueUSMMemcpy2D
    UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_WRITE = 38, ///< Enumerator for ::urEnqueueDeviceGlobalVariableWrite
    UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_READ = 39,  ///< Enumerator for ::urEnqueueDeviceGlobalVariableRead
    UR_FUNCTION_EVENT_GET_INFO = 40,                       ///< Enumerator for ::urEventGetInfo
    UR_FUNCTION_EVENT_GET_PROFILING_INFO = 41,             ///< Enumerator for ::urEventGetProfilingInfo
    UR_FUNCTION_EVENT_WAIT = 42,                           ///< Enumerator for ::urEventWait
    UR_FUNCTION_EVENT_RETAIN = 43,                         ///< Enumerator for ::urEventRetain
    UR_FUNCTION_EVENT_RELEASE = 44,                        ///< Enumerator for ::urEventRelease
    UR_FUNCTION_EVENT_GET_NATIVE_HANDLE = 45,              ///< Enumerator for ::urEventGetNativeHandle
    UR_FUNCTION_EVENT_CREATE_WITH_NATIVE_HANDLE = 46,      ///< Enumerator for ::urEventCreateWithNativeHandle
    UR_FUNCTION_EVENT_SET_CALLBACK = 47,                   ///< Enumerator for ::urEventSetCallback
    UR_FUNCTION_KERNEL_CREATE = 48,                        ///< Enumerator for ::urKernelCreate
    UR_FUNCTION_KERNEL_SET_ARG_VALUE = 49,                 ///< Enumerator for ::urKernelSetArgValue
    UR_FUNCTION_KERNEL_SET_ARG_LOCAL = 50,                 ///< Enumerator for ::urKernelSetArgLocal
    UR_FUNCTION_KERNEL_GET_INFO = 51,                      ///< Enumerator for ::urKernelGetInfo
    UR_FUNCTION_KERNEL_GET_GROUP_INFO = 52,                ///< Enumerator for ::urKernelGetGroupInfo
    UR_FUNCTION_KERNEL_GET_SUB_GROUP_INFO = 53,            ///< Enumerator for ::urKernelGetSubGroupInfo
    UR_FUNCTION_KERNEL_RETAIN = 54,                        ///< Enumerator for ::urKernelRetain
    UR_FUNCTION_KERNEL_RELEASE = 55,                       ///< Enumerator for ::urKernelRelease
    UR_FUNCTION_KERNEL_SET_ARG_POINTER = 56,               ///< Enumerator for ::urKernelSetArgPointer
    UR_FUNCTION_KERNEL_SET_EXEC_INFO = 57,                 ///< Enumerator for ::urKernelSetExecInfo
    UR_FUNCTION_KERNEL_SET_ARG_SAMPLER = 58,               ///< Enumerator for ::urKernelSetArgSampler
    UR_FUNCTION_KERNEL_SET_ARG_MEM_OBJ = 59,               ///< Enumerator for ::urKernelSetArgMemObj
    UR_FUNCTION_KERNEL_SET_SPECIALIZATION_CONSTANTS = 60,  ///< Enumerator for ::urKernelSetSpecializationConstants
    UR_FUNCTION_KERNEL_GET_NATIVE_HANDLE = 61,             ///< Enumerator for ::urKernelGetNativeHandle
    UR_FUNCTION_KERNEL_CREATE_WITH_NATIVE_HANDLE = 62,     ///< Enumerator for ::urKernelCreateWithNativeHandle
    UR_FUNCTION_MEM_IMAGE_CREATE = 63,                     ///< Enumerator for ::urMemImageCreate
    UR_FUNCTION_MEM_BUFFER_CREATE = 64,                    ///< Enumerator for ::urMemBufferCreate
    UR_FUNCTION_MEM_RETAIN = 65,                           ///< Enumerator for ::urMemRetain
    UR_FUNCTION_MEM_RELEASE = 66,                          ///< Enumerator for ::urMemRelease
    UR_FUNCTION_MEM_BUFFER_PARTITION = 67,                 ///< Enumerator for ::urMemBufferPartition
    UR_FUNCTION_MEM_GET_NATIVE_HANDLE = 68,                ///< Enumerator for ::urMemGetNativeHandle
    UR_FUNCTION_MEM_CREATE_WITH_NATIVE_HANDLE = 69,        ///< Enumerator for ::urMemCreateWithNativeHandle
    UR_FUNCTION_MEM_GET_INFO = 70,                         ///< Enumerator for ::urMemGetInfo
    UR_FUNCTION_MEM_IMAGE_GET_INFO = 71,                   ///< Enumerator for ::urMemImageGetInfo
    UR_FUNCTION_PLATFORM_GET = 72,                         ///< Enumerator for ::urPlatformGet
    UR_FUNCTION_PLATFORM_GET_INFO = 73,                    ///< Enumerator for ::urPlatformGetInfo
    UR_FUNCTION_PLATFORM_GET_API_VERSION = 74,             ///< Enumerator for ::urPlatformGetApiVersion
    UR_FUNCTION_PLATFORM_GET_NATIVE_HANDLE = 75,           ///< Enumerator for ::urPlatformGetNativeHandle
    UR_FUNCTION_PLATFORM_CREATE_WITH_NATIVE_HANDLE = 76,   ///< Enumerator for ::urPlatformCreateWithNativeHandle
    UR_FUNCTION_GET_LAST_RESULT = 77,                      ///< Enumerator for ::urGetLastResult
    UR_FUNCTION_PROGRAM_CREATE_WITH_IL = 78,               ///< Enumerator for ::urProgramCreateWithIL
    UR_FUNCTION_PROGRAM_CREATE_WITH_BINARY = 79,           ///< Enumerator for ::urProgramCreateWithBinary
    UR_FUNCTION_PROGRAM_BUILD = 80,                        ///< Enumerator for ::urProgramBuild
    UR_FUNCTION_PROGRAM_COMPILE = 81,                      ///< Enumerator for ::urProgramCompile
    UR_FUNCTION_PROGRAM_LINK = 82,                         ///< Enumerator for ::urProgramLink
    UR_FUNCTION_PROGRAM_RETAIN = 83,                       ///< Enumerator for ::urProgramRetain
    UR_FUNCTION_PROGRAM_RELEASE = 84,                      ///< Enumerator for ::urProgramRelease
    UR_FUNCTION_PROGRAM_GET_FUNCTION_POINTER = 85,         ///< Enumerator for ::urProgramGetFunctionPointer
    UR_FUNCTION_PROGRAM_GET_INFO = 86,                     ///< Enumerator for ::urProgramGetInfo
    UR_FUNCTION_PROGRAM_GET_BUILD_INFO = 87,               ///< Enumerator for ::urProgramGetBuildInfo
    UR_FUNCTION_PROGRAM_SET_SPECIALIZATION_CONSTANTS = 88, ///< Enumerator for ::urProgramSetSpecializationConstants
    UR_FUNCTION_PROGRAM_GET_NATIVE_HANDLE = 89,            ///< Enumerator for ::urProgramGetNativeHandle
    UR_FUNCTION_PROGRAM_CREATE_WITH_NATIVE_HANDLE = 90,    ///< Enumerator for ::urProgramCreateWithNativeHandle
    UR_FUNCTION_QUEUE_GET_INFO = 91,                       ///< Enumerator for ::urQueueGetInfo
    UR_FUNCTION_QUEUE_CREATE = 92,                         ///< Enumerator for ::urQueueCreate
    UR_FUNCTION_QUEUE_RETAIN = 93,                         ///< Enumerator for ::urQueueRetain
    UR_FUNCTION_QUEUE_RELEASE = 94,                        ///< Enumerator for ::urQueueRelease
    UR_FUNCTION_QUEUE_GET_NATIVE_HANDLE = 95,              ///< Enumerator for ::urQueueGetNativeHandle
    UR_FUNCTION_QUEUE_CREATE_WITH_NATIVE_HANDLE = 96,      ///< Enumerator for ::urQueueCreateWithNativeHandle
    UR_FUNCTION_QUEUE_FINISH = 97,                         ///< Enumerator for ::urQueueFinish
    UR_FUNCTION_QUEUE_FLUSH = 98,                          ///< Enumerator for ::urQueueFlush
    UR_FUNCTION_INIT = 99,                                 ///< Enumerator for ::urInit
    UR_FUNCTION_TEAR_DOWN = 100,                           ///< Enumerator for ::urTearDown
    UR_FUNCTION_SAMPLER_CREATE = 101,                      ///< Enumerator for ::urSamplerCreate
    UR_FUNCTION_SAMPLER_RETAIN = 102,                      ///< Enumerator for ::urSamplerRetain
    UR_FUNCTION_SAMPLER_RELEASE = 103,                     ///< Enumerator for ::urSamplerRelease
    UR_FUNCTION_SAMPLER_GET_INFO = 104,                    ///< Enumerator for ::urSamplerGetInfo
    UR_FUNCTION_SAMPLER_GET_NATIVE_HANDLE = 105,           ///< Enumerator for ::urSamplerGetNativeHandle
    UR_FUNCTION_SAMPLER_CREATE_WITH_NATIVE_HANDLE = 106,   ///< Enumerator for ::urSamplerCreateWithNativeHandle
    UR_FUNCTION_USM_HOST_ALLOC = 107,                      ///< Enumerator for ::urUSMHostAlloc
    UR_FUNCTION_USM_DEVICE_ALLOC = 108,                    ///< Enumerator for ::urUSMDeviceAlloc
    UR_FUNCTION_USM_SHARED_ALLOC = 109,                    ///< Enumerator for ::urUSMSharedAlloc
    UR_FUNCTION_USM_FREE = 110,                            ///< Enumerator for ::urUSMFree
    UR_FUNCTION_USM_GET_MEM_ALLOC_INFO = 111,              ///< Enumerator for ::urUSMGetMemAllocInfo
    UR_FUNCTION_USM_POOL_CREATE = 112,                     ///< Enumerator for ::urUSMPoolCreate
    UR_FUNCTION_USM_POOL_DESTROY = 113,                    ///< Enumerator for ::urUSMPoolDestroy
    /// @cond
    UR_FUNCTION_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_function_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region enqueue
#endif
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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel,               ///< [in] handle of the kernel object
    uint32_t workDim,                         ///< [in] number of dimensions, from 1 to 3, to specify the global and
                                              ///< work-group work-items
    const size_t *pGlobalWorkOffset,          ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkSize,            ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< number of global work-items in workDim that will execute the kernel
                                              ///< function
    const size_t *pLocalWorkSize,             ///< [in][optional] pointer to an array of workDim unsigned values that
                                              ///< specify the number of local work-items forming a work-group that will
                                              ///< execute the kernel function.
                                              ///< If nullptr, the runtime implementation will choose the work-group
                                              ///< size.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueEventsWait(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                              ///< previously enqueued commands
                                              ///< must be complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                              ///< previously enqueued commands
                                              ///< must be complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                            ///< [in] offset in bytes in the buffer object
    size_t size,                              ///< [in] size in bytes of data being read
    void *pDst,                               ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                            ///< [in] offset in bytes in the buffer object
    size_t size,                              ///< [in] size in bytes of data being written
    const void *pSrc,                         ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
///         + `bufferSlicePitch != 0 && bufferSlicePitch < region.height * bufferRowPitch`
///         + `bufferSlicePitch != 0 && bufferSlicePitch % bufferRowPitch != 0`
///         + `hostSlicePitch != 0 && hostSlicePitch < region.height * hostRowPitch`
///         + `hostSlicePitch != 0 && hostSlicePitch % hostRowPitch != 0`
///         + If the combination of `bufferOrigin`, `region`, `bufferRowPitch`, and `bufferSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin,            ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,              ///< [in] 3D offset in the host region
    ur_rect_region_t region,                  ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                    ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                  ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t hostRowPitch,                      ///< [in] length of each row in bytes in the host memory region pointed by
                                              ///< dst
    size_t hostSlicePitch,                    ///< [in] length of each 2D slice in bytes in the host memory region
                                              ///< pointed by dst
    void *pDst,                               ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
///         + `bufferSlicePitch != 0 && bufferSlicePitch < region.height * bufferRowPitch`
///         + `bufferSlicePitch != 0 && bufferSlicePitch % bufferRowPitch != 0`
///         + `hostSlicePitch != 0 && hostSlicePitch < region.height * hostRowPitch`
///         + `hostSlicePitch != 0 && hostSlicePitch % hostRowPitch != 0`
///         + If the combination of `bufferOrigin`, `region`, `bufferRowPitch`, and `bufferSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin,            ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,              ///< [in] 3D offset in the host region
    ur_rect_region_t region,                  ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                    ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                  ///< [in] length of each 2D slice in bytes in the buffer object being
                                              ///< written
    size_t hostRowPitch,                      ///< [in] length of each row in bytes in the host memory region pointed by
                                              ///< src
    size_t hostSlicePitch,                    ///< [in] length of each 2D slice in bytes in the host memory region
                                              ///< pointed by src
    void *pSrc,                               ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,               ///< [in] handle of the src buffer object
    ur_mem_handle_t hBufferDst,               ///< [in] handle of the dest buffer object
    size_t srcOffset,                         ///< [in] offset into hBufferSrc to begin copying from
    size_t dstOffset,                         ///< [in] offset info hBufferDst to begin copying into
    size_t size,                              ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
///         + `srcRowPitch != 0 && srcRowPitch < region.height`
///         + `dstRowPitch != 0 && dstRowPitch < region.height`
///         + `srcSlicePitch != 0 && srcSlicePitch < region.height * srcRowPitch`
///         + `srcSlicePitch != 0 && srcSlicePitch % srcRowPitch != 0`
///         + `dstSlicePitch != 0 && dstSlicePitch < region.height * dstRowPitch`
///         + `dstSlicePitch != 0 && dstSlicePitch % dstRowPitch != 0`
///         + If the combination of `srcOrigin`, `region`, `srcRowPitch`, and `srcSlicePitch` results in an out-of-bounds access.
///         + If the combination of `dstOrigin`, `region`, `dstRowPitch`, and `dstSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,               ///< [in] handle of the source buffer object
    ur_mem_handle_t hBufferDst,               ///< [in] handle of the dest buffer object
    ur_rect_offset_t srcOrigin,               ///< [in] 3D offset in the source buffer
    ur_rect_offset_t dstOrigin,               ///< [in] 3D offset in the destination buffer
    ur_rect_region_t region,                  ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t srcRowPitch,                       ///< [in] length of each row in bytes in the source buffer object
    size_t srcSlicePitch,                     ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t dstRowPitch,                       ///< [in] length of each row in bytes in the destination buffer object
    size_t dstSlicePitch,                     ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    const void *pPattern,                     ///< [in] pointer to the fill pattern
    size_t patternSize,                       ///< [in] size in bytes of the pattern
    size_t offset,                            ///< [in] offset into the buffer
    size_t size,                              ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemImageRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                   ///< [in] handle of the image object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                  ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    size_t rowPitch,                          ///< [in] length of each row in bytes
    size_t slicePitch,                        ///< [in] length of each 2D slice of the 3D image
    void *pDst,                               ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                   ///< [in] handle of the image object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                  ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    size_t rowPitch,                          ///< [in] length of each row in bytes
    size_t slicePitch,                        ///< [in] length of each 2D slice of the 3D image
    void *pSrc,                               ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

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
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImageSrc,                ///< [in] handle of the src image object
    ur_mem_handle_t hImageDst,                ///< [in] handle of the dest image object
    ur_rect_offset_t srcOrigin,               ///< [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
                                              ///< image
    ur_rect_offset_t dstOrigin,               ///< [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
                                              ///< or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Map flags
typedef uint32_t ur_map_flags_t;
typedef enum ur_map_flag_t {
    UR_MAP_FLAG_READ = UR_BIT(0),  ///< Map for read access
    UR_MAP_FLAG_WRITE = UR_BIT(1), ///< Map for write access
    /// @cond
    UR_MAP_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_map_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Map flags
typedef uint32_t ur_usm_migration_flags_t;
typedef enum ur_usm_migration_flag_t {
    UR_USM_MIGRATION_FLAG_DEFAULT = UR_BIT(0), ///< Default migration TODO: Add more enums!
    /// @cond
    UR_USM_MIGRATION_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_migration_flag_t;

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < mapFlags`
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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object
    bool blockingMap,                         ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t mapFlags,                  ///< [in] flags for read, write, readwrite mapping
    size_t offset,                            ///< [in] offset in bytes of the buffer region being mapped
    size_t size,                              ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent,               ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
    void **ppRetMap                           ///< [out] return mapped pointer.  TODO: move it before
                                              ///< numEventsInWaitList?
);

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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemUnmap(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hMem,                     ///< [in] handle of the memory (buffer or image) object
    void *pMappedPtr,                         ///< [in] mapped host address
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ptr`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + `size % patternSize != 0`
///         + `patternSize == 0`
///         + `patternSize > size`
///         + `patternSize != 0 && ((patternSize & (patternSize - 1)) != 0)`
///         + If `size` is higher than the allocation size of `ptr`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMFill(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    void *ptr,                                ///< [in] pointer to USM memory object
    size_t patternSize,                       ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                                              ///< than or equal to width.
    const void *pPattern,                     ///< [in] pointer with the bytes of the pattern to set.
    size_t size,                              ///< [in] size in bytes to be set. Must be a multiple of patternSize.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy USM memory
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    bool blocking,                            ///< [in] blocking or non-blocking copy
    void *pDst,                               ///< [in] pointer to the destination USM memory object
    const void *pSrc,                         ///< [in] pointer to the source USM memory object
    size_t size,                              ///< [in] size in bytes to be copied
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to prefetch USM memory
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < flags`
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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    const void *pMem,                         ///< [in] pointer to the USM memory object
    size_t size,                              ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t flags,           ///< [in] USM prefetch flags
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to set USM memory advice
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_ADVICE_BIAS_UNCACHED < advice`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMMemAdvise(
    ur_queue_handle_t hQueue,  ///< [in] handle of the queue object
    const void *pMem,          ///< [in] pointer to the USM memory object
    size_t size,               ///< [in] size in bytes to be advised
    ur_mem_advice_t advice,    ///< [in] USM memory advice
    ur_event_handle_t *phEvent ///< [out][optional] return an event object that identifies this particular
                               ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill 2D USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `pitch == 0`
///         + `pitch < width`
///         + `width == 0`
///         + `width % patternSize != 0`
///         + `height == 0`
///         + `patternSize == 0`
///         + `patternSize > width`
///         + `patternSize != 0 && ((patternSize & (patternSize - 1)) != 0)`
///         + If `pitch * height` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    void *pMem,                               ///< [in] pointer to memory to be filled.
    size_t pitch,                             ///< [in] the total width of the destination memory including padding.
    size_t patternSize,                       ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                                              ///< than or equal to width.
    const void *pPattern,                     ///< [in] pointer with the bytes of the pattern to set.
    size_t width,                             ///< [in] the width in bytes of each row to fill. Must be a multiple of
                                              ///< patternSize.
    size_t height,                            ///< [in] the height of the columns to fill.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy 2D USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    bool blocking,                            ///< [in] indicates if this operation should block the host.
    void *pDst,                               ///< [in] pointer to memory where data will be copied.
    size_t dstPitch,                          ///< [in] the total width of the source memory including padding.
    const void *pSrc,                         ///< [in] pointer to memory to be copied.
    size_t srcPitch,                          ///< [in] the total width of the source memory including padding.
    size_t width,                             ///< [in] the width in bytes of each row to be copied.
    size_t height,                            ///< [in] the height of columns to be copied.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write data from the host to device global
///        variable.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    ur_program_handle_t hProgram,             ///< [in] handle of the program containing the device global variable.
    const char *name,                         ///< [in] the unique identifier for the device global variable.
    bool blockingWrite,                       ///< [in] indicates if this operation should block.
    size_t count,                             ///< [in] the number of bytes to copy.
    size_t offset,                            ///< [in] the byte offset into the device global variable to start copying.
    const void *pSrc,                         ///< [in] pointer to where the data must be copied from.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read data from a device global variable to the
///        host.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
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
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    ur_program_handle_t hProgram,             ///< [in] handle of the program containing the device global variable.
    const char *name,                         ///< [in] the unique identifier for the device global variable.
    bool blockingRead,                        ///< [in] indicates if this operation should block.
    size_t count,                             ///< [in] the number of bytes to copy.
    size_t offset,                            ///< [in] the byte offset into the device global variable to start copying.
    void *pDst,                               ///< [in] pointer to where the data must be copied to.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime API Callbacks
#if !defined(__GNUC__)
#pragma region callbacks
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urPlatformGet
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_params_t {
    uint32_t *pNumEntries;
    ur_platform_handle_t **pphPlatforms;
    uint32_t **ppNumPlatforms;
} ur_platform_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urPlatformGet
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnPlatformGetCb_t)(
    ur_platform_get_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urPlatformGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_info_params_t {
    ur_platform_handle_t *phPlatform;
    ur_platform_info_t *pPlatformInfoType;
    size_t *pSize;
    void **ppPlatformInfo;
    size_t **ppSizeRet;
} ur_platform_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urPlatformGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnPlatformGetInfoCb_t)(
    ur_platform_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urPlatformGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_native_handle_params_t {
    ur_platform_handle_t *phPlatform;
    ur_native_handle_t **pphNativePlatform;
} ur_platform_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urPlatformGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnPlatformGetNativeHandleCb_t)(
    ur_platform_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urPlatformCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_create_with_native_handle_params_t {
    ur_native_handle_t *phNativePlatform;
    ur_platform_handle_t **pphPlatform;
} ur_platform_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urPlatformCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnPlatformCreateWithNativeHandleCb_t)(
    ur_platform_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urPlatformGetApiVersion
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_api_version_params_t {
    ur_platform_handle_t *phDriver;
    ur_api_version_t **ppVersion;
} ur_platform_get_api_version_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urPlatformGetApiVersion
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnPlatformGetApiVersionCb_t)(
    ur_platform_get_api_version_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Platform callback functions pointers
typedef struct ur_platform_callbacks_t {
    ur_pfnPlatformGetCb_t pfnGetCb;
    ur_pfnPlatformGetInfoCb_t pfnGetInfoCb;
    ur_pfnPlatformGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnPlatformCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
    ur_pfnPlatformGetApiVersionCb_t pfnGetApiVersionCb;
} ur_platform_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urContextCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_create_params_t {
    uint32_t *pDeviceCount;
    const ur_device_handle_t **pphDevices;
    const ur_context_properties_t **ppProperties;
    ur_context_handle_t **pphContext;
} ur_context_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urContextCreate
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnContextCreateCb_t)(
    ur_context_create_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urContextRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_retain_params_t {
    ur_context_handle_t *phContext;
} ur_context_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urContextRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnContextRetainCb_t)(
    ur_context_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urContextRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_release_params_t {
    ur_context_handle_t *phContext;
} ur_context_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urContextRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnContextReleaseCb_t)(
    ur_context_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urContextGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_get_info_params_t {
    ur_context_handle_t *phContext;
    ur_context_info_t *pContextInfoType;
    size_t *ppropSize;
    void **ppContextInfo;
    size_t **ppPropSizeRet;
} ur_context_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urContextGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnContextGetInfoCb_t)(
    ur_context_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urContextGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_get_native_handle_params_t {
    ur_context_handle_t *phContext;
    ur_native_handle_t **pphNativeContext;
} ur_context_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urContextGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnContextGetNativeHandleCb_t)(
    ur_context_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urContextCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeContext;
    ur_context_handle_t **pphContext;
} ur_context_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urContextCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnContextCreateWithNativeHandleCb_t)(
    ur_context_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urContextSetExtendedDeleter
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_set_extended_deleter_params_t {
    ur_context_handle_t *phContext;
    ur_context_extended_deleter_t *ppfnDeleter;
    void **ppUserData;
} ur_context_set_extended_deleter_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urContextSetExtendedDeleter
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnContextSetExtendedDeleterCb_t)(
    ur_context_set_extended_deleter_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Context callback functions pointers
typedef struct ur_context_callbacks_t {
    ur_pfnContextCreateCb_t pfnCreateCb;
    ur_pfnContextRetainCb_t pfnRetainCb;
    ur_pfnContextReleaseCb_t pfnReleaseCb;
    ur_pfnContextGetInfoCb_t pfnGetInfoCb;
    ur_pfnContextGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnContextCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
    ur_pfnContextSetExtendedDeleterCb_t pfnSetExtendedDeleterCb;
} ur_context_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_get_info_params_t {
    ur_event_handle_t *phEvent;
    ur_event_info_t *ppropName;
    size_t *ppropValueSize;
    void **ppPropValue;
    size_t **ppPropValueSizeRet;
} ur_event_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventGetInfoCb_t)(
    ur_event_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventGetProfilingInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_get_profiling_info_params_t {
    ur_event_handle_t *phEvent;
    ur_profiling_info_t *ppropName;
    size_t *ppropValueSize;
    void **ppPropValue;
    size_t **ppPropValueSizeRet;
} ur_event_get_profiling_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventGetProfilingInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventGetProfilingInfoCb_t)(
    ur_event_get_profiling_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventWait
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_wait_params_t {
    uint32_t *pnumEvents;
    const ur_event_handle_t **pphEventWaitList;
} ur_event_wait_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventWait
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventWaitCb_t)(
    ur_event_wait_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_retain_params_t {
    ur_event_handle_t *phEvent;
} ur_event_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventRetainCb_t)(
    ur_event_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_release_params_t {
    ur_event_handle_t *phEvent;
} ur_event_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventReleaseCb_t)(
    ur_event_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_get_native_handle_params_t {
    ur_event_handle_t *phEvent;
    ur_native_handle_t **pphNativeEvent;
} ur_event_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventGetNativeHandleCb_t)(
    ur_event_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeEvent;
    ur_context_handle_t *phContext;
    ur_event_handle_t **pphEvent;
} ur_event_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventCreateWithNativeHandleCb_t)(
    ur_event_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEventSetCallback
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_set_callback_params_t {
    ur_event_handle_t *phEvent;
    ur_execution_info_t *pexecStatus;
    ur_event_callback_t *ppfnNotify;
    void **ppUserData;
} ur_event_set_callback_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEventSetCallback
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEventSetCallbackCb_t)(
    ur_event_set_callback_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Event callback functions pointers
typedef struct ur_event_callbacks_t {
    ur_pfnEventGetInfoCb_t pfnGetInfoCb;
    ur_pfnEventGetProfilingInfoCb_t pfnGetProfilingInfoCb;
    ur_pfnEventWaitCb_t pfnWaitCb;
    ur_pfnEventRetainCb_t pfnRetainCb;
    ur_pfnEventReleaseCb_t pfnReleaseCb;
    ur_pfnEventGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnEventCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
    ur_pfnEventSetCallbackCb_t pfnSetCallbackCb;
} ur_event_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramCreateWithIL
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_create_with_il_params_t {
    ur_context_handle_t *phContext;
    const void **ppIL;
    size_t *plength;
    const ur_program_properties_t **ppProperties;
    ur_program_handle_t **pphProgram;
} ur_program_create_with_il_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramCreateWithIL
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramCreateWithILCb_t)(
    ur_program_create_with_il_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramCreateWithBinary
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_create_with_binary_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    size_t *psize;
    const uint8_t **ppBinary;
    const ur_program_properties_t **ppProperties;
    ur_program_handle_t **pphProgram;
} ur_program_create_with_binary_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramCreateWithBinary
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramCreateWithBinaryCb_t)(
    ur_program_create_with_binary_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramBuild
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_build_params_t {
    ur_context_handle_t *phContext;
    ur_program_handle_t *phProgram;
    const char **ppOptions;
} ur_program_build_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramBuild
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramBuildCb_t)(
    ur_program_build_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramCompile
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_compile_params_t {
    ur_context_handle_t *phContext;
    ur_program_handle_t *phProgram;
    const char **ppOptions;
} ur_program_compile_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramCompile
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramCompileCb_t)(
    ur_program_compile_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramLink
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_link_params_t {
    ur_context_handle_t *phContext;
    uint32_t *pcount;
    const ur_program_handle_t **pphPrograms;
    const char **ppOptions;
    ur_program_handle_t **pphProgram;
} ur_program_link_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramLink
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramLinkCb_t)(
    ur_program_link_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_retain_params_t {
    ur_program_handle_t *phProgram;
} ur_program_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramRetainCb_t)(
    ur_program_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_release_params_t {
    ur_program_handle_t *phProgram;
} ur_program_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramReleaseCb_t)(
    ur_program_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramGetFunctionPointer
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_function_pointer_params_t {
    ur_device_handle_t *phDevice;
    ur_program_handle_t *phProgram;
    const char **ppFunctionName;
    void ***pppFunctionPointer;
} ur_program_get_function_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramGetFunctionPointer
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramGetFunctionPointerCb_t)(
    ur_program_get_function_pointer_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_info_params_t {
    ur_program_handle_t *phProgram;
    ur_program_info_t *ppropName;
    size_t *ppropSize;
    void **ppProgramInfo;
    size_t **ppPropSizeRet;
} ur_program_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramGetInfoCb_t)(
    ur_program_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramGetBuildInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_build_info_params_t {
    ur_program_handle_t *phProgram;
    ur_device_handle_t *phDevice;
    ur_program_build_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_program_get_build_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramGetBuildInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramGetBuildInfoCb_t)(
    ur_program_get_build_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramSetSpecializationConstants
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_set_specialization_constants_params_t {
    ur_program_handle_t *phProgram;
    uint32_t *pcount;
    const ur_specialization_constant_info_t **ppSpecConstants;
} ur_program_set_specialization_constants_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramSetSpecializationConstants
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramSetSpecializationConstantsCb_t)(
    ur_program_set_specialization_constants_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_native_handle_params_t {
    ur_program_handle_t *phProgram;
    ur_native_handle_t **pphNativeProgram;
} ur_program_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramGetNativeHandleCb_t)(
    ur_program_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urProgramCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeProgram;
    ur_context_handle_t *phContext;
    ur_program_handle_t **pphProgram;
} ur_program_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urProgramCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnProgramCreateWithNativeHandleCb_t)(
    ur_program_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Program callback functions pointers
typedef struct ur_program_callbacks_t {
    ur_pfnProgramCreateWithILCb_t pfnCreateWithILCb;
    ur_pfnProgramCreateWithBinaryCb_t pfnCreateWithBinaryCb;
    ur_pfnProgramBuildCb_t pfnBuildCb;
    ur_pfnProgramCompileCb_t pfnCompileCb;
    ur_pfnProgramLinkCb_t pfnLinkCb;
    ur_pfnProgramRetainCb_t pfnRetainCb;
    ur_pfnProgramReleaseCb_t pfnReleaseCb;
    ur_pfnProgramGetFunctionPointerCb_t pfnGetFunctionPointerCb;
    ur_pfnProgramGetInfoCb_t pfnGetInfoCb;
    ur_pfnProgramGetBuildInfoCb_t pfnGetBuildInfoCb;
    ur_pfnProgramSetSpecializationConstantsCb_t pfnSetSpecializationConstantsCb;
    ur_pfnProgramGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnProgramCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
} ur_program_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_create_params_t {
    ur_program_handle_t *phProgram;
    const char **ppKernelName;
    ur_kernel_handle_t **pphKernel;
} ur_kernel_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelCreate
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelCreateCb_t)(
    ur_kernel_create_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_kernel_info_t *ppropName;
    size_t *ppropSize;
    void **ppKernelInfo;
    size_t **ppPropSizeRet;
} ur_kernel_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelGetInfoCb_t)(
    ur_kernel_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelGetGroupInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_group_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_device_handle_t *phDevice;
    ur_kernel_group_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_kernel_get_group_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelGetGroupInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelGetGroupInfoCb_t)(
    ur_kernel_get_group_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelGetSubGroupInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_sub_group_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_device_handle_t *phDevice;
    ur_kernel_sub_group_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_kernel_get_sub_group_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelGetSubGroupInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelGetSubGroupInfoCb_t)(
    ur_kernel_get_sub_group_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_retain_params_t {
    ur_kernel_handle_t *phKernel;
} ur_kernel_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelRetainCb_t)(
    ur_kernel_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_release_params_t {
    ur_kernel_handle_t *phKernel;
} ur_kernel_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelReleaseCb_t)(
    ur_kernel_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_native_handle_params_t {
    ur_kernel_handle_t *phKernel;
    ur_native_handle_t **pphNativeKernel;
} ur_kernel_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelGetNativeHandleCb_t)(
    ur_kernel_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeKernel;
    ur_context_handle_t *phContext;
    ur_kernel_handle_t **pphKernel;
} ur_kernel_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelCreateWithNativeHandleCb_t)(
    ur_kernel_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelSetArgValue
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_value_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    size_t *pargSize;
    const void **ppArgValue;
} ur_kernel_set_arg_value_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelSetArgValue
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelSetArgValueCb_t)(
    ur_kernel_set_arg_value_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelSetArgLocal
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_local_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    size_t *pargSize;
} ur_kernel_set_arg_local_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelSetArgLocal
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelSetArgLocalCb_t)(
    ur_kernel_set_arg_local_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelSetArgPointer
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_pointer_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    const void **ppArgValue;
} ur_kernel_set_arg_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelSetArgPointer
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelSetArgPointerCb_t)(
    ur_kernel_set_arg_pointer_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelSetExecInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_exec_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_kernel_exec_info_t *ppropName;
    size_t *ppropSize;
    const void **ppPropValue;
} ur_kernel_set_exec_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelSetExecInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelSetExecInfoCb_t)(
    ur_kernel_set_exec_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelSetArgSampler
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_sampler_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    ur_sampler_handle_t *phArgValue;
} ur_kernel_set_arg_sampler_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelSetArgSampler
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelSetArgSamplerCb_t)(
    ur_kernel_set_arg_sampler_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelSetArgMemObj
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_mem_obj_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    ur_mem_handle_t *phArgValue;
} ur_kernel_set_arg_mem_obj_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelSetArgMemObj
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelSetArgMemObjCb_t)(
    ur_kernel_set_arg_mem_obj_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urKernelSetSpecializationConstants
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_specialization_constants_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pcount;
    const ur_specialization_constant_info_t **ppSpecConstants;
} ur_kernel_set_specialization_constants_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urKernelSetSpecializationConstants
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnKernelSetSpecializationConstantsCb_t)(
    ur_kernel_set_specialization_constants_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Kernel callback functions pointers
typedef struct ur_kernel_callbacks_t {
    ur_pfnKernelCreateCb_t pfnCreateCb;
    ur_pfnKernelGetInfoCb_t pfnGetInfoCb;
    ur_pfnKernelGetGroupInfoCb_t pfnGetGroupInfoCb;
    ur_pfnKernelGetSubGroupInfoCb_t pfnGetSubGroupInfoCb;
    ur_pfnKernelRetainCb_t pfnRetainCb;
    ur_pfnKernelReleaseCb_t pfnReleaseCb;
    ur_pfnKernelGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnKernelCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
    ur_pfnKernelSetArgValueCb_t pfnSetArgValueCb;
    ur_pfnKernelSetArgLocalCb_t pfnSetArgLocalCb;
    ur_pfnKernelSetArgPointerCb_t pfnSetArgPointerCb;
    ur_pfnKernelSetExecInfoCb_t pfnSetExecInfoCb;
    ur_pfnKernelSetArgSamplerCb_t pfnSetArgSamplerCb;
    ur_pfnKernelSetArgMemObjCb_t pfnSetArgMemObjCb;
    ur_pfnKernelSetSpecializationConstantsCb_t pfnSetSpecializationConstantsCb;
} ur_kernel_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urSamplerCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_create_params_t {
    ur_context_handle_t *phContext;
    const ur_sampler_property_t **ppProps;
    ur_sampler_handle_t **pphSampler;
} ur_sampler_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urSamplerCreate
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnSamplerCreateCb_t)(
    ur_sampler_create_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urSamplerRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_retain_params_t {
    ur_sampler_handle_t *phSampler;
} ur_sampler_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urSamplerRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnSamplerRetainCb_t)(
    ur_sampler_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urSamplerRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_release_params_t {
    ur_sampler_handle_t *phSampler;
} ur_sampler_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urSamplerRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnSamplerReleaseCb_t)(
    ur_sampler_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urSamplerGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_get_info_params_t {
    ur_sampler_handle_t *phSampler;
    ur_sampler_info_t *ppropName;
    size_t *ppropValueSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_sampler_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urSamplerGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnSamplerGetInfoCb_t)(
    ur_sampler_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urSamplerGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_get_native_handle_params_t {
    ur_sampler_handle_t *phSampler;
    ur_native_handle_t **pphNativeSampler;
} ur_sampler_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urSamplerGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnSamplerGetNativeHandleCb_t)(
    ur_sampler_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urSamplerCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeSampler;
    ur_context_handle_t *phContext;
    ur_sampler_handle_t **pphSampler;
} ur_sampler_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urSamplerCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnSamplerCreateWithNativeHandleCb_t)(
    ur_sampler_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Sampler callback functions pointers
typedef struct ur_sampler_callbacks_t {
    ur_pfnSamplerCreateCb_t pfnCreateCb;
    ur_pfnSamplerRetainCb_t pfnRetainCb;
    ur_pfnSamplerReleaseCb_t pfnReleaseCb;
    ur_pfnSamplerGetInfoCb_t pfnGetInfoCb;
    ur_pfnSamplerGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnSamplerCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
} ur_sampler_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemImageCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_image_create_params_t {
    ur_context_handle_t *phContext;
    ur_mem_flags_t *pflags;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    void **ppHost;
    ur_mem_handle_t **pphMem;
} ur_mem_image_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemImageCreate
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemImageCreateCb_t)(
    ur_mem_image_create_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemBufferCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_buffer_create_params_t {
    ur_context_handle_t *phContext;
    ur_mem_flags_t *pflags;
    size_t *psize;
    const ur_buffer_properties_t **ppProperties;
    ur_mem_handle_t **pphBuffer;
} ur_mem_buffer_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemBufferCreate
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemBufferCreateCb_t)(
    ur_mem_buffer_create_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_retain_params_t {
    ur_mem_handle_t *phMem;
} ur_mem_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemRetainCb_t)(
    ur_mem_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_release_params_t {
    ur_mem_handle_t *phMem;
} ur_mem_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemReleaseCb_t)(
    ur_mem_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemBufferPartition
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_buffer_partition_params_t {
    ur_mem_handle_t *phBuffer;
    ur_mem_flags_t *pflags;
    ur_buffer_create_type_t *pbufferCreateType;
    const ur_buffer_region_t **ppRegion;
    ur_mem_handle_t **pphMem;
} ur_mem_buffer_partition_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemBufferPartition
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemBufferPartitionCb_t)(
    ur_mem_buffer_partition_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_get_native_handle_params_t {
    ur_mem_handle_t *phMem;
    ur_native_handle_t **pphNativeMem;
} ur_mem_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemGetNativeHandleCb_t)(
    ur_mem_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeMem;
    ur_context_handle_t *phContext;
    ur_mem_handle_t **pphMem;
} ur_mem_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemCreateWithNativeHandleCb_t)(
    ur_mem_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_get_info_params_t {
    ur_mem_handle_t *phMemory;
    ur_mem_info_t *pMemInfoType;
    size_t *ppropSize;
    void **ppMemInfo;
    size_t **ppPropSizeRet;
} ur_mem_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemGetInfoCb_t)(
    ur_mem_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urMemImageGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_image_get_info_params_t {
    ur_mem_handle_t *phMemory;
    ur_image_info_t *pImgInfoType;
    size_t *ppropSize;
    void **ppImgInfo;
    size_t **ppPropSizeRet;
} ur_mem_image_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urMemImageGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnMemImageGetInfoCb_t)(
    ur_mem_image_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Mem callback functions pointers
typedef struct ur_mem_callbacks_t {
    ur_pfnMemImageCreateCb_t pfnImageCreateCb;
    ur_pfnMemBufferCreateCb_t pfnBufferCreateCb;
    ur_pfnMemRetainCb_t pfnRetainCb;
    ur_pfnMemReleaseCb_t pfnReleaseCb;
    ur_pfnMemBufferPartitionCb_t pfnBufferPartitionCb;
    ur_pfnMemGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnMemCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
    ur_pfnMemGetInfoCb_t pfnGetInfoCb;
    ur_pfnMemImageGetInfoCb_t pfnImageGetInfoCb;
} ur_mem_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueKernelLaunch
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_kernel_launch_params_t {
    ur_queue_handle_t *phQueue;
    ur_kernel_handle_t *phKernel;
    uint32_t *pworkDim;
    const size_t **ppGlobalWorkOffset;
    const size_t **ppGlobalWorkSize;
    const size_t **ppLocalWorkSize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_kernel_launch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueKernelLaunch
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueKernelLaunchCb_t)(
    ur_enqueue_kernel_launch_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueEventsWait
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_events_wait_params_t {
    ur_queue_handle_t *phQueue;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_events_wait_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueEventsWait
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueEventsWaitCb_t)(
    ur_enqueue_events_wait_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueEventsWaitWithBarrier
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_events_wait_with_barrier_params_t {
    ur_queue_handle_t *phQueue;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_events_wait_with_barrier_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueEventsWaitWithBarrier
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueEventsWaitWithBarrierCb_t)(
    ur_enqueue_events_wait_with_barrier_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferRead
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_read_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingRead;
    size_t *poffset;
    size_t *psize;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferRead
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferReadCb_t)(
    ur_enqueue_mem_buffer_read_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferWrite
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_write_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingWrite;
    size_t *poffset;
    size_t *psize;
    const void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferWrite
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferWriteCb_t)(
    ur_enqueue_mem_buffer_write_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferReadRect
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_read_rect_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingRead;
    ur_rect_offset_t *pbufferOrigin;
    ur_rect_offset_t *phostOrigin;
    ur_rect_region_t *pregion;
    size_t *pbufferRowPitch;
    size_t *pbufferSlicePitch;
    size_t *phostRowPitch;
    size_t *phostSlicePitch;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_read_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferReadRect
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferReadRectCb_t)(
    ur_enqueue_mem_buffer_read_rect_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferWriteRect
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_write_rect_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingWrite;
    ur_rect_offset_t *pbufferOrigin;
    ur_rect_offset_t *phostOrigin;
    ur_rect_region_t *pregion;
    size_t *pbufferRowPitch;
    size_t *pbufferSlicePitch;
    size_t *phostRowPitch;
    size_t *phostSlicePitch;
    void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_write_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferWriteRect
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferWriteRectCb_t)(
    ur_enqueue_mem_buffer_write_rect_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferCopy
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_copy_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBufferSrc;
    ur_mem_handle_t *phBufferDst;
    size_t *psrcOffset;
    size_t *pdstOffset;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferCopy
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferCopyCb_t)(
    ur_enqueue_mem_buffer_copy_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferCopyRect
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_copy_rect_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBufferSrc;
    ur_mem_handle_t *phBufferDst;
    ur_rect_offset_t *psrcOrigin;
    ur_rect_offset_t *pdstOrigin;
    ur_rect_region_t *pregion;
    size_t *psrcRowPitch;
    size_t *psrcSlicePitch;
    size_t *pdstRowPitch;
    size_t *pdstSlicePitch;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_copy_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferCopyRect
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferCopyRectCb_t)(
    ur_enqueue_mem_buffer_copy_rect_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferFill
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_fill_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    const void **ppPattern;
    size_t *ppatternSize;
    size_t *poffset;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_fill_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferFill
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferFillCb_t)(
    ur_enqueue_mem_buffer_fill_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemImageRead
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_image_read_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phImage;
    bool *pblockingRead;
    ur_rect_offset_t *porigin;
    ur_rect_region_t *pregion;
    size_t *prowPitch;
    size_t *pslicePitch;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_image_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemImageRead
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemImageReadCb_t)(
    ur_enqueue_mem_image_read_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemImageWrite
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_image_write_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phImage;
    bool *pblockingWrite;
    ur_rect_offset_t *porigin;
    ur_rect_region_t *pregion;
    size_t *prowPitch;
    size_t *pslicePitch;
    void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_image_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemImageWrite
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemImageWriteCb_t)(
    ur_enqueue_mem_image_write_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemImageCopy
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_image_copy_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phImageSrc;
    ur_mem_handle_t *phImageDst;
    ur_rect_offset_t *psrcOrigin;
    ur_rect_offset_t *pdstOrigin;
    ur_rect_region_t *pregion;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_image_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemImageCopy
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemImageCopyCb_t)(
    ur_enqueue_mem_image_copy_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemBufferMap
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_map_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingMap;
    ur_map_flags_t *pmapFlags;
    size_t *poffset;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
    void ***pppRetMap;
} ur_enqueue_mem_buffer_map_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemBufferMap
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemBufferMapCb_t)(
    ur_enqueue_mem_buffer_map_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueMemUnmap
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_unmap_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phMem;
    void **ppMappedPtr;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_unmap_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueMemUnmap
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueMemUnmapCb_t)(
    ur_enqueue_mem_unmap_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueUSMFill
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_fill_params_t {
    ur_queue_handle_t *phQueue;
    void **pptr;
    size_t *ppatternSize;
    const void **ppPattern;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_fill_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueUSMFill
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueUSMFillCb_t)(
    ur_enqueue_usm_fill_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueUSMMemcpy
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_memcpy_params_t {
    ur_queue_handle_t *phQueue;
    bool *pblocking;
    void **ppDst;
    const void **ppSrc;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_memcpy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueUSMMemcpy
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueUSMMemcpyCb_t)(
    ur_enqueue_usm_memcpy_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueUSMPrefetch
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_prefetch_params_t {
    ur_queue_handle_t *phQueue;
    const void **ppMem;
    size_t *psize;
    ur_usm_migration_flags_t *pflags;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_prefetch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueUSMPrefetch
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueUSMPrefetchCb_t)(
    ur_enqueue_usm_prefetch_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueUSMMemAdvise
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_mem_advise_params_t {
    ur_queue_handle_t *phQueue;
    const void **ppMem;
    size_t *psize;
    ur_mem_advice_t *padvice;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_mem_advise_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueUSMMemAdvise
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueUSMMemAdviseCb_t)(
    ur_enqueue_usm_mem_advise_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueUSMFill2D
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_fill2_d_params_t {
    ur_queue_handle_t *phQueue;
    void **ppMem;
    size_t *ppitch;
    size_t *ppatternSize;
    const void **ppPattern;
    size_t *pwidth;
    size_t *pheight;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_fill2_d_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueUSMFill2D
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueUSMFill2DCb_t)(
    ur_enqueue_usm_fill2_d_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueUSMMemcpy2D
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_memcpy2_d_params_t {
    ur_queue_handle_t *phQueue;
    bool *pblocking;
    void **ppDst;
    size_t *pdstPitch;
    const void **ppSrc;
    size_t *psrcPitch;
    size_t *pwidth;
    size_t *pheight;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_memcpy2_d_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueUSMMemcpy2D
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueUSMMemcpy2DCb_t)(
    ur_enqueue_usm_memcpy2_d_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueDeviceGlobalVariableWrite
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_device_global_variable_write_params_t {
    ur_queue_handle_t *phQueue;
    ur_program_handle_t *phProgram;
    const char **pname;
    bool *pblockingWrite;
    size_t *pcount;
    size_t *poffset;
    const void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_device_global_variable_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueDeviceGlobalVariableWrite
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueDeviceGlobalVariableWriteCb_t)(
    ur_enqueue_device_global_variable_write_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urEnqueueDeviceGlobalVariableRead
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_device_global_variable_read_params_t {
    ur_queue_handle_t *phQueue;
    ur_program_handle_t *phProgram;
    const char **pname;
    bool *pblockingRead;
    size_t *pcount;
    size_t *poffset;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_device_global_variable_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urEnqueueDeviceGlobalVariableRead
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnEnqueueDeviceGlobalVariableReadCb_t)(
    ur_enqueue_device_global_variable_read_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Enqueue callback functions pointers
typedef struct ur_enqueue_callbacks_t {
    ur_pfnEnqueueKernelLaunchCb_t pfnKernelLaunchCb;
    ur_pfnEnqueueEventsWaitCb_t pfnEventsWaitCb;
    ur_pfnEnqueueEventsWaitWithBarrierCb_t pfnEventsWaitWithBarrierCb;
    ur_pfnEnqueueMemBufferReadCb_t pfnMemBufferReadCb;
    ur_pfnEnqueueMemBufferWriteCb_t pfnMemBufferWriteCb;
    ur_pfnEnqueueMemBufferReadRectCb_t pfnMemBufferReadRectCb;
    ur_pfnEnqueueMemBufferWriteRectCb_t pfnMemBufferWriteRectCb;
    ur_pfnEnqueueMemBufferCopyCb_t pfnMemBufferCopyCb;
    ur_pfnEnqueueMemBufferCopyRectCb_t pfnMemBufferCopyRectCb;
    ur_pfnEnqueueMemBufferFillCb_t pfnMemBufferFillCb;
    ur_pfnEnqueueMemImageReadCb_t pfnMemImageReadCb;
    ur_pfnEnqueueMemImageWriteCb_t pfnMemImageWriteCb;
    ur_pfnEnqueueMemImageCopyCb_t pfnMemImageCopyCb;
    ur_pfnEnqueueMemBufferMapCb_t pfnMemBufferMapCb;
    ur_pfnEnqueueMemUnmapCb_t pfnMemUnmapCb;
    ur_pfnEnqueueUSMFillCb_t pfnUSMFillCb;
    ur_pfnEnqueueUSMMemcpyCb_t pfnUSMMemcpyCb;
    ur_pfnEnqueueUSMPrefetchCb_t pfnUSMPrefetchCb;
    ur_pfnEnqueueUSMMemAdviseCb_t pfnUSMMemAdviseCb;
    ur_pfnEnqueueUSMFill2DCb_t pfnUSMFill2DCb;
    ur_pfnEnqueueUSMMemcpy2DCb_t pfnUSMMemcpy2DCb;
    ur_pfnEnqueueDeviceGlobalVariableWriteCb_t pfnDeviceGlobalVariableWriteCb;
    ur_pfnEnqueueDeviceGlobalVariableReadCb_t pfnDeviceGlobalVariableReadCb;
} ur_enqueue_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urInit
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_init_params_t {
    ur_device_init_flags_t *pdevice_flags;
} ur_init_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urInit
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnInitCb_t)(
    ur_init_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urGetLastResult
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_get_last_result_params_t {
    ur_platform_handle_t *phPlatform;
    const char ***pppMessage;
} ur_get_last_result_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urGetLastResult
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnGetLastResultCb_t)(
    ur_get_last_result_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urTearDown
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_tear_down_params_t {
    void **ppParams;
} ur_tear_down_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urTearDown
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnTearDownCb_t)(
    ur_tear_down_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Global callback functions pointers
typedef struct ur_global_callbacks_t {
    ur_pfnInitCb_t pfnInitCb;
    ur_pfnGetLastResultCb_t pfnGetLastResultCb;
    ur_pfnTearDownCb_t pfnTearDownCb;
} ur_global_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_get_info_params_t {
    ur_queue_handle_t *phQueue;
    ur_queue_info_t *ppropName;
    size_t *ppropValueSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_queue_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueGetInfoCb_t)(
    ur_queue_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_create_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_queue_property_t **ppProps;
    ur_queue_handle_t **pphQueue;
} ur_queue_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueCreate
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueCreateCb_t)(
    ur_queue_create_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_retain_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueRetainCb_t)(
    ur_queue_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_release_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueReleaseCb_t)(
    ur_queue_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_get_native_handle_params_t {
    ur_queue_handle_t *phQueue;
    ur_native_handle_t **pphNativeQueue;
} ur_queue_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueGetNativeHandleCb_t)(
    ur_queue_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeQueue;
    ur_context_handle_t *phContext;
    ur_queue_handle_t **pphQueue;
} ur_queue_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueCreateWithNativeHandleCb_t)(
    ur_queue_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueFinish
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_finish_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_finish_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueFinish
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueFinishCb_t)(
    ur_queue_finish_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urQueueFlush
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_flush_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_flush_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urQueueFlush
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnQueueFlushCb_t)(
    ur_queue_flush_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Queue callback functions pointers
typedef struct ur_queue_callbacks_t {
    ur_pfnQueueGetInfoCb_t pfnGetInfoCb;
    ur_pfnQueueCreateCb_t pfnCreateCb;
    ur_pfnQueueRetainCb_t pfnRetainCb;
    ur_pfnQueueReleaseCb_t pfnReleaseCb;
    ur_pfnQueueGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnQueueCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
    ur_pfnQueueFinishCb_t pfnFinishCb;
    ur_pfnQueueFlushCb_t pfnFlushCb;
} ur_queue_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urUSMHostAlloc
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_host_alloc_params_t {
    ur_context_handle_t *phContext;
    ur_usm_desc_t **ppUSMDesc;
    ur_usm_pool_handle_t *ppool;
    size_t *psize;
    uint32_t *palign;
    void ***pppMem;
} ur_usm_host_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urUSMHostAlloc
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnUSMHostAllocCb_t)(
    ur_usm_host_alloc_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urUSMDeviceAlloc
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_device_alloc_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_usm_desc_t **ppUSMDesc;
    ur_usm_pool_handle_t *ppool;
    size_t *psize;
    uint32_t *palign;
    void ***pppMem;
} ur_usm_device_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urUSMDeviceAlloc
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnUSMDeviceAllocCb_t)(
    ur_usm_device_alloc_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urUSMSharedAlloc
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_shared_alloc_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_usm_desc_t **ppUSMDesc;
    ur_usm_pool_handle_t *ppool;
    size_t *psize;
    uint32_t *palign;
    void ***pppMem;
} ur_usm_shared_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urUSMSharedAlloc
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnUSMSharedAllocCb_t)(
    ur_usm_shared_alloc_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urUSMFree
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_free_params_t {
    ur_context_handle_t *phContext;
    void **ppMem;
} ur_usm_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urUSMFree
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnUSMFreeCb_t)(
    ur_usm_free_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urUSMGetMemAllocInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_get_mem_alloc_info_params_t {
    ur_context_handle_t *phContext;
    const void **ppMem;
    ur_usm_alloc_info_t *ppropName;
    size_t *ppropValueSize;
    void **ppPropValue;
    size_t **ppPropValueSizeRet;
} ur_usm_get_mem_alloc_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urUSMGetMemAllocInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnUSMGetMemAllocInfoCb_t)(
    ur_usm_get_mem_alloc_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urUSMPoolCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_pool_create_params_t {
    ur_context_handle_t *phContext;
    ur_usm_pool_desc_t **ppPoolDesc;
    ur_usm_pool_handle_t **pppPool;
} ur_usm_pool_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urUSMPoolCreate
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnUSMPoolCreateCb_t)(
    ur_usm_pool_create_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urUSMPoolDestroy
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_pool_destroy_params_t {
    ur_context_handle_t *phContext;
    ur_usm_pool_handle_t *ppPool;
} ur_usm_pool_destroy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urUSMPoolDestroy
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnUSMPoolDestroyCb_t)(
    ur_usm_pool_destroy_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of USM callback functions pointers
typedef struct ur_usm_callbacks_t {
    ur_pfnUSMHostAllocCb_t pfnHostAllocCb;
    ur_pfnUSMDeviceAllocCb_t pfnDeviceAllocCb;
    ur_pfnUSMSharedAllocCb_t pfnSharedAllocCb;
    ur_pfnUSMFreeCb_t pfnFreeCb;
    ur_pfnUSMGetMemAllocInfoCb_t pfnGetMemAllocInfoCb;
    ur_pfnUSMPoolCreateCb_t pfnPoolCreateCb;
    ur_pfnUSMPoolDestroyCb_t pfnPoolDestroyCb;
} ur_usm_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceGet
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_params_t {
    ur_platform_handle_t *phPlatform;
    ur_device_type_t *pDeviceType;
    uint32_t *pNumEntries;
    ur_device_handle_t **pphDevices;
    uint32_t **ppNumDevices;
} ur_device_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceGet
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceGetCb_t)(
    ur_device_get_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_info_params_t {
    ur_device_handle_t *phDevice;
    ur_device_info_t *pinfoType;
    size_t *ppropSize;
    void **ppDeviceInfo;
    size_t **ppPropSizeRet;
} ur_device_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceGetInfo
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceGetInfoCb_t)(
    ur_device_get_info_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_retain_params_t {
    ur_device_handle_t *phDevice;
} ur_device_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceRetain
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceRetainCb_t)(
    ur_device_retain_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_release_params_t {
    ur_device_handle_t *phDevice;
} ur_device_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceRelease
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceReleaseCb_t)(
    ur_device_release_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDevicePartition
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_partition_params_t {
    ur_device_handle_t *phDevice;
    const ur_device_partition_property_t **ppProperties;
    uint32_t *pNumDevices;
    ur_device_handle_t **pphSubDevices;
    uint32_t **ppNumDevicesRet;
} ur_device_partition_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDevicePartition
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDevicePartitionCb_t)(
    ur_device_partition_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceSelectBinary
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_select_binary_params_t {
    ur_device_handle_t *phDevice;
    const uint8_t ***pppBinaries;
    uint32_t *pNumBinaries;
    uint32_t **ppSelectedBinary;
} ur_device_select_binary_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceSelectBinary
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceSelectBinaryCb_t)(
    ur_device_select_binary_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_native_handle_params_t {
    ur_device_handle_t *phDevice;
    ur_native_handle_t **pphNativeDevice;
} ur_device_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceGetNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceGetNativeHandleCb_t)(
    ur_device_get_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeDevice;
    ur_platform_handle_t *phPlatform;
    ur_device_handle_t **pphDevice;
} ur_device_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceCreateWithNativeHandle
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceCreateWithNativeHandleCb_t)(
    ur_device_create_with_native_handle_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for urDeviceGetGlobalTimestamps
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_global_timestamps_params_t {
    ur_device_handle_t *phDevice;
    uint64_t **ppDeviceTimestamp;
    uint64_t **ppHostTimestamp;
} ur_device_get_global_timestamps_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for urDeviceGetGlobalTimestamps
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void(UR_APICALL *ur_pfnDeviceGetGlobalTimestampsCb_t)(
    ur_device_get_global_timestamps_params_t *params,
    ur_result_t result,
    void *pTracerUserData,
    void **ppTracerInstanceUserData);

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Device callback functions pointers
typedef struct ur_device_callbacks_t {
    ur_pfnDeviceGetCb_t pfnGetCb;
    ur_pfnDeviceGetInfoCb_t pfnGetInfoCb;
    ur_pfnDeviceRetainCb_t pfnRetainCb;
    ur_pfnDeviceReleaseCb_t pfnReleaseCb;
    ur_pfnDevicePartitionCb_t pfnPartitionCb;
    ur_pfnDeviceSelectBinaryCb_t pfnSelectBinaryCb;
    ur_pfnDeviceGetNativeHandleCb_t pfnGetNativeHandleCb;
    ur_pfnDeviceCreateWithNativeHandleCb_t pfnCreateWithNativeHandleCb;
    ur_pfnDeviceGetGlobalTimestampsCb_t pfnGetGlobalTimestampsCb;
} ur_device_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Container for all callbacks
typedef struct ur_callbacks_t {
    ur_platform_callbacks_t Platform;
    ur_context_callbacks_t Context;
    ur_event_callbacks_t Event;
    ur_program_callbacks_t Program;
    ur_kernel_callbacks_t Kernel;
    ur_sampler_callbacks_t Sampler;
    ur_mem_callbacks_t Mem;
    ur_enqueue_callbacks_t Enqueue;
    ur_global_callbacks_t Global;
    ur_queue_callbacks_t Queue;
    ur_usm_callbacks_t USM;
    ur_device_callbacks_t Device;
} ur_callbacks_t;

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // UR_API_H_INCLUDED
