/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zer_api.h
 * @version v0.5-r0.5
 *
 */
#ifndef _ZER_API_H
#define _ZER_API_H
#if defined(__cplusplus)
#pragma once
#endif

// standard headers
#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Level-Zero Runtime API common types
#if !defined(__GNUC__)
#pragma region common
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_MAKE_VERSION
/// @brief Generates generic 'oneAPI' API versions
#define ZER_MAKE_VERSION( _major, _minor )  (( _major << 16 )|( _minor & 0x0000ffff))
#endif // ZER_MAKE_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_MAJOR_VERSION
/// @brief Extracts 'oneAPI' API major version
#define ZER_MAJOR_VERSION( _ver )  ( _ver >> 16 )
#endif // ZER_MAJOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_MINOR_VERSION
/// @brief Extracts 'oneAPI' API minor version
#define ZER_MINOR_VERSION( _ver )  ( _ver & 0x0000ffff )
#endif // ZER_MINOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_APICALL
#if defined(_WIN32)
/// @brief Calling convention for all API functions
#define ZER_APICALL  __cdecl
#else
#define ZER_APICALL  
#endif // defined(_WIN32)
#endif // ZER_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_APIEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define ZER_APIEXPORT  __declspec(dllexport)
#else
#define ZER_APIEXPORT  
#endif // defined(_WIN32)
#endif // ZER_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_DLLEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define ZER_DLLEXPORT  __declspec(dllexport)
#endif // defined(_WIN32)
#endif // ZER_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_DLLEXPORT
#if __GNUC__ >= 4
/// @brief GCC-specific dllexport storage-class attribute
#define ZER_DLLEXPORT  __attribute__ ((visibility ("default")))
#else
#define ZER_DLLEXPORT  
#endif // __GNUC__ >= 4
#endif // ZER_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief compiler-independent type
typedef uint8_t zer_bool_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a platform instance
typedef struct _zer_platform_handle_t *zer_platform_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of platform's device object
typedef struct _zer_device_handle_t *zer_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of context object
typedef struct _zer_context_handle_t *zer_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of event object
typedef struct _zer_event_handle_t *zer_event_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of Program object
typedef struct _zer_program_handle_t *zer_program_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of Module object
typedef struct _zer_module_handle_t *zer_module_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of module's Kernel object
typedef struct _zer_kernel_handle_t *zer_kernel_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a queue object
typedef struct _zer_queue_handle_t *zer_queue_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a native object
typedef struct _zer_native_handle_t *zer_native_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a Sampler object
typedef struct _zer_sampler_handle_t *zer_sampler_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of memory object which can either be buffer or image
typedef struct _zer_mem_handle_t *zer_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZER_BIT
/// @brief Generic macro for enumerator bit masks
#define ZER_BIT( _i )  ( 1 << _i )
#endif // ZER_BIT

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes
typedef enum _zer_result_t
{
    ZER_RESULT_SUCCESS = 0,                         ///< Success
    ZER_RESULT_INVALID_KERNEL_NAME = 1,             ///< Invalid kernel name
    ZER_RESULT_INVALID_OPERATION = 2,               ///< Invalid operation
    ZER_RESULT_INVALID_KERNEL = 3,                  ///< Invalid kernel
    ZER_RESULT_INVALID_QUEUE_PROPERTIES = 4,        ///< Invalid queue properties
    ZER_RESULT_INVALID_VALUE = 5,                   ///< Invalid Value
    ZER_RESULT_INVALID_CONTEXT = 6,                 ///< Invalid context
    ZER_RESULT_INVALID_PLATFORM = 7,                ///< Invalid platform
    ZER_RESULT_INVALID_DEVICE = 8,                  ///< Invalid device
    ZER_RESULT_INVALID_BINARY = 9,                  ///< Invalid binary
    ZER_RESULT_INVALID_QUEUE = 10,                  ///< Invalid queue
    ZER_RESULT_OUT_OF_HOST_MEMORY = 11,             ///< Out of host memory
    ZER_RESULT_INVALID_PROGRAM = 12,                ///< Invalid program
    ZER_RESULT_INVALID_PROGRAM_EXECUTABLE = 13,     ///< Invalid program executable
    ZER_RESULT_INVALID_SAMPLER = 14,                ///< Invalid sampler
    ZER_RESULT_INVALID_BUFFER_SIZE = 15,            ///< Invalid buffer size
    ZER_RESULT_INVALID_MEM_OBJECT = 16,             ///< Invalid memory object
    ZER_RESULT_OUT_OF_RESOURCES = 17,               ///< Out of resources
    ZER_RESULT_INVALID_EVENT = 18,                  ///< Invalid event
    ZER_RESULT_INVALID_EVENT_WAIT_LIST = 19,        ///< Invalid event wait list
    ZER_RESULT_MISALIGNED_SUB_BUFFER_OFFSET = 20,   ///< Misaligned sub buffer offset
    ZER_RESULT_BUILD_PROGRAM_FAILURE = 21,          ///< Build program failure
    ZER_RESULT_INVALID_WORK_GROUP_SIZE = 22,        ///< Invalid work group size
    ZER_RESULT_COMPILER_NOT_AVAILABLE = 23,         ///< Compiler not available
    ZER_RESULT_PROFILING_INFO_NOT_AVAILABLE = 24,   ///< Profiling info not available
    ZER_RESULT_DEVICE_NOT_FOUND = 25,               ///< Device not found
    ZER_RESULT_INVALID_WORK_ITEM_SIZE = 26,         ///< Invalid work item size
    ZER_RESULT_INVALID_WORK_DIMENSION = 27,         ///< Invalid work dimension
    ZER_RESULT_INVALID_KERNEL_ARGS = 28,            ///< Invalid kernel args
    ZER_RESULT_INVALID_IMAGE_SIZE = 29,             ///< Invalid image size
    ZER_RESULT_INVALID_IMAGE_FORMAT_DESCRIPTOR = 30,///< Invalid image format descriptor
    ZER_RESULT_IMAGE_FORMAT_NOT_SUPPORTED = 31,     ///< Image format not supported
    ZER_RESULT_MEM_OBJECT_ALLOCATION_FAILURE = 32,  ///< Memory object allocation failure
    ZER_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE = 33,   ///< Program object parameter is invalid.
    ZER_RESULT_ERROR_UNINITIALIZED = 0x78000001,    ///< [Validation] driver is not initialized
    ZER_RESULT_ERROR_DEVICE_LOST = 0x78000002,      ///< Device hung, reset, was removed, or driver update occurred
    ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY = 0x78000003,   ///< Insufficient host memory to satisfy call
    ZER_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 0x78000004, ///< Insufficient device memory to satisfy call
    ZER_RESULT_ERROR_MODULE_BUILD_FAILURE = 0x78000005, ///< Error occurred when building module, see build log for details
    ZER_RESULT_ERROR_MODULE_LINK_FAILURE = 0x78000006,  ///< Error occurred when linking modules, see build log for details
    ZER_RESULT_ERROR_DEVICE_REQUIRES_RESET = 0x78000007,///< Device requires a reset
    ZER_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 0x78000008,///< Device currently in low power state
    ZER_RESULT_ERROR_UNSUPPORTED_VERSION = 0x78000009,  ///< [Validation] generic error code for unsupported versions
    ZER_RESULT_ERROR_UNSUPPORTED_FEATURE = 0x7800000a,  ///< [Validation] generic error code for unsupported features
    ZER_RESULT_ERROR_INVALID_ARGUMENT = 0x7800000b, ///< [Validation] generic error code for invalid arguments
    ZER_RESULT_ERROR_INVALID_NULL_HANDLE = 0x7800000c,  ///< [Validation] handle argument is not valid
    ZER_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 0x7800000d, ///< [Validation] object pointed to by handle still in-use by device
    ZER_RESULT_ERROR_INVALID_NULL_POINTER = 0x7800000e, ///< [Validation] pointer argument may not be nullptr
    ZER_RESULT_ERROR_INVALID_SIZE = 0x7800000f,     ///< [Validation] size argument is invalid (e.g., must not be zero)
    ZER_RESULT_ERROR_UNSUPPORTED_SIZE = 0x78000010, ///< [Validation] size argument is not supported by the device (e.g., too
                                                    ///< large)
    ZER_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 0x78000011,///< [Validation] alignment argument is not supported by the device (e.g.,
                                                    ///< too small)
    ZER_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 0x78000012,   ///< [Validation] synchronization object in invalid state
    ZER_RESULT_ERROR_INVALID_ENUMERATION = 0x78000013,  ///< [Validation] enumerator argument is not valid
    ZER_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 0x78000014,  ///< [Validation] enumerator argument is not supported by the device
    ZER_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 0x78000015, ///< [Validation] image format is not supported by the device
    ZER_RESULT_ERROR_INVALID_NATIVE_BINARY = 0x78000016,///< [Validation] native binary is not supported by the device
    ZER_RESULT_ERROR_INVALID_GLOBAL_NAME = 0x78000017,  ///< [Validation] global variable is not found in the module
    ZER_RESULT_ERROR_INVALID_KERNEL_NAME = 0x78000018,  ///< [Validation] kernel name is not found in the module
    ZER_RESULT_ERROR_INVALID_FUNCTION_NAME = 0x78000019,///< [Validation] function name is not found in the module
    ZER_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 0x7800001a, ///< [Validation] group size dimension is not valid for the kernel or
                                                    ///< device
    ZER_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 0x7800001b,   ///< [Validation] global width dimension is not valid for the kernel or
                                                    ///< device
    ZER_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 0x7800001c,///< [Validation] kernel argument index is not valid for kernel
    ZER_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 0x7800001d, ///< [Validation] kernel argument size does not match kernel
    ZER_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 0x7800001e,   ///< [Validation] value of kernel attribute is not valid for the kernel or
                                                    ///< device
    ZER_RESULT_ERROR_INVALID_MODULE_UNLINKED = 0x7800001f,  ///< [Validation] module with imports needs to be linked before kernels can
                                                    ///< be created from it.
    ZER_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 0x78000020,///< [Validation] command list type does not match command queue type
    ZER_RESULT_ERROR_OVERLAPPING_REGIONS = 0x78000021,  ///< [Validation] copy operations do not support overlapping regions of
                                                    ///< memory
    ZER_RESULT_INVALID_HOST_PTR = 0x78000022,       ///< Invalid host pointer
    ZER_RESULT_INVALID_USM_SIZE = 0x78000023,       ///< Invalid USM size
    ZER_RESULT_OBJECT_ALLOCATION_FAILURE = 0x78000024,  ///< Objection allocation failure
    ZER_RESULT_ERROR_UNKNOWN = 0x7ffffffe,          ///< Unknown or internal error
    ZER_RESULT_FORCE_UINT32 = 0x7fffffff

} zer_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _zer_structure_type_t
{
    ZER_STRUCTURE_TYPE_IMAGE_DESC = 0,              ///< $zer_image_desc_t
    ZER_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct _zer_base_properties_t
{
    zer_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure

} zer_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct _zer_base_desc_t
{
    zer_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure

} zer_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3D offset argument passed to buffer rect operations
typedef struct _zer_rect_offset_t
{
    uint64_t x;                                     ///< [in] x offset (bytes)
    uint64_t y;                                     ///< [in] y offset (scalar)
    uint64_t z;                                     ///< [in] z offset (scalar)

} zer_rect_offset_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3D region argument passed to buffer rect operations
typedef struct _zer_rect_region_t
{
    uint64_t width;                                 ///< [in] width (bytes)
    uint64_t height;                                ///< [in] height (scalar)
    uint64_t depth;                                 ///< [in] scalar (scalar)

} zer_rect_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_base_properties_t
typedef struct _zer_base_properties_t zer_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_base_desc_t
typedef struct _zer_base_desc_t zer_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_rect_offset_t
typedef struct _zer_rect_offset_t zer_rect_offset_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_rect_region_t
typedef struct _zer_rect_region_t zer_rect_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_image_format_t
typedef struct _zer_image_format_t zer_image_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_image_desc_t
typedef struct _zer_image_desc_t zer_image_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_buffer_region_t
typedef struct _zer_buffer_region_t zer_buffer_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_sampler_property_value_t
typedef struct _zer_sampler_property_value_t zer_sampler_property_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zer_device_partition_property_value_t
typedef struct _zer_device_partition_property_value_t zer_device_partition_property_value_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Context
#if !defined(__GNUC__)
#pragma region context
#endif
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
///       subsequent call to ::zerContextRelease.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @remarks
///   _Analogues_
///     - **clCreateContext**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phDevices`
///         + `nullptr == phContext`
///     - ::ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerContextCreate(
    uint32_t DeviceCount,                           ///< [in] the number of devices given in phDevices
    zer_device_handle_t* phDevices,                 ///< [in][range(0, DeviceCount)] array of handle of devices.
    zer_context_handle_t* phContext                 ///< [out] pointer to handle of context object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes a reference of the context handle indicating it's in use until
///        paired ::zerContextRelease is called
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerContextGetReference(
    zer_context_handle_t hContext                   ///< [in] handle of the context to get a reference of.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported context info
typedef enum _zer_context_info_t
{
    ZER_CONTEXT_INFO_NUM_DEVICES = 1,               ///< [uint32_t] The number of the devices in the context
    ZER_CONTEXT_INFO_DEVICES = 2,                   ///< [::zer_context_handle_t...] The array of the device handles in the
                                                    ///< context
    ZER_CONTEXT_INFO_FORCE_UINT32 = 0x7fffffff

} zer_context_info_t;

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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerContextRelease(
    zer_context_handle_t hContext                   ///< [in] handle of the context to release.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_CONTEXT_INFO_DEVICES < ContextInfoType`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerContextGetInfo(
    zer_context_handle_t hContext,                  ///< [in] handle of the context
    zer_context_info_t ContextInfoType,             ///< [in] type of the info to retrieve
    size_t* pSize,                                  ///< [in,out] pointer to the number of bytes needed to return info queried.
                                                    ///< the call shall update it with the real number of bytes needed to
                                                    ///< return the info
    void* pContextInfo                              ///< [out][optional] array of bytes holding the info.
                                                    ///< if *pSize is not equal to the real number of bytes needed to return
                                                    ///< the info then the ::ZER_RESULT_ERROR_INVALID_SIZE error is returned
                                                    ///< and pContextInfo is not used.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeContext`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerContextGetNativeHandle(
    zer_context_handle_t hContext,                  ///< [in] handle of the context.
    zer_native_handle_t* phNativeContext            ///< [out] a pointer to the native handle of the context.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///         + `nullptr == hNativeContext`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phContext`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerContextCreateWithNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_native_handle_t hNativeContext,             ///< [in] the native handle of the context.
    zer_context_handle_t* phContext                 ///< [out] pointer to the handle of the context object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hKernel`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == globalWorkOffset`
///         + `nullptr == globalWorkSize`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_KERNEL
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_WORK_DIMENSION
///     - ::ZER_RESULT_INVALID_WORK_GROUP_SIZE
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueKernelLaunch(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the kernel object
    uint32_t workDim,                               ///< [in] number of dimensions, from 1 to 3, to specify the global and
                                                    ///< work-group work-items
    const size_t* globalWorkOffset,                 ///< [in] pointer to an array of workDim unsigned values that specify the
                                                    ///< offset used to calculate the global ID of a work-item
    const size_t* globalWorkSize,                   ///< [in] pointer to an array of workDim unsigned values that specify the
                                                    ///< number of global work-items in workDim that will execute the kernel
                                                    ///< function
    const size_t* localWorkSize,                    ///< [in][optional] pointer to an array of workDim unsigned values that
                                                    ///< specify the number of local work-items forming a work-group that will
                                                    ///< execute the kernel function.
                                                    ///< If nullptr, the runtime implementation will choose the work-group
                                                    ///< size. 
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before the kernel execution.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                                    ///< event. 
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular kernel
                                                    ///< execution instance.
                                                    ///< Contrary to clEnqueueNDRangeKernel, its input can not be a nullptr. 
                                                    ///< TODO: change to allow nullptr.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueEventsWait(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                                    ///< previously enqueued commands
                                                    ///< must be complete. 
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueEventsWaitWithBarrier(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                                    ///< previously enqueued commands
                                                    ///< must be complete. 
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hBuffer`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dst`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferRead(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hBuffer,                       ///< [in] handle of the buffer object
    bool blockingRead,                              ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                                  ///< [in] offset in bytes in the buffer object
    size_t size,                                    ///< [in] size in bytes of data being read
    void* dst,                                      ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hBuffer`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == src`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferWrite(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hBuffer,                       ///< [in] handle of the buffer object
    bool blockingWrite,                             ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                                  ///< [in] offset in bytes in the buffer object
    size_t size,                                    ///< [in] size in bytes of data being written
    const void* src,                                ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hBuffer`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dst`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferReadRect(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hBuffer,                       ///< [in] handle of the buffer object
    bool blockingRead,                              ///< [in] indicates blocking (true), non-blocking (false)
    zer_rect_offset_t bufferOffset,                 ///< [in] 3D offset in the buffer
    zer_rect_offset_t hostOffset,                   ///< [in] 3D offset in the host region
    zer_rect_region_t region,                       ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                          ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                        ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t hostRowPitch,                            ///< [in] length of each row in bytes in the host memory region pointed by
                                                    ///< dst
    size_t hostSlicePitch,                          ///< [in] length of each 2D slice in bytes in the host memory region
                                                    ///< pointed by dst
    void* dst,                                      ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hBuffer`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == src`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferWriteRect(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hBuffer,                       ///< [in] handle of the buffer object
    bool blockingWrite,                             ///< [in] indicates blocking (true), non-blocking (false)
    zer_rect_offset_t bufferOffset,                 ///< [in] 3D offset in the buffer
    zer_rect_offset_t hostOffset,                   ///< [in] 3D offset in the host region
    zer_rect_region_t region,                       ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                          ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                        ///< [in] length of each 2D slice in bytes in the buffer object being
                                                    ///< written
    size_t hostRowPitch,                            ///< [in] length of each row in bytes in the host memory region pointed by
                                                    ///< src
    size_t hostSlicePitch,                          ///< [in] length of each 2D slice in bytes in the host memory region
                                                    ///< pointed by src
    void* src,                                      ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy from a buffer object to another
/// 
/// @details
///     - The source and destination 2D or 3D rectangular regions can have
///       different shapes.
/// 
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyBuffer**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hBufferSrc`
///         + `nullptr == hBufferDst`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferCopy(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hBufferSrc,                    ///< [in] handle of the src buffer object
    zer_mem_handle_t hBufferDst,                    ///< [in] handle of the dest buffer object
    size_t size,                                    ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hBufferSrc`
///         + `nullptr == hBufferDst`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferCopyRect(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hBufferSrc,                    ///< [in] handle of the source buffer object
    zer_mem_handle_t hBufferDst,                    ///< [in] handle of the dest buffer object
    zer_rect_offset_t srcOrigin,                    ///< [in] 3D offset in the source buffer
    zer_rect_offset_t dstOrigin,                    ///< [in] 3D offset in the destination buffer
    zer_rect_region_t srcRegion,                    ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t srcRowPitch,                             ///< [in] length of each row in bytes in the source buffer object
    size_t srcSlicePitch,                           ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t dstRowPitch,                             ///< [in] length of each row in bytes in the destination buffer object
    size_t dstSlicePitch,                           ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hBuffer`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pattern`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferFill(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hBuffer,                       ///< [in] handle of the buffer object
    const void* pattern,                            ///< [in] pointer to the fill pattern
    size_t patternSize,                             ///< [in] size in bytes of the pattern
    size_t offset,                                  ///< [in] offset into the buffer
    size_t size,                                    ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hImage`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dst`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemImageRead(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hImage,                        ///< [in] handle of the image object
    bool blockingRead,                              ///< [in] indicates blocking (true), non-blocking (false)
    zer_rect_offset_t origin,                       ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    zer_rect_region_t region,                       ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                                    ///< image
    size_t rowPitch,                                ///< [in] length of each row in bytes
    size_t slicePitch,                              ///< [in] length of each 2D slice of the 3D image
    void* dst,                                      ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hImage`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == src`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemImageWrite(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hImage,                        ///< [in] handle of the image object
    bool blockingWrite,                             ///< [in] indicates blocking (true), non-blocking (false)
    zer_rect_offset_t origin,                       ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    zer_rect_region_t region,                       ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                                    ///< image
    size_t inputRowPitch,                           ///< [in] length of each row in bytes
    size_t inputSlicePitch,                         ///< [in] length of each 2D slice of the 3D image
    void* src,                                      ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy from an image object to another
/// 
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyImage**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hImageSrc`
///         + `nullptr == hImageDst`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemImageCopy(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hImageSrc,                     ///< [in] handle of the src image object
    zer_mem_handle_t hImageDst,                     ///< [in] handle of the dest image object
    zer_rect_offset_t srcOrigin,                    ///< [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
                                                    ///< image
    zer_rect_offset_t dstOrigin,                    ///< [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
                                                    ///< or 3D image
    zer_rect_region_t region,                       ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                                    ///< image
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Map flags
typedef uint32_t zer_map_flags_t;
typedef enum _zer_map_flag_t
{
    ZER_MAP_FLAG_READ = ZER_BIT(0),                 ///< Map for read access
    ZER_MAP_FLAG_WRITE = ZER_BIT(1),                ///< Map for write access
    ZER_MAP_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_map_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Map flags
typedef uint32_t zer_usm_migration_flags_t;
typedef enum _zer_usm_migration_flag_t
{
    ZER_USM_MIGRATION_FLAG_DEFAULT = ZER_BIT(0),    ///< Default migration TODO: Add more enums! 
    ZER_USM_MIGRATION_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_usm_migration_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to map a region of the buffer object into the host
///        address space and return a pointer to the mapped region
/// 
/// @details
///     - Input parameter blockingMap indicates if the map is blocking or
///       non-blocking.
///     - Currently, no direct support in Leverl Zero. Implemented as a shared
///       allocation followed by copying on discrete GPU
///     - TODO: add a driver function in Level Zero?
/// 
/// @remarks
///   _Analogues_
///     - **clEnqueueMapBuffer**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == buffer`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < mapFlags`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///         + `nullptr == retMap`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemBufferMap(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t buffer,                        ///< [in] handle of the buffer object
    bool blockingMap,                               ///< [in] indicates blocking (true), non-blocking (false)
    zer_map_flags_t mapFlags,                       ///< [in] flags for read, write, readwrite mapping
    size_t offset,                                  ///< [in] offset in bytes of the buffer region being mapped
    size_t size,                                    ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event,                      ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    void** retMap                                   ///< [in,out] return mapped pointer.  TODO: move it before
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hMem`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == mappedPtr`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueMemUnmap(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_mem_handle_t hMem,                          ///< [in] handle of the memory (buffer or image) object
    void* mappedPtr,                                ///< [in] mapped host address
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to set USM memory object value
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueUSMMemset(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    void* ptr,                                      ///< [in] pointer to USM memory object
    int8_t byteValue,                               ///< [in] byte value to fill
    size_t count,                                   ///< [in] size in bytes to be set
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy USM memory
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == dstPrt`
///         + `nullptr == srcPrt`
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueUSMMemcpy(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    bool blocking,                                  ///< [in] blocking or non-blocking copy
    void* dstPrt,                                   ///< [in] pointer to the destination USM memory object
    const void* srcPrt,                             ///< [in] pointer to the source USM memory object
    size_t size,                                    ///< [in] size in bytes to be copied
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to prefetch USM memory
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == event`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < flags`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueUSMPrefetch(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    const void* ptr,                                ///< [in] pointer to the USM memory object
    size_t size,                                    ///< [in] size in bytes to be fetched
    zer_usm_migration_flags_t flags,                ///< [in] USM prefetch flags
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const zer_event_handle_t* eventWaitList,        ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory advice
typedef enum _zer_mem_advice_t
{
    ZER_MEM_ADVICE_MEM_ADVICE_DEFAULT = 0,          ///< The USM memory advice is default
    ZER_MEM_ADVICE_FORCE_UINT32 = 0x7fffffff

} zer_mem_advice_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to set USM memory advice
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == event`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_MEM_ADVICE_MEM_ADVICE_DEFAULT < advice`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEnqueueUSMMemAdvice(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    const void* ptr,                                ///< [in] pointer to the USM memory object
    size_t size,                                    ///< [in] size in bytes to be adviced
    zer_mem_advice_t advice,                        ///< [in] USM memory advice
    zer_event_handle_t* event                       ///< [in,out] return an event object that identifies this particular
                                                    ///< command instance.
                                                    ///< Input can not be a nullptr.
                                                    ///< TODO: change to allow nullptr. 
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs
#if !defined(__GNUC__)
#pragma region event
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Event query information type
typedef enum _zer_event_info_t
{
    ZER_EVENT_INFO_EVENT_INFO_COMMAND_QUEUE = 0,    ///< Command queue information of an event object
    ZER_EVENT_INFO_EVENT_INFO_CONTEXT = 1,          ///< Context information of an event object
    ZER_EVENT_INFO_EVENT_INFO_COMMAND_TYPE = 2,     ///< Command type information of an event object
    ZER_EVENT_INFO_EVENT_INFO_COMMAND_EXECUTION_STATUS = 3, ///< Command execution status of an event object
    ZER_EVENT_INFO_EVENT_INFO_REFERENCE_COUNT = 4,  ///< Reference count of an event object
    ZER_EVENT_INFO_FORCE_UINT32 = 0x7fffffff

} zer_event_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profiling query information type
typedef enum _zer_profiling_info_t
{
    ZER_PROFILING_INFO_PROFILING_INFO_COMMAND_QUEUED = 0,   ///< A 64-bit value of current device counter in nanoseconds when the event
                                                    ///< is enqueued
    ZER_PROFILING_INFO_PROFILING_INFO_COMMAND_SUBMIT = 1,   ///< A 64-bit value of current device counter in nanoseconds when the event
                                                    ///< is submitted
    ZER_PROFILING_INFO_PROFILING_INFO_COMMAND_START = 2,///< A 64-bit value of current device counter in nanoseconds when the event
                                                    ///< starts execution
    ZER_PROFILING_INFO_PROFILING_INFO_COMMAND_END = 3,  ///< A 64-bit value of current device counter in nanoseconds when the event
                                                    ///< has finished execution
    ZER_PROFILING_INFO_FORCE_UINT32 = 0x7fffffff

} zer_profiling_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create an event object. Events allow applications to enqueue commands
///        that wait on an event to finish or signal a command completion.
/// 
/// @remarks
///   _Analogues_
///     - **clCreateUserEvent**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == context`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pEvent`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventCreate(
    zer_context_handle_t context,                   ///< [in] handle of the context object
    zer_event_handle_t* pEvent                      ///< [out] pointer to handle of the event object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get event object information
/// 
/// @remarks
///   _Analogues_
///     - **clGetEventInfo**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == event`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_EVENT_INFO_EVENT_INFO_REFERENCE_COUNT < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propValue`
///         + `nullptr == propValueSizeRet`
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_OUT_OF_RESOURCES
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventGetInfo(
    zer_event_handle_t event,                       ///< [in] handle of the event object
    zer_event_info_t propName,                      ///< [in] the name of the event property to query
    size_t propValueSize,                           ///< [in] size in bytes of the event property value
    void* propValue,                                ///< [out] value of the event property
    size_t* propValueSizeRet                        ///< [out] bytes returned in event property
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == event`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_PROFILING_INFO_PROFILING_INFO_COMMAND_END < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propValue`
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_OUT_OF_RESOURCES
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventGetProfilingInfo(
    zer_event_handle_t event,                       ///< [in] handle of the event object
    zer_profiling_info_t propName,                  ///< [in] the name of the profiling property to query
    size_t propValueSize,                           ///< [in] size in bytes of the profiling property value
    void* propValue,                                ///< [out] value of the profiling property
    size_t propValueSizeRet                         ///< [out] bytes returned in profiling property
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for a list of events to finish.
/// 
/// @remarks
///   _Analogues_
///     - **clWaitForEvent**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == eventList`
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventWait(
    uint32_t numEvents,                             ///< [in] number of events in the event list
    const zer_event_handle_t* eventList             ///< [in][range(0, numEvents)] pointer to a list of events to wait for
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_OUT_OF_RESOURCES
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventGetReference(
    zer_event_handle_t event                        ///< [in] handle of the event object
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == event`
///     - ::ZER_RESULT_INVALID_EVENT
///     - ::ZER_RESULT_OUT_OF_RESOURCES
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventRelease(
    zer_event_handle_t event                        ///< [in] handle of the event object
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEvent`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeEvent`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventGetNativeHandle(
    zer_event_handle_t hEvent,                      ///< [in] handle of the event.
    zer_native_handle_t* phNativeEvent              ///< [out] a pointer to the native handle of the event.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///         + `nullptr == hNativeEvent`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phEvent`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerEventCreateWithNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_native_handle_t hNativeEvent,               ///< [in] the native handle of the event.
    zer_event_handle_t* phEvent                     ///< [out] pointer to the handle of the event object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs
#if !defined(__GNUC__)
#pragma region memory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Memory flags
typedef uint32_t zer_mem_flags_t;
typedef enum _zer_mem_flag_t
{
    ZER_MEM_FLAG_MEM_READ_WRITE = ZER_BIT(0),       ///< The memory object will be read and written by a kernel. This is the
                                                    ///< default
    ZER_MEM_FLAG_MEM_WRITE_ONLY = ZER_BIT(1),       ///< The memory object will be written but not read by a kernel
    ZER_MEM_FLAG_MEM_READ_ONLY = ZER_BIT(2),        ///< The memory object is a read-only inside a kernel
    ZER_MEM_FLAG_MEM_USE_HOST_POINTER = ZER_BIT(3), ///< Use memory pointed by a host pointer parameter as the storage bits for
                                                    ///< the memory object
    ZER_MEM_FLAG_MEM_ALLOC_HOST_POINTER = ZER_BIT(4),   ///< Allocate memory object from host accessible memory
    ZER_MEM_FLAG_MEM_ALLOC_COPY_HOST_POINTER = ZER_BIT(5),  ///< Allocate memory and copy the data from host pointer pointed memory
    ZER_MEM_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_mem_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory types
typedef enum _zer_image_type_t
{
    ZER_IMAGE_TYPE_MEM_TYPE_BUFFER = 0,             ///< Buffer object
    ZER_IMAGE_TYPE_MEM_TYPE_IMAGE2D = 1,            ///< 2D image object
    ZER_IMAGE_TYPE_MEM_TYPE_IMAGE3D = 2,            ///< 3D image object
    ZER_IMAGE_TYPE_MEM_TYPE_IMAGE2D_ARRAY = 3,      ///< 2D image array object
    ZER_IMAGE_TYPE_MEM_TYPE_IMAGE1D = 4,            ///< 1D image object
    ZER_IMAGE_TYPE_MEM_TYPE_IMAGE1D_ARRAY = 5,      ///< 1D image array object
    ZER_IMAGE_TYPE_MEM_TYPE_IMAGE1D_BUFFER = 6,     ///< 1D image buffer object
    ZER_IMAGE_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_image_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image channel order info: number of channels and the channel layout
typedef enum _zer_image_channel_order_t
{
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_A = 0,    ///< channel order A
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_R = 1,    ///< channel order R
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_RG = 2,   ///< channel order RG
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_RA = 3,   ///< channel order RA
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_RGB = 4,  ///< channel order RGB
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_RGBA = 5, ///< channel order RGBA
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_BGRA = 6, ///< channel order BGRA
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_ARGB = 7, ///< channel order ARGB
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_INTENSITY = 8,///< channel order intensity
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_LUMINANCE = 9,///< channel order luminance
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_RX = 10,  ///< channel order Rx
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_RGX = 11, ///< channel order RGx
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_RGBX = 12,///< channel order RGBx
    ZER_IMAGE_CHANNEL_ORDER_CHANNEL_ORDER_SRGBA = 13,   ///< channel order sRGBA
    ZER_IMAGE_CHANNEL_ORDER_FORCE_UINT32 = 0x7fffffff

} zer_image_channel_order_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image channel type info: describe the size of the channel data type
typedef enum _zer_image_channel_type_t
{
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_SNORM_INT8 = 0, ///< channel type snorm int8
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_SNORM_INT16 = 1,///< channel type snorm int16
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_UNORM_INT8 = 2, ///< channel type unorm int8
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_UNORM_INT16 = 3,///< channel type unorm int16
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_UNORM_SHORT_565 = 4,///< channel type unorm short 565
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_UNORM_SHORT_555 = 5,///< channel type unorm short 555
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_INT_101010 = 6, ///< channel type int 101010
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_SIGNED_INT8 = 7,///< channel type signed int8
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_SIGNED_INT16 = 8,   ///< channel type signed int16
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_SIGNED_INT32 = 9,   ///< channel type signed int32
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_UNSIGNED_INT8 = 10, ///< channel type unsigned int8
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_UNSIGNED_INT16 = 11,///< channel type unsigned int16
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_UNSIGNED_INT32 = 12,///< channel type unsigned int32
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_HALF_FLOAT = 13,///< channel type half float
    ZER_IMAGE_CHANNEL_TYPE_CHANNEL_TYPE_FLOAT = 14, ///< channel type float
    ZER_IMAGE_CHANNEL_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_image_channel_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image format including channel layout and data type
typedef struct _zer_image_format_t
{
    zer_image_channel_order_t channelOrder;         ///< [in] image channel order
    zer_image_channel_type_t channelType;           ///< [in] image channel type

} zer_image_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image descriptor type.
typedef struct _zer_image_desc_t
{
    zer_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zer_image_type_t type;                          ///< [in] image type
    size_t width;                                   ///< [in] image width
    size_t height;                                  ///< [in] image height
    size_t depth;                                   ///< [in] image depth
    size_t arraySize;                               ///< [in] image array size
    size_t rowPitch;                                ///< [in] image row pitch
    size_t slicePitch;                              ///< [in] image slice pitch
    uint32_t numMipLevel;                           ///< [in] number of MIP levels
    uint32_t numSamples;                            ///< [in] number of samples

} zer_image_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create an image object
/// 
/// @remarks
///   _Analogues_
///     - **clCreateImage**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
///         + `::ZER_IMAGE_TYPE_MEM_TYPE_IMAGE1D_BUFFER < imageDesc->type`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == imageFormat`
///         + `nullptr == imageDesc`
///         + `nullptr == hostPtr`
///         + `nullptr == phMem`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_IMAGE_FORMAT_DESCRIPTOR
///     - ::ZER_RESULT_INVALID_IMAGE_SIZE
///     - ::ZER_RESULT_INVALID_OPERATION
///     - ::ZER_RESULT_INVALID_HOST_PTR
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemImageCreate(
    zer_context_handle_t hContext,                  ///< [in] handle of the context object
    zer_mem_flags_t flags,                          ///< [in] allocation and usage information flags
    const zer_image_format_t* imageFormat,          ///< [in] pointer to image format specification
    const zer_image_desc_t* imageDesc,              ///< [in] pointer to image description
    void* hostPtr,                                  ///< [in] pointer to the buffer data
    zer_mem_handle_t* phMem                         ///< [out] pointer to handle of image object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a memory buffer
/// 
/// @remarks
///   _Analogues_
///     - **clCreateBuffer**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == hostPtr`
///         + `nullptr == phBuffer`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_BUFFER_SIZE
///     - ::ZER_RESULT_INVALID_HOST_PTR
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemBufferCreate(
    zer_context_handle_t hContext,                  ///< [in] handle of the context object
    zer_mem_flags_t flags,                          ///< [in] allocation and usage information flags
    size_t size,                                    ///< [in] size in bytes of the memory object to be allocated
    void* hostPtr,                                  ///< [in] pointer to the buffer data
    zer_mem_handle_t* phBuffer                      ///< [out] pointer to handle of the memory buffer created
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMem`
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemGetReference(
    zer_mem_handle_t hMem                           ///< [in] handle of the memory object to get access
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMem`
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemRelease(
    zer_mem_handle_t hMem                           ///< [in] handle of the memory object to release
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer region type, used to describe a sub buffer
typedef struct _zer_buffer_region_t
{
    size_t origin;                                  ///< [in] buffer origin offset
    size_t size;                                    ///< [in] size of the buffer region

} zer_buffer_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer creation type
typedef enum _zer_buffer_create_type_t
{
    ZER_BUFFER_CREATE_TYPE_BUFFER_CREATE_TYPE_REGION = 0,   ///< buffer create type is region
    ZER_BUFFER_CREATE_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_buffer_create_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a sub buffer representing a region in an existing buffer
/// 
/// @remarks
///   _Analogues_
///     - **clCreateSubBuffer**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hBuffer`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
///         + `::ZER_BUFFER_CREATE_TYPE_BUFFER_CREATE_TYPE_REGION < bufferCreateType`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pBufferCreateInfo`
///         + `nullptr == phMem`
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OBJECT_ALLOCATION_FAILURE
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_BUFFER_SIZE
///     - ::ZER_RESULT_INVALID_HOST_PTR
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemBufferPartition(
    zer_mem_handle_t hBuffer,                       ///< [in] handle of the buffer object to allocate from
    zer_mem_flags_t flags,                          ///< [in] allocation and usage information flags
    zer_buffer_create_type_t bufferCreateType,      ///< [in] buffer creation type
    zer_buffer_region_t* pBufferCreateInfo,         ///< [in] pointer to buffer create region information
    zer_mem_handle_t* phMem                         ///< [out] pointer to the handle of sub buffer created
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMem`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeMem`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemGetNativeHandle(
    zer_mem_handle_t hMem,                          ///< [in] handle of the mem.
    zer_native_handle_t* phNativeMem                ///< [out] a pointer to the native handle of the mem.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///         + `nullptr == hNativeMem`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phMem`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemCreateWithNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_native_handle_t hNativeMem,                 ///< [in] the native handle of the mem.
    zer_mem_handle_t* phMem                         ///< [out] pointer to the handle of the mem object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs
#if !defined(__GNUC__)
#pragma region misc
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Tear down L0 runtime instance and release all its resources
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pParams`
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerTearDown(
    void* pParams                                   ///< [in] pointer to tear down parameters
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs
#if !defined(__GNUC__)
#pragma region queue
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Query queue info
typedef enum _zer_queue_info_t
{
    ZER_QUEUE_INFO_CONTEXT = 0,                     ///< Queue context info
    ZER_QUEUE_INFO_DEVICE = 1,                      ///< Queue device info
    ZER_QUEUE_INFO_DEVICE_DEFAULT = 2,              ///< Queue device default info
    ZER_QUEUE_INFO_PROPERTIES = 3,                  ///< Queue properties info
    ZER_QUEUE_INFO_REFERENCE_COUNT = 4,             ///< Queue reference count
    ZER_QUEUE_INFO_SIZE = 5,                        ///< Queue size info
    ZER_QUEUE_INFO_FORCE_UINT32 = 0x7fffffff

} zer_queue_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue properties
typedef uint32_t zer_queue_flags_t;
typedef enum _zer_queue_flag_t
{
    ZER_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE = ZER_BIT(0),  ///< Enable/disable out of order execution
    ZER_QUEUE_FLAG_PROFILING_ENABLE = ZER_BIT(1),   ///< Enable/disable profiling
    ZER_QUEUE_FLAG_ON_DEVICE = ZER_BIT(2),          ///< Is a device queue
    ZER_QUEUE_FLAG_ON_DEVICE_DEFAULT = ZER_BIT(3),  ///< Is the default queue for a device
    ZER_QUEUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_queue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a command queue
/// 
/// @remarks
///   _Analogues_
///     - **clGetCommandQueueInfo**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_QUEUE_INFO_SIZE < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propValue`
///         + `nullptr == pSize`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerQueueGetInfo(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue object
    zer_queue_info_t propName,                      ///< [in] name of the queue property to query
    size_t propValueSize,                           ///< [in] size in bytes of the queue property value provided
    void* propValue,                                ///< [out] value of the queue property
    size_t* pSize                                   ///< [out] size in bytes returned in queue property value
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a command queue for a device in a context
/// 
/// @remarks
///   _Analogues_
///     - **clCreateCommandQueueWithProperties**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `0xf < props`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phQueue`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_DEVICE
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_QUEUE_PROPERTIES
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerQueueCreate(
    zer_context_handle_t hContext,                  ///< [in] handle of the context object
    zer_device_handle_t hDevice,                    ///< [in] handle of the device object
    zer_queue_flags_t props,                        ///< [in] initialization properties.
                                                    ///< must be 0 (default) or a combination of ::zer_queue_flags_t.
    zer_queue_handle_t* phQueue                     ///< [out] pointer to handle of queue object created
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerQueueGetReference(
    zer_queue_handle_t hQueue                       ///< [in] handle of the queue object to get access
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_INVALID_QUEUE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerQueueRelease(
    zer_queue_handle_t hQueue                       ///< [in] handle of the queue object to release
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeQueue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerQueueGetNativeHandle(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue.
    zer_native_handle_t* phNativeQueue              ///< [out] a pointer to the native handle of the queue.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hQueue`
///         + `nullptr == hNativeQueue`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phQueue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerQueueCreateWithNativeHandle(
    zer_queue_handle_t hQueue,                      ///< [in] handle of the queue instance
    zer_native_handle_t hNativeQueue,               ///< [in] the native handle of the queue.
    zer_queue_handle_t* phQueue                     ///< [out] pointer to the handle of the queue object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs
#if !defined(__GNUC__)
#pragma region sampler
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Get sample object information
typedef enum _zer_sampler_info_t
{
    ZER_SAMPLER_INFO_SAMPLER_INFO_REFERENCE_COUNT = 0,  ///< Sampler reference count info
    ZER_SAMPLER_INFO_SAMPLER_INFO_CONTEXT = 1,      ///< Sampler context info
    ZER_SAMPLER_INFO_SAMPLER_INFO_NORMALIZED_COORDS = 2,///< Sampler normalized coordindate setting
    ZER_SAMPLER_INFO_SAMPLER_INFO_ADDRESSING_MODE = 3,  ///< Sampler addressing mode setting
    ZER_SAMPLER_INFO_SAMPLER_INFO_FILTER_MODE = 4,  ///< Sampler filter mode setting
    ZER_SAMPLER_INFO_SAMPLER_INFO_MIP_FILTER_MODE = 5,  ///< Sampler MIP filter mode setting
    ZER_SAMPLER_INFO_SAMPLER_INFO_LOD_MIN = 6,      ///< Sampler LOD Min value
    ZER_SAMPLER_INFO_SAMPLER_INFO_LOD_MAX = 7,      ///< Sampler LOD Max value
    ZER_SAMPLER_INFO_FORCE_UINT32 = 0x7fffffff

} zer_sampler_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler properties
typedef enum _zer_sampler_properties_t
{
    ZER_SAMPLER_PROPERTIES_SAMPLER_PROPERTIES_NORMALIZED_COORDS = 0,///< Sampler normalized coordinates
    ZER_SAMPLER_PROPERTIES_SAMPLER_PROPERTIES_ADDRESSING_MODE = 1,  ///< Sampler addressing mode
    ZER_SAMPLER_PROPERTIES_SAMPLER_PROPERTIES_FILTER_MODE = 2,  ///< Sampler filter mode
    ZER_SAMPLER_PROPERTIES_FORCE_UINT32 = 0x7fffffff

} zer_sampler_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler properties <name, value> pair
typedef struct _zer_sampler_property_value_t
{
    zer_sampler_properties_t propName;              ///< [in] Sampler property
    uint32_t propValue;                             ///< [in] Sampler property value

} zer_sampler_property_value_t;

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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == props`
///         + `nullptr == phSampler`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_OPERATION
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerSamplerCreate(
    zer_context_handle_t hContext,                  ///< [in] handle of the context object
    const zer_sampler_property_value_t* props,      ///< [in] specifies a list of sampler property names and their
                                                    ///< corresponding values.
    zer_sampler_handle_t* phSampler                 ///< [out] pointer to handle of sampler object created
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hSampler`
///     - ::ZER_RESULT_INVALID_SAMPLER
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerSamplerGetReference(
    zer_sampler_handle_t hSampler                   ///< [in] handle of the sampler object to get access
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hSampler`
///     - ::ZER_RESULT_INVALID_SAMPLER
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerSamplerRelease(
    zer_sampler_handle_t hSampler                   ///< [in] handle of the sampler object to release
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a sampler object
/// 
/// @remarks
///   _Analogues_
///     - **clGetSamplerInfo**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hSampler`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_SAMPLER_INFO_SAMPLER_INFO_LOD_MAX < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propValue`
///         + `nullptr == pSize`
///     - ::ZER_RESULT_INVALID_SAMPLER
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerSamplerGetInfo(
    zer_sampler_handle_t hSampler,                  ///< [in] handle of the sampler object
    zer_sampler_info_t propName,                    ///< [in] name of the sampler property to query
    size_t propValueSize,                           ///< [in] size in bytes of the sampler property value provided
    void* propValue,                                ///< [out] value of the sampler property
    size_t* pSize                                   ///< [out] size in bytes returned in sampler property value
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hSampler`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeSampler`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerSamplerGetNativeHandle(
    zer_sampler_handle_t hSampler,                  ///< [in] handle of the sampler.
    zer_native_handle_t* phNativeSampler            ///< [out] a pointer to the native handle of the sampler.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hSampler`
///         + `nullptr == hNativeSampler`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phSampler`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerSamplerCreateWithNativeHandle(
    zer_sampler_handle_t hSampler,                  ///< [in] handle of the sampler instance
    zer_native_handle_t hNativeSampler,             ///< [in] the native handle of the sampler.
    zer_sampler_handle_t* phSampler                 ///< [out] pointer to the handle of the sampler object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero APIs
#if !defined(__GNUC__)
#pragma region usm
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory property flags
typedef uint32_t zer_usm_mem_flags_t;
typedef enum _zer_usm_mem_flag_t
{
    ZER_USM_MEM_FLAG_MEM_ALLOC_FLAGS_INTEL = ZER_BIT(0),///< The USM memory allocation is from Intel USM
    ZER_USM_MEM_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_usm_mem_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory allocation information type
typedef enum _zer_mem_info_t
{
    ZER_MEM_INFO_MEM_ALLOC_TYPE = 0,                ///< Memory allocation type info
    ZER_MEM_INFO_MEM_ALLOC_BASE_PTR = 1,            ///< Memory allocation base pointer info
    ZER_MEM_INFO_MEM_ALLOC_SIZE = 2,                ///< Memory allocation size info
    ZER_MEM_INFO_MEM_ALLOC_DEVICE = 3,              ///< Memory allocation device info
    ZER_MEM_INFO_FORCE_UINT32 = 0x7fffffff

} zer_mem_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate host memory
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == context`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pUSMFlag`
///         + `nullptr == pptr`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_USM_SIZE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerUSMHostAlloc(
    zer_context_handle_t context,                   ///< [in] handle of the context object
    zer_usm_mem_flags_t* pUSMFlag,                  ///< [in] USM memory allocation flags
    size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,                                 ///< [in] alignment of the USM memory object
    void** pptr                                     ///< [out] pointer to USM host memory object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate device memory
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == context`
///         + `nullptr == device`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pUSMProp`
///         + `nullptr == pptr`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_USM_SIZE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerUSMDeviceAlloc(
    zer_context_handle_t context,                   ///< [in] handle of the context object
    zer_device_handle_t device,                     ///< [in] handle of the device object
    zer_usm_mem_flags_t* pUSMProp,                  ///< [in] USM memory properties
    size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,                                 ///< [in] alignment of the USM memory object
    void** pptr                                     ///< [out] pointer to USM device memory object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate shared memory
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == context`
///         + `nullptr == device`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pUSMProp`
///         + `nullptr == pptr`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_USM_SIZE
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
///     - ::ZER_RESULT_OUT_OF_RESOURCES
ZER_APIEXPORT zer_result_t ZER_APICALL
zerUSMSharedAlloc(
    zer_context_handle_t context,                   ///< [in] handle of the context object
    zer_device_handle_t device,                     ///< [in] handle of the device object
    zer_usm_mem_flags_t* pUSMProp,                  ///< [in] USM memory properties
    size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,                                 ///< [in] alignment of the USM memory object
    void** pptr                                     ///< [out] pointer to USM shared memory object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Free the USM memory object
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == context`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemFree(
    zer_context_handle_t context,                   ///< [in] handle of the context object
    void* ptr                                       ///< [in] pointer to USM memory object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get USM memory object allocation information
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == context`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ptr`
///         + `nullptr == propValue`
///         + `nullptr == propValueSizeRet`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_MEM_INFO_MEM_ALLOC_DEVICE < propName`
///     - ::ZER_RESULT_INVALID_CONTEXT
///     - ::ZER_RESULT_INVALID_VALUE
///     - ::ZER_RESULT_INVALID_MEM_OBJECT
///     - ::ZER_RESULT_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerMemGetMemAllocInfo(
    zer_context_handle_t context,                   ///< [in] handle of the context object
    const void* ptr,                                ///< [in] pointer to USM memory object
    zer_mem_info_t propName,                        ///< [in] the name of the USM allocation property to query
    size_t propValueSize,                           ///< [in] size in bytes of the USM allocation property value
    void* propValue,                                ///< [out] value of the USM allocation property
    size_t* propValueSizeRet                        ///< [out] bytes returned in USM allocation property
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Device
#if !defined(__GNUC__)
#pragma region device
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum _zer_device_type_t
{
    ZER_DEVICE_TYPE_DEFAULT = 1,                    ///< The default device type as preferred by the runtime
    ZER_DEVICE_TYPE_ALL = 2,                        ///< Devices of all types
    ZER_DEVICE_TYPE_GPU = 3,                        ///< Graphics Processing Unit
    ZER_DEVICE_TYPE_CPU = 4,                        ///< Central Processing Unit
    ZER_DEVICE_TYPE_FPGA = 5,                       ///< Field Programmable Gate Array
    ZER_DEVICE_TYPE_MCA = 6,                        ///< Memory Copy Accelerator
    ZER_DEVICE_TYPE_VPU = 7,                        ///< Vision Processing Unit
    ZER_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_device_type_t;

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
///       with a subsequent call to ::zerDeviceRelease.
///     - The application may call this function from simultaneous threads, the
///       implementation must be thread-safe
/// 
/// @remarks
///   _Analogues_
///     - **clGetDeviceIDs**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_DEVICE_TYPE_VPU < DevicesType`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDeviceGet(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_device_type_t DevicesType,                  ///< [in] the type of the devices.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of devices.
                                                    ///< If count is zero, then the call shall update the value with the total
                                                    ///< number of devices available.
                                                    ///< If count is greater than the number of devices available, then the
                                                    ///< call shall update the value with the correct number of devices available.
    zer_device_handle_t* phDevices                  ///< [out][optional][range(0, *pCount)] array of handle of devices.
                                                    ///< If count is less than the number of devices available, then platform
                                                    ///< shall only retrieve that number of devices.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device info
typedef enum _zer_device_info_t
{
    ZER_DEVICE_INFO_TYPE = 0,                       ///< ::zer_device_type_t: type of the device
    ZER_DEVICE_INFO_VENDOR_ID = 1,                  ///< uint32_t: vendor Id of the device
    ZER_DEVICE_INFO_MAX_COMPUTE_UNITS = 2,          ///< uint32_t: the number of compute units
    ZER_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS = 3,   ///< uint32_t: max work item dimensions
    ZER_DEVICE_INFO_MAX_WORK_ITEM_SIZES = 4,        ///< size_t[]: return an array of max work item sizes
    ZER_DEVICE_INFO_MAX_WORK_GROUP_SIZE = 5,        ///< size_t: max work group size
    ZER_DEVICE_INFO_SINGLE_FP_CONFIG = 6,           ///< Return a bit field of ::zer_fp_capability_flags_t: single precision
                                                    ///< floating point capability
    ZER_DEVICE_INFO_HALF_FP_CONFIG = 7,             ///< Return a bit field of ::zer_fp_capability_flags_t: half precsion
                                                    ///< floating point capability
    ZER_DEVICE_INFO_DOUBLE_FP_CONFIG = 8,           ///< Return a bit field of ::zer_fp_capability_flags_t: double precision
                                                    ///< floating point capability
    ZER_DEVICE_INFO_QUEUE_PROPERTIES = 9,           ///< Return a bit field of ::zer_queue_flags_t: command queue properties
                                                    ///< supported by the device
    ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR = 10,   ///< uint32_t: preferred vector width for char
    ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT = 11,  ///< uint32_t: preferred vector width for short
    ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT = 12,///< uint32_t: preferred vector width for int
    ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG = 13,   ///< uint32_t: preferred vector width for long
    ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT = 14,  ///< uint32_t: preferred vector width for float
    ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE = 15, ///< uint32_t: preferred vector width for double
    ZER_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF = 16,   ///< uint32_t: preferred vector width for half float
    ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR = 17,  ///< uint32_t: native vector width for char
    ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT = 18, ///< uint32_t: native vector width for short
    ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT = 19,   ///< uint32_t: native vector width for int
    ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG = 20,  ///< uint32_t: native vector width for long
    ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT = 21, ///< uint32_t: native vector width for float
    ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE = 22,///< uint32_t: native vector width for double
    ZER_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF = 23,  ///< uint32_t: native vector width for half float
    ZER_DEVICE_INFO_MAX_CLOCK_FREQUENCY = 24,       ///< uint32_t: max clock frequency in MHz
    ZER_DEVICE_INFO_ADDRESS_BITS = 25,              ///< uint32_t: address bits
    ZER_DEVICE_INFO_MAX_MEM_ALLOC_SIZE = 26,        ///< uint64_t: max memory allocation size
    ZER_DEVICE_INFO_IMAGE_SUPPORTED = 27,           ///< bool: images are supported
    ZER_DEVICE_INFO_MAX_READ_IMAGE_ARGS = 28,       ///< uint32_t: max number of image objects arguments of a kernel declared
                                                    ///< with the read_only qualifier
    ZER_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS = 29,      ///< uint32_t: max number of image objects arguments of a kernel declared
                                                    ///< with the write_only qualifier
    ZER_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS = 30, ///< uint32_t: max number of image objects arguments of a kernel declared
                                                    ///< with the read_write qualifier
    ZER_DEVICE_INFO_IMAGE2D_MAX_WIDTH = 31,         ///< size_t: max width of Image2D object
    ZER_DEVICE_INFO_IMAGE2D_MAX_HEIGHT = 32,        ///< size_t: max heigh of Image2D object
    ZER_DEVICE_INFO_IMAGE3D_MAX_WIDTH = 33,         ///< size_t: max width of Image3D object
    ZER_DEVICE_INFO_IMAGE3D_MAX_HEIGHT = 34,        ///< size_t: max height of Image3D object
    ZER_DEVICE_INFO_IMAGE3D_MAX_DEPTH = 35,         ///< size_t: max depth of Image3D object
    ZER_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE = 36,     ///< size_t: max image buffer size
    ZER_DEVICE_INFO_IMAGE_MAX_ARRAR_SIZE = 37,      ///< size_t: max image array size
    ZER_DEVICE_INFO_MAX_SAMPLERS = 38,              ///< uint32_t: max number of samplers that can be used in a kernel
    ZER_DEVICE_INFO_MAX_PARAMETER_SIZE = 39,        ///< size_t: max size in bytes of all arguments passed to a kernel
    ZER_DEVICE_INFO_MEM_BASE_ADDR_ALIGN = 40,       ///< uint32_t: memory base address alignment
    ZER_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE = 41,     ///< ::zer_device_mem_cache_type_t: global memory cache type
    ZER_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE = 42, ///< uint32_t: global memory cache line size in bytes
    ZER_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE = 43,     ///< uint64_t: size of global memory cache in bytes
    ZER_DEVICE_INFO_GLOBAL_MEM_SIZE = 44,           ///< uint64_t: size of global memory in bytes
    ZER_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE = 45,  ///< uint64_t: max constant buffer size in bytes
    ZER_DEVICE_INFO_MAX_CONSTANT_ARGS = 46,         ///< uint32_t: max number of __const declared arguments in a kernel
    ZER_DEVICE_INFO_LOCAL_MEM_TYPE = 47,            ///< ::zer_device_local_mem_type_t: local memory type
    ZER_DEVICE_INFO_LOCAL_MEM_SIZE = 48,            ///< uint64_t: local memory size in bytes
    ZER_DEVICE_INFO_ERROR_CORRECTION_SUPPORT = 49,  ///< bool: support error correction to gloal and local memory
    ZER_DEVICE_INFO_HOST_UNIFIED_MEMORY = 50,       ///< bool: unifed host device memory
    ZER_DEVICE_INFO_PROFILING_TIMER_RESOLUTION = 51,///< size_t: profiling timer resolution in nanoseconds
    ZER_DEVICE_INFO_ENDIAN_LITTLE = 52,             ///< bool: little endian byte order
    ZER_DEVICE_INFO_AVAILABLE = 53,                 ///< bool: device is available
    ZER_DEVICE_INFO_COMPILER_AVAILABLE = 54,        ///< bool: device compiler is available
    ZER_DEVICE_INFO_LINKER_AVAILABLE = 55,          ///< bool: device linker is available
    ZER_DEVICE_INFO_EXECUTION_CAPABILITIES = 56,    ///< ::zer_device_exec_capability_flags_t: device kernel execution
                                                    ///< capability bit-field
    ZER_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES = 57,///< ::zer_queue_flags_t: device command queue property bit-field
    ZER_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES = 58,  ///< ::zer_queue_flags_t: host queue property bit-field
    ZER_DEVICE_INFO_BUILT_IN_KERNELS = 59,          ///< char[]: a semi-colon separated list of built-in kernels
    ZER_DEVICE_INFO_PLATFORM = 60,                  ///< ::zer_platform_handle_t: the platform associated with the device
    ZER_DEVICE_INFO_REFERENCE_COUNT = 61,           ///< uint32_t: reference count
    ZER_DEVICE_INFO_IL_VERSION = 62,                ///< char[]: IL version
    ZER_DEVICE_INFO_NAME = 63,                      ///< char[]: Device name
    ZER_DEVICE_INFO_VENDOR = 64,                    ///< char[]: Device vendor
    ZER_DEVICE_INFO_DRIVER_VERSION = 65,            ///< char[]: Driver version
    ZER_DEVICE_INFO_PROFILE = 66,                   ///< char[]: Device profile
    ZER_DEVICE_INFO_VERSION = 67,                   ///< char[]: Device version
    ZER_DEVICE_INFO_OPENCL_C_VERSION = 68,          ///< char[]: OpenCL C version
    ZER_DEVICE_INFO_EXTENSIONS = 69,                ///< char[]: Return a space separated list of extension names
    ZER_DEVICE_INFO_PRINTF_BUFFER_SIZE = 70,        ///< size_t: Maximum size in bytes of internal printf buffer
    ZER_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC = 71,   ///< bool: prefer user synchronization when sharing object with other API
    ZER_DEVICE_INFO_PARENT_DEVICE = 72,             ///< ::zer_device_handle_t: return parent device handle
    ZER_DEVICE_INFO_PARTITION_PROPERTIES = 73,      ///< uint32_t: return a bit-field of partition properties
                                                    ///< ::zer_device_partition_property_flags_t
    ZER_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES = 74, ///< uint32_t: maximum number of sub-devices when the device is partitioned
    ZER_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN = 75, ///< uint32_t: return a bit-field of affinity domain
                                                    ///< ::zer_device_affinity_domain_flags_t
    ZER_DEVICE_INFO_PARTITION_TYPE = 76,            ///< uint32_t: return a bit-field of
                                                    ///< ::zer_device_partition_property_flags_t for properties specified in
                                                    ///< ::zerDevicePartition
    ZER_DEVICE_INFO_MAX_NUM_SUB_GROUPS = 77,        ///< uint32_t: max number of sub groups
    ZER_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 78,///< bool: support sub group independent forward progress
    ZER_DEVICE_INFO_SUB_GROUP_SIZES_INTEL = 79,     ///< uint32_t[]: return an array of sub group sizes supported on Intel
                                                    ///< device
    ZER_DEVICE_INFO_USM_HOST_SUPPORT = 80,          ///< bool: support USM host memory access
    ZER_DEVICE_INFO_USM_DEVICE_SUPPORT = 81,        ///< bool: support USM device memory access
    ZER_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT = 82, ///< bool: support USM single device shared memory access
    ZER_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT = 83,  ///< bool: support USM cross device shared memory access
    ZER_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT = 84, ///< bool: support USM system wide shared memory access
    ZER_DEVICE_INFO_UUID = 85,                      ///< char[]: return device UUID
    ZER_DEVICE_INFO_PCI_ADDRESS = 86,               ///< char[]: return device PCI address
    ZER_DEVICE_INFO_GPU_EU_COUNT = 87,              ///< uint32_t: return Intel GPU EU count
    ZER_DEVICE_INFO_GPU_EU_SIMD_WIDTH = 88,         ///< uint32_t: return Intel GPU EU SIMD width
    ZER_DEVICE_INFO_GPU_EU_SLICES = 89,             ///< uint32_t: return Intel GPU number of slices
    ZER_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE = 90,   ///< uint32_t: return Intel GPU number of subslices per slice
    ZER_DEVICE_INFO_MAX_MEMORY_BANDWIDTH = 91,      ///< uint32_t: return max memory bandwidth in Mb/s
    ZER_DEVICE_INFO_IMAGE_SRGB = 92,                ///< bool: image is SRGB
    ZER_DEVICE_INFO_ATOMIC_64 = 93,                 ///< bool: support 64 bit atomics
    ZER_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES = 94,  ///< uint32_t: atomics memory order capabilities
    ZER_DEVICE_INFO_FORCE_UINT32 = 0x7fffffff

} zer_device_info_t;

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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES < infoType`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDeviceGetInfo(
    zer_device_handle_t hDevice,                    ///< [in] handle of the device instance
    zer_device_info_t infoType,                     ///< [in] type of the info to retrieve
    size_t* pSize,                                  ///< [in,out] pointer to the number of bytes needed to return info queried.
                                                    ///< The call shall update it with the real number of bytes needed to
                                                    ///< return the info
    void* pDeviceInfo                               ///< [out][optional] array of bytes holding the info.
                                                    ///< If *pSize input is not 0 and not equal to the real number of bytes
                                                    ///< needed to return the info
                                                    ///< then the ::ZER_RESULT_ERROR_INVALID_SIZE error is returned and
                                                    ///< pDeviceInfo is not used.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes a reference of the device handle indicating it's in use until
///        paired ::zerDeviceRelease is called
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDeviceGetReference(
    zer_device_handle_t hDevice                     ///< [in] handle of the device to get a reference of.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDeviceRelease(
    zer_device_handle_t hDevice                     ///< [in] handle of the device to release.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Device partition property
typedef uint32_t zer_device_partition_property_flags_t;
typedef enum _zer_device_partition_property_flag_t
{
    ZER_DEVICE_PARTITION_PROPERTY_FLAG_EQUALLY = ZER_BIT(0),///< Support equal partition
    ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_COUNTS = ZER_BIT(1),  ///< Support partition by count
    ZER_DEVICE_PARTITION_PROPERTY_FLAG_BY_AFFINITY_DOMAIN = ZER_BIT(2), ///< Support partition by affinity domain
    ZER_DEVICE_PARTITION_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_device_partition_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Partition property value
typedef struct _zer_device_partition_property_value_t
{
    zer_device_partition_property_flags_t property; ///< [in] device partition property flags
    uint32_t value;                                 ///< [in] partition value

} zer_device_partition_property_value_t;

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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == Properties`
///         + `nullptr == pCount`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDevicePartition(
    zer_device_handle_t hDevice,                    ///< [in] handle of the device to partition.
    zer_device_partition_property_value_t* Properties,  ///< [in] null-terminated array of <property, value> pair of the requested partitioning.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of sub-devices.
                                                    ///< If count is zero, then the function shall update the value with the
                                                    ///< total number of sub-devices available.
                                                    ///< If count is greater than the number of sub-devices available, then the
                                                    ///< function shall update the value with the correct number of sub-devices available.
    zer_device_handle_t* phSubDevices               ///< [out][optional][range(0, *pCount)] array of handle of devices.
                                                    ///< If count is less than the number of sub-devices available, then the
                                                    ///< function shall only retrieve that number of sub-devices.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Selects the most appropriate device binary based on runtime
///        information and the IR characteristics.
/// 
/// @details
///     - The input binaries are various AOT images, and possibly a SPIR-V
///       binary for JIT compilation.
///     - The selected binary will be able to be run on the target device.
///     - If no suitable binary can be found then function returns
///       ${X}_INVALID_BINARY.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == Binaries`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDeviceSelectBinary(
    zer_device_handle_t hDevice,                    ///< [in] handle of the device to select binary for.
    uint32_t NumBinaries,                           ///< [in] the number of binaries passed in Binaries. Must greater or equal
                                                    ///< than zero.
    const uint8_t** Binaries,                       ///< [in] the array of binaries to select from.
    uint32_t SelectedBinary                         ///< [out] the index of the selected binary in the input array of binaries.
                                                    ///< If a suitable binary was not found the function returns ${X}_INVALID_BINARY.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief FP capabilities
typedef uint32_t zer_fp_capability_flags_t;
typedef enum _zer_fp_capability_flag_t
{
    ZER_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT = ZER_BIT(0),  ///< Support correctly rounded divide and sqrt
    ZER_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST = ZER_BIT(1),   ///< Support round to nearest
    ZER_FP_CAPABILITY_FLAG_ROUND_TO_ZERO = ZER_BIT(2),  ///< Support round to zero
    ZER_FP_CAPABILITY_FLAG_ROUND_TO_INF = ZER_BIT(3),   ///< Support round to infinity
    ZER_FP_CAPABILITY_FLAG_INF_NAN = ZER_BIT(4),    ///< Support INF to NAN
    ZER_FP_CAPABILITY_FLAG_DENORM = ZER_BIT(5),     ///< Support denorm
    ZER_FP_CAPABILITY_FLAG_FMA = ZER_BIT(6),        ///< Support FMA
    ZER_FP_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_fp_capability_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device memory cache type
typedef enum _zer_device_mem_cache_type_t
{
    ZER_DEVICE_MEM_CACHE_TYPE_NONE = 0,             ///< Has none cache
    ZER_DEVICE_MEM_CACHE_TYPE_READ_ONLY_CACHE = 1,  ///< Has read only cache
    ZER_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE = 2, ///< Has read write cache
    ZER_DEVICE_MEM_CACHE_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_device_mem_cache_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device local memory type
typedef enum _zer_device_local_mem_type_t
{
    ZER_DEVICE_LOCAL_MEM_TYPE_LOCAL = 0,            ///< Dedicated local memory
    ZER_DEVICE_LOCAL_MEM_TYPE_GLOBAL = 1,           ///< Global memory
    ZER_DEVICE_LOCAL_MEM_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_device_local_mem_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device kernel execution capability
typedef uint32_t zer_device_exec_capability_flags_t;
typedef enum _zer_device_exec_capability_flag_t
{
    ZER_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL = ZER_BIT(0),///< Support kernel execution
    ZER_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL = ZER_BIT(1), ///< Support native kernel execution
    ZER_DEVICE_EXEC_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_device_exec_capability_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device affinity domain
typedef uint32_t zer_device_affinity_domain_flags_t;
typedef enum _zer_device_affinity_domain_flag_t
{
    ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA = ZER_BIT(0),  ///< By NUMA
    ZER_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE = ZER_BIT(1),///< BY next partitionable
    ZER_DEVICE_AFFINITY_DOMAIN_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_device_affinity_domain_flag_t;

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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeDevice`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDeviceGetNativeHandle(
    zer_device_handle_t hDevice,                    ///< [in] handle of the device.
    zer_native_handle_t* phNativeDevice             ///< [out] a pointer to the native handle of the device.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///         + `nullptr == hNativeDevice`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phDevice`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerDeviceCreateWithNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_native_handle_t hNativeDevice,              ///< [in] the native handle of the device.
    zer_device_handle_t* phDevice                   ///< [out] pointer to the handle of the device object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Program
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pKernelName`
///         + `nullptr == phKernel`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelCreate(
    zer_program_handle_t hProgram,                  ///< [in] handle of the program instance
    const char* pKernelName,                        ///< [in] pointer to null-terminated string.
    zer_kernel_handle_t* phKernel                   ///< [out] pointer to handle of kernel object created.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set kernel argument for a kernel.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clSetKernelArg**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelSetArg(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                 ///< [in] size of argument type
    const void* pArgValue                           ///< [in][optional] argument value represented as matching arg type. If
                                                    ///< null then argument value is considered null.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel object information
typedef enum _zer_kernel_info_t
{
    ZER_KERNEL_INFO_FUNCTION_NAME = 0,              ///< Return Kernel function name, return type char[]
    ZER_KERNEL_INFO_NUM_ARGS = 1,                   ///< Return Kernel number of arguments
    ZER_KERNEL_INFO_REFERENCE_COUNT = 2,            ///< Return Kernel reference count
    ZER_KERNEL_INFO_CONTEXT = 3,                    ///< Return Context object associated with Kernel
    ZER_KERNEL_INFO_PROGRAM = 4,                    ///< Return Program object associated with Kernel
    ZER_KERNEL_INFO_ATTRIBUTES = 5,                 ///< Return Kernel attributes, return type char[]
    ZER_KERNEL_INFO_FORCE_UINT32 = 0x7fffffff

} zer_kernel_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel Work Group information
typedef enum _zer_kernel_group_info_t
{
    ZER_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE = 0,     ///< Return Work Group maximum global size, return type size_t[3]
    ZER_KERNEL_GROUP_INFO_WORK_GROUP_SIZE = 1,      ///< Return maximum Work Group size, return type size_t
    ZER_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE = 2,  ///< Return Work Group size required by the source code, such as
                                                    ///< __attribute__((required_work_group_size(X,Y,Z)), return type size_t[3]
    ZER_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE = 3,       ///< Return local memory required by the Kernel, return type size_t
    ZER_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 4,   ///< Return preferred multiple of Work Group size for launch, return type
                                                    ///< size_t
    ZER_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE = 5,     ///< Return minimum amount of private memory in bytes used by each work
                                                    ///< item in the Kernel, return type size_t
    ZER_KERNEL_GROUP_INFO_FORCE_UINT32 = 0x7fffffff

} zer_kernel_group_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel SubGroup information
typedef enum _zer_kernel_sub_group_info_t
{
    ZER_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE = 0,   ///< Return maximum SubGroup size, return type uint32_t
    ZER_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS = 1,   ///< Return maximum number of SubGroup, return type uint32_t
    ZER_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS = 2,   ///< Return number of SubGroup required by the source code, return type
                                                    ///< uint32_t
    ZER_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL = 3, ///< Return SubGroup size required by Intel, return type uint32_t
    ZER_KERNEL_SUB_GROUP_INFO_FORCE_UINT32 = 0x7fffffff

} zer_kernel_sub_group_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set additional Kernel execution information
typedef enum _zer_kernel_exec_info_t
{
    ZER_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS = 0,   ///< Kernel might access data through USM pointer, type bool_t*
    ZER_KERNEL_EXEC_INFO_USM_PTRS = 1,              ///< Provide an explicit list of USM pointers that the kernel will access,
                                                    ///< type void*[].
    ZER_KERNEL_EXEC_INFO_FORCE_UINT32 = 0x7fffffff

} zer_kernel_exec_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a Kernel object
/// 
/// @remarks
///   _Analogues_
///     - **clGetKernelInfo**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_KERNEL_INFO_ATTRIBUTES < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propSize`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelGetInfo(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the Kernel object
    zer_kernel_info_t propName,                     ///< [in] name of the Kernel property to query
    size_t* propSize,                               ///< [in,out] pointer to the size of the Kernel property value
                                                    ///< If *propSize is 0 or greater than the number of bytes of the Kernel property,
                                                    ///< the call shall update the value with actual number of bytes of the
                                                    ///< Kernel property.            
    void* propValue                                 ///< [in,out][optional][range(0, *propSize)] value of the Kernel property.
                                                    ///< If *propSize is less than the number of bytes for the Kernel property,
                                                    ///< only the first *propSize bytes will be returned.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Query work Group information about a Kernel object
/// 
/// @remarks
///   _Analogues_
///     - **clGetKernelWorkGroupInfo**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propValue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelGetGroupInfo(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the Kernel object
    zer_device_handle_t hDevice,                    ///< [in] handle of the Device object
    zer_kernel_group_info_t propName,               ///< [in] name of the work Group property to query
    size_t propSize,                                ///< [in] size of the Kernel Work Group property value
    void* propValue                                 ///< [in,out][range(0, propSize)] value of the Kernel Work Group property.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Query SubGroup information about a Kernel object
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propValue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelGetSubGroupInfo(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the Kernel object
    zer_device_handle_t hDevice,                    ///< [in] handle of the Device object
    zer_kernel_sub_group_info_t propName,           ///< [in] name of the SubGroup property to query
    size_t propSize,                                ///< [in] size of the Kernel SubGroup property value
    void* propValue                                 ///< [in,out][range(0, propSize)] value of the Kernel SubGroup property.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelGetReference(
    zer_kernel_handle_t hKernel                     ///< [in] handle for the Kernel to retain
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelRelease(
    zer_kernel_handle_t hKernel                     ///< [in] handle for the Kernel to release
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a USM pointer as the argument value of a Kernel.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @remarks
///   _Analogues_
///     - **clSetKernelArgSVMPointer**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelSetArgPointer(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                 ///< [in] size of argument type
    const void* pArgValue                           ///< [in][optional] SVM pointer to memory location holding the argument
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_KERNEL_EXEC_INFO_USM_PTRS < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propValue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelSetExecInfo(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the kernel object
    zer_kernel_exec_info_t propName,                ///< [in] name of the execution attribute
    size_t propSize,                                ///< [in] size in byte the attribute value
    const void* propValue                           ///< [in][range(0, propSize)] pointer to memory location holding the
                                                    ///< property value.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Sampler object as the argument value of a Kernel.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///         + `nullptr == pArgValue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelSetArgSampler(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    zer_sampler_handle_t pArgValue                  ///< [in] handle of Sampler object.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Memory object as the argument value of a Kernel.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///         + `nullptr == pArgValue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelSetArgMemObj(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    zer_mem_handle_t pArgValue                      ///< [in] handle of Memory object.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeKernel`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelGetNativeHandle(
    zer_kernel_handle_t hKernel,                    ///< [in] handle of the kernel.
    zer_native_handle_t* phNativeKernel             ///< [out] a pointer to the native handle of the kernel.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///         + `nullptr == hNativeKernel`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phKernel`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerKernelCreateWithNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_native_handle_t hNativeKernel,              ///< [in] the native handle of the kernel.
    zer_kernel_handle_t* phKernel                   ///< [out] pointer to the handle of the kernel object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Module
#if !defined(__GNUC__)
#pragma region module
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Create Module object from IL.
/// 
/// @details
///     - Multiple calls to this function will return identical device handles,
///       in the same order.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pIL`
///         + `nullptr == pOptions`
///         + `nullptr == phModule`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerModuleCreate(
    zer_context_handle_t hContext,                  ///< [in] handle of the context instance
    const void* pIL,                                ///< [in] pointer to IL string.
    uint32_t length,                                ///< [in] length of IL in bytes.
    const char* pOptions,                           ///< [in] pointer to compiler options null-terminated string.
    void** pfnNotify,                               ///< [in][optional] A function pointer to a notification routine that is
                                                    ///< called when program compilation is complete.
    void* pUserData,                                ///< [in][optional] Passed as an argument when pfnNotify is called.
    zer_module_handle_t* phModule                   ///< [out] pointer to handle of Module object created.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the Module object.
/// 
/// @details
///     - Get a reference to the Module object handle. Increment its reference
///       count
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerModuleGetReference(
    zer_module_handle_t hModule                     ///< [in] handle for the Module to retain
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Release Module.
/// 
/// @details
///     - Decrement reference count and destroy the Module if reference count
///       becomes zero.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerModuleRelease(
    zer_module_handle_t hModule                     ///< [in] handle for the Module to release
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native module handle.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeModule`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerModuleGetNativeHandle(
    zer_module_handle_t hModule,                    ///< [in] handle of the module.
    zer_native_handle_t* phNativeModule             ///< [out] a pointer to the native handle of the module.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime module object from native module handle.
/// 
/// @details
///     - Creates runtime module handle from native driver module handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///         + `nullptr == hNativeModule`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phModule`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerModuleCreateWithNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_native_handle_t hNativeModule,              ///< [in] the native handle of the module.
    zer_module_handle_t* phModule                   ///< [out] pointer to the handle of the module object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Platform
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerPlatformGet(
    uint32_t* pCount,                               ///< [in,out] pointer to the number of platforms.
                                                    ///< if count is zero, then the call shall update the value with the total
                                                    ///< number of platforms available.
                                                    ///< if count is greater than the number of platforms available, then the
                                                    ///< call shall update the value with the correct number of platforms available.
    zer_platform_handle_t* phPlatforms              ///< [out][optional][range(0, *pCount)] array of handle of platforms.
                                                    ///< if count is less than the number of platforms available, then platform
                                                    ///< shall only retrieve that number of platforms.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported platform info
typedef enum _zer_platform_info_t
{
    ZER_PLATFORM_INFO_NAME = 1,                     ///< [char*] The string denoting name of the platform. The size of the info
                                                    ///< needs to be dynamically queried.
    ZER_PLATFORM_INFO_VENDOR_NAME = 2,              ///< [char*] The string denoting name of the vendor of the platform. The
                                                    ///< size of the info needs to be dynamically queried.
    ZER_PLATFORM_INFO_VERSION = 3,                  ///< [char*] The string denoting the version of the platform. The size of
                                                    ///< the info needs to be dynamically queried.
    ZER_PLATFORM_INFO_EXTENSIONS = 4,               ///< [char*] The string denoting extensions supported by the platform. The
                                                    ///< size of the info needs to be dynamically queried.
    ZER_PLATFORM_INFO_PROFILE = 5,                  ///< [char*] The string denoting profile of the platform. The size of the
                                                    ///< info needs to be dynamically queried.
    ZER_PLATFORM_INFO_FORCE_UINT32 = 0x7fffffff

} zer_platform_info_t;

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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_PLATFORM_INFO_PROFILE < PlatformInfoType`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerPlatformGetInfo(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform
    zer_platform_info_t PlatformInfoType,           ///< [in] type of the info to retrieve
    size_t* pSize,                                  ///< [in,out] pointer to the number of bytes needed to return info queried.
                                                    ///< the call shall update it with the real number of bytes needed to
                                                    ///< return the info
    void* pPlatformInfo                             ///< [out][optional] array of bytes holding the info.
                                                    ///< if *pSize is not equal to the real number of bytes needed to return
                                                    ///< the info then the ::ZER_RESULT_ERROR_INVALID_SIZE error is returned
                                                    ///< and pPlatformInfo is not used.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported API versions
/// 
/// @details
///     - API versions contain major and minor attributes, use
///       ::ZER_MAJOR_VERSION and ::ZER_MINOR_VERSION
typedef enum _zer_api_version_t
{
    ZER_API_VERSION_0_9 = ZER_MAKE_VERSION( 0, 9 ), ///< version 0.9
    ZER_API_VERSION_CURRENT = ZER_MAKE_VERSION( 0, 9 ), ///< latest known version
    ZER_API_VERSION_FORCE_UINT32 = 0x7fffffff

} zer_api_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the API version supported by the specified platform
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == version`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerPlatformGetApiVersion(
    zer_platform_handle_t hDriver,                  ///< [in] handle of the platform
    zer_api_version_t* version                      ///< [out] api version
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativePlatform`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerPlatformGetNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform.
    zer_native_handle_t* phNativePlatform           ///< [out] a pointer to the native handle of the platform.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPlatform`
///         + `nullptr == hNativePlatform`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phPlatform`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerPlatformCreateWithNativeHandle(
    zer_platform_handle_t hPlatform,                ///< [in] handle of the platform instance
    zer_native_handle_t hNativePlatform,            ///< [in] the native handle of the platform.
    zer_platform_handle_t* phPlatform               ///< [out] pointer to the handle of the platform object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Program
#if !defined(__GNUC__)
#pragma region program
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Create Program from input SPIR-V modules.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @remarks
///   _Analogues_
///     - **clCreateProgram**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phModules`
///         + `nullptr == phProgram`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramCreate(
    zer_context_handle_t hContext,                  ///< [in] handle of the context instance
    uint32_t count,                                 ///< [in] number of module handles in module list.
    const zer_module_handle_t** phModules,          ///< [in] pointers to array of modules.
    const char* pOptions,                           ///< [in][optional] pointer to linker options null-terminated string.
    zer_program_handle_t* phProgram                 ///< [out] pointer to handle of program object created.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create program object from native binary.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @remarks
///   _Analogues_
///     - **clCreateProgramWithBinary**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pBinary`
///         + `nullptr == phProgram`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramCreateWithBinary(
    zer_context_handle_t hContext,                  ///< [in] handle of the context instance
    zer_device_handle_t hDevice,                    ///< [in] handle to device associated with binary.
    uint32_t size,                                  ///< [in] size in bytes.
    const uint8_t* pBinary,                         ///< [in] pointer to binary.
    zer_program_handle_t* phProgram                 ///< [out] pointer to handle of Program object created.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramGetReference(
    zer_program_handle_t hProgram                   ///< [in] handle for the Program to retain
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramRelease(
    zer_program_handle_t hProgram                   ///< [in] handle for the Program to release
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves a device function pointer to a user-defined function.
/// 
/// @details
///     - Retrieves a pointer to the functions with the given name and defined
///       in the given program.
///     - ::ZER_RESULT_ERROR_INVALID_FUNCTION_NAME is returned if the function
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///         + `nullptr == hProgram`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pFunctionName`
///         + `nullptr == pFunctionPointer`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramGetFunctionPointer(
    zer_device_handle_t hDevice,                    ///< [in] handle of the device to retrieve pointer for.
    zer_program_handle_t hProgram,                  ///< [in] handle of the program to search for function in.
                                                    ///< The program must already be built to the specified device, or
                                                    ///< otherwise ::ZER_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char* pFunctionName,                      ///< [in] A null-terminates string denoting the mangled function name.
    void** pFunctionPointer                         ///< [out] Returns the pointer to the function if it is found in the program.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Program object information
typedef enum _zer_program_info_t
{
    ZER_PROGRAM_INFO_REFERENCE_COUNT = 0,           ///< Program reference count info
    ZER_PROGRAM_INFO_CONTEXT = 1,                   ///< Program context info
    ZER_PROGRAM_INFO_NUM_DEVICES = 2,               ///< Return number of devices associated with Program
    ZER_PROGRAM_INFO_DEVICES = 3,                   ///< Return list of devices associated with Program, return type
                                                    ///< uint32_t[].
    ZER_PROGRAM_INFO_SOURCE = 4,                    ///< Return program source associated with Program, return type char[].
    ZER_PROGRAM_INFO_BINARY_SIZES = 5,              ///< Return program binary sizes for each device, return type size_t[].
    ZER_PROGRAM_INFO_BINARIES = 6,                  ///< Return program binaries for all devices for this Program, return type
                                                    ///< uchar[].
    ZER_PROGRAM_INFO_NUM_KERNELS = 7,               ///< Number of kernels in Program, return type size_t
    ZER_PROGRAM_INFO_KERNEL_NAMES = 8,              ///< Return a semi-colon separated list of kernel names in Program, return
                                                    ///< type char[]
    ZER_PROGRAM_INFO_FORCE_UINT32 = 0x7fffffff

} zer_program_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a Program object
/// 
/// @remarks
///   _Analogues_
///     - **clGetProgramInfo**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_PROGRAM_INFO_KERNEL_NAMES < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propSize`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramGetInfo(
    zer_program_handle_t hProgram,                  ///< [in] handle of the Program object
    zer_program_info_t propName,                    ///< [in] name of the Program property to query
    size_t* propSize,                               ///< [in,out] pointer to the size of the Program property.
                                                    ///< If *propSize is 0 or greater than the number of bytes of the Program property,
                                                    ///< the call shall update the value with actual number of bytes of the
                                                    ///< Program property.
    void* propValue                                 ///< [in,out][optional][range(0, *propSize)] value of the Program property.
                                                    ///< If *propSize is less than the number of bytes for the Program property,
                                                    ///< only the first *propSize bytes will be returned.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Program object build status
typedef enum _zer_program_build_status_t
{
    ZER_PROGRAM_BUILD_STATUS_NONE = 0,              ///< Program build status none
    ZER_PROGRAM_BUILD_STATUS_ERROR = 1,             ///< Program build error
    ZER_PROGRAM_BUILD_STATUS_SUCCESS = 2,           ///< Program build success
    ZER_PROGRAM_BUILD_STATUS_IN_PROGRESS = 3,       ///< Program build in progress
    ZER_PROGRAM_BUILD_STATUS_FORCE_UINT32 = 0x7fffffff

} zer_program_build_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program object binary type
typedef enum _zer_program_binary_type_t
{
    ZER_PROGRAM_BINARY_TYPE_NONE = 0,               ///< No program binary is associated with device
    ZER_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 1,    ///< Program binary is compiled object
    ZER_PROGRAM_BINARY_TYPE_LIBRARY = 2,            ///< Program binary is library object
    ZER_PROGRAM_BINARY_TYPE_EXECUTABLE = 3,         ///< Program binary is executable
    ZER_PROGRAM_BINARY_TYPE_FORCE_UINT32 = 0x7fffffff

} zer_program_binary_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Program object build information
typedef enum _zer_program_build_info_t
{
    ZER_PROGRAM_BUILD_INFO_STATUS = 0,              ///< Program build status, return type ::zer_program_build_status_t
    ZER_PROGRAM_BUILD_INFO_OPTIONS = 1,             ///< Program build options, return type char[]
    ZER_PROGRAM_BUILD_INFO_LOG = 2,                 ///< Program build log, return type char[]
    ZER_PROGRAM_BUILD_INFO_BINARY_TYPE = 3,         ///< Program binary type, return type ::zer_program_binary_type_t
    ZER_PROGRAM_BUILD_INFO_FORCE_UINT32 = 0x7fffffff

} zer_program_build_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query build information about a Program object for a Device
/// 
/// @remarks
///   _Analogues_
///     - **clGetProgramBuildInfo**
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
///         + `nullptr == hDevice`
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZER_PROGRAM_BUILD_INFO_BINARY_TYPE < propName`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == propSize`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramGetBuildInfo(
    zer_program_handle_t hProgram,                  ///< [in] handle of the Program object
    zer_device_handle_t hDevice,                    ///< [in] handle of the Device object
    zer_program_build_info_t propName,              ///< [in] name of the Program build info to query
    size_t* propSize,                               ///< [in,out] pointer to the size of the Program build info property.
                                                    ///< If *propSize is 0 or greater than the number of bytes of the build
                                                    ///< info property,
                                                    ///< the call shall update the value with actual number of bytes of the
                                                    ///< build info property.
    void* propValue                                 ///< [in,out][optional][range(0, *propSize)] value of the Program build property.
                                                    ///< If *propSize is less than the number of bytes for the Program build property,
                                                    ///< only the first *propSize bytes will be returned.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Program object specialization constant to a specific value
/// 
/// @returns
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == specValue`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramSetSpecializationConstant(
    zer_program_handle_t hProgram,                  ///< [in] handle of the Program object
    uint32_t specId,                                ///< [in] specification constant Id
    size_t specSize,                                ///< [in] size of the specialization constant value
    const void* specValue                           ///< [in] pointer to the specialization value bytes
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phNativeProgram`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramGetNativeHandle(
    zer_program_handle_t hProgram,                  ///< [in] handle of the program.
    zer_native_handle_t* phNativeProgram            ///< [out] a pointer to the native handle of the program.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hProgram`
///         + `nullptr == hNativeProgram`
///     - ::ZER_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phProgram`
ZER_APIEXPORT zer_result_t ZER_APICALL
zerProgramCreateWithNativeHandle(
    zer_program_handle_t hProgram,                  ///< [in] handle of the program instance
    zer_native_handle_t hNativeProgram,             ///< [in] the native handle of the program.
    zer_program_handle_t* phProgram                 ///< [out] pointer to the handle of the program object created.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Runtime APIs for Runtime
#if !defined(__GNUC__)
#pragma region runtime
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported platform initialization flags
typedef uint32_t zer_platform_init_flags_t;
typedef enum _zer_platform_init_flag_t
{
    ZER_PLATFORM_INIT_FLAG_LEVEL_ZERO = ZER_BIT(0), ///< initialize Unified Runtime platform drivers
    ZER_PLATFORM_INIT_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_platform_init_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device initialization flags
typedef uint32_t zer_device_init_flags_t;
typedef enum _zer_device_init_flag_t
{
    ZER_DEVICE_INIT_FLAG_GPU = ZER_BIT(0),          ///< initialize GPU device drivers
    ZER_DEVICE_INIT_FLAG_FORCE_UINT32 = 0x7fffffff

} zer_device_init_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Initialize the 'oneAPI' driver(s)
/// 
/// @details
///     - The application must call this function before calling any other
///       function.
///     - If this function is not called then all other functions will return
///       ::ZER_RESULT_ERROR_UNINITIALIZED.
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
///     - ::ZER_RESULT_SUCCESS
///     - ::ZER_RESULT_ERROR_UNINITIALIZED
///     - ::ZER_RESULT_ERROR_DEVICE_LOST
///     - ::ZER_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x1 < platform_flags`
///         + `0x1 < device_flags`
///     - ::ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY
ZER_APIEXPORT zer_result_t ZER_APICALL
zerInit(
    zer_platform_init_flags_t platform_flags,       ///< [in] platform initialization flags.
                                                    ///< must be 0 (default) or a combination of ::zer_platform_init_flag_t.
    zer_device_init_flags_t device_flags            ///< [in] device initialization flags.
                                                    ///< must be 0 (default) or a combination of ::zer_device_init_flag_t.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero API Callbacks
#if !defined(__GNUC__)
#pragma region callbacks
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerPlatformGet 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_platform_get_params_t
{
    uint32_t** ppCount;
    zer_platform_handle_t** pphPlatforms;
} zer_platform_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerPlatformGet 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnPlatformGetCb_t)(
    zer_platform_get_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerPlatformGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_platform_get_info_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_platform_info_t* pPlatformInfoType;
    size_t** ppSize;
    void** ppPlatformInfo;
} zer_platform_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerPlatformGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnPlatformGetInfoCb_t)(
    zer_platform_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerPlatformGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_platform_get_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t** pphNativePlatform;
} zer_platform_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerPlatformGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnPlatformGetNativeHandleCb_t)(
    zer_platform_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerPlatformCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_platform_create_with_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t* phNativePlatform;
    zer_platform_handle_t** pphPlatform;
} zer_platform_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerPlatformCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnPlatformCreateWithNativeHandleCb_t)(
    zer_platform_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerPlatformGetApiVersion 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_platform_get_api_version_params_t
{
    zer_platform_handle_t* phDriver;
    zer_api_version_t** pversion;
} zer_platform_get_api_version_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerPlatformGetApiVersion 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnPlatformGetApiVersionCb_t)(
    zer_platform_get_api_version_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Platform callback functions pointers
typedef struct _zer_platform_callbacks_t
{
    zer_pfnPlatformGetCb_t                                          pfnGetCb;
    zer_pfnPlatformGetInfoCb_t                                      pfnGetInfoCb;
    zer_pfnPlatformGetNativeHandleCb_t                              pfnGetNativeHandleCb;
    zer_pfnPlatformCreateWithNativeHandleCb_t                       pfnCreateWithNativeHandleCb;
    zer_pfnPlatformGetApiVersionCb_t                                pfnGetApiVersionCb;
} zer_platform_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerContextCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_context_create_params_t
{
    uint32_t* pDeviceCount;
    zer_device_handle_t** pphDevices;
    zer_context_handle_t** pphContext;
} zer_context_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerContextCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnContextCreateCb_t)(
    zer_context_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerContextGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_context_get_reference_params_t
{
    zer_context_handle_t* phContext;
} zer_context_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerContextGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnContextGetReferenceCb_t)(
    zer_context_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerContextRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_context_release_params_t
{
    zer_context_handle_t* phContext;
} zer_context_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerContextRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnContextReleaseCb_t)(
    zer_context_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerContextGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_context_get_info_params_t
{
    zer_context_handle_t* phContext;
    zer_context_info_t* pContextInfoType;
    size_t** ppSize;
    void** ppContextInfo;
} zer_context_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerContextGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnContextGetInfoCb_t)(
    zer_context_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerContextGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_context_get_native_handle_params_t
{
    zer_context_handle_t* phContext;
    zer_native_handle_t** pphNativeContext;
} zer_context_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerContextGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnContextGetNativeHandleCb_t)(
    zer_context_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerContextCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_context_create_with_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t* phNativeContext;
    zer_context_handle_t** pphContext;
} zer_context_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerContextCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnContextCreateWithNativeHandleCb_t)(
    zer_context_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Context callback functions pointers
typedef struct _zer_context_callbacks_t
{
    zer_pfnContextCreateCb_t                                        pfnCreateCb;
    zer_pfnContextGetReferenceCb_t                                  pfnGetReferenceCb;
    zer_pfnContextReleaseCb_t                                       pfnReleaseCb;
    zer_pfnContextGetInfoCb_t                                       pfnGetInfoCb;
    zer_pfnContextGetNativeHandleCb_t                               pfnGetNativeHandleCb;
    zer_pfnContextCreateWithNativeHandleCb_t                        pfnCreateWithNativeHandleCb;
} zer_context_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_create_params_t
{
    zer_context_handle_t* pcontext;
    zer_event_handle_t** ppEvent;
} zer_event_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventCreateCb_t)(
    zer_event_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_get_info_params_t
{
    zer_event_handle_t* pevent;
    zer_event_info_t* ppropName;
    size_t* ppropValueSize;
    void** ppropValue;
    size_t** ppropValueSizeRet;
} zer_event_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventGetInfoCb_t)(
    zer_event_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventGetProfilingInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_get_profiling_info_params_t
{
    zer_event_handle_t* pevent;
    zer_profiling_info_t* ppropName;
    size_t* ppropValueSize;
    void** ppropValue;
    size_t* ppropValueSizeRet;
} zer_event_get_profiling_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventGetProfilingInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventGetProfilingInfoCb_t)(
    zer_event_get_profiling_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventWait 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_wait_params_t
{
    uint32_t* pnumEvents;
    const zer_event_handle_t** peventList;
} zer_event_wait_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventWait 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventWaitCb_t)(
    zer_event_wait_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_get_reference_params_t
{
    zer_event_handle_t* pevent;
} zer_event_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventGetReferenceCb_t)(
    zer_event_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_release_params_t
{
    zer_event_handle_t* pevent;
} zer_event_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventReleaseCb_t)(
    zer_event_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_get_native_handle_params_t
{
    zer_event_handle_t* phEvent;
    zer_native_handle_t** pphNativeEvent;
} zer_event_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventGetNativeHandleCb_t)(
    zer_event_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEventCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_event_create_with_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t* phNativeEvent;
    zer_event_handle_t** pphEvent;
} zer_event_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEventCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEventCreateWithNativeHandleCb_t)(
    zer_event_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Event callback functions pointers
typedef struct _zer_event_callbacks_t
{
    zer_pfnEventCreateCb_t                                          pfnCreateCb;
    zer_pfnEventGetInfoCb_t                                         pfnGetInfoCb;
    zer_pfnEventGetProfilingInfoCb_t                                pfnGetProfilingInfoCb;
    zer_pfnEventWaitCb_t                                            pfnWaitCb;
    zer_pfnEventGetReferenceCb_t                                    pfnGetReferenceCb;
    zer_pfnEventReleaseCb_t                                         pfnReleaseCb;
    zer_pfnEventGetNativeHandleCb_t                                 pfnGetNativeHandleCb;
    zer_pfnEventCreateWithNativeHandleCb_t                          pfnCreateWithNativeHandleCb;
} zer_event_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_create_params_t
{
    zer_context_handle_t* phContext;
    uint32_t* pcount;
    const zer_module_handle_t*** pphModules;
    const char** ppOptions;
    zer_program_handle_t** pphProgram;
} zer_program_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramCreateCb_t)(
    zer_program_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramCreateWithBinary 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_create_with_binary_params_t
{
    zer_context_handle_t* phContext;
    zer_device_handle_t* phDevice;
    uint32_t* psize;
    const uint8_t** ppBinary;
    zer_program_handle_t** pphProgram;
} zer_program_create_with_binary_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramCreateWithBinary 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramCreateWithBinaryCb_t)(
    zer_program_create_with_binary_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_get_reference_params_t
{
    zer_program_handle_t* phProgram;
} zer_program_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramGetReferenceCb_t)(
    zer_program_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_release_params_t
{
    zer_program_handle_t* phProgram;
} zer_program_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramReleaseCb_t)(
    zer_program_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramGetFunctionPointer 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_get_function_pointer_params_t
{
    zer_device_handle_t* phDevice;
    zer_program_handle_t* phProgram;
    const char** ppFunctionName;
    void*** ppFunctionPointer;
} zer_program_get_function_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramGetFunctionPointer 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramGetFunctionPointerCb_t)(
    zer_program_get_function_pointer_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_get_info_params_t
{
    zer_program_handle_t* phProgram;
    zer_program_info_t* ppropName;
    size_t** ppropSize;
    void** ppropValue;
} zer_program_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramGetInfoCb_t)(
    zer_program_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramGetBuildInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_get_build_info_params_t
{
    zer_program_handle_t* phProgram;
    zer_device_handle_t* phDevice;
    zer_program_build_info_t* ppropName;
    size_t** ppropSize;
    void** ppropValue;
} zer_program_get_build_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramGetBuildInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramGetBuildInfoCb_t)(
    zer_program_get_build_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramSetSpecializationConstant 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_set_specialization_constant_params_t
{
    zer_program_handle_t* phProgram;
    uint32_t* pspecId;
    size_t* pspecSize;
    const void** pspecValue;
} zer_program_set_specialization_constant_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramSetSpecializationConstant 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramSetSpecializationConstantCb_t)(
    zer_program_set_specialization_constant_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_get_native_handle_params_t
{
    zer_program_handle_t* phProgram;
    zer_native_handle_t** pphNativeProgram;
} zer_program_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramGetNativeHandleCb_t)(
    zer_program_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerProgramCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_program_create_with_native_handle_params_t
{
    zer_program_handle_t* phProgram;
    zer_native_handle_t* phNativeProgram;
    zer_program_handle_t** pphProgram;
} zer_program_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerProgramCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnProgramCreateWithNativeHandleCb_t)(
    zer_program_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Program callback functions pointers
typedef struct _zer_program_callbacks_t
{
    zer_pfnProgramCreateCb_t                                        pfnCreateCb;
    zer_pfnProgramCreateWithBinaryCb_t                              pfnCreateWithBinaryCb;
    zer_pfnProgramGetReferenceCb_t                                  pfnGetReferenceCb;
    zer_pfnProgramReleaseCb_t                                       pfnReleaseCb;
    zer_pfnProgramGetFunctionPointerCb_t                            pfnGetFunctionPointerCb;
    zer_pfnProgramGetInfoCb_t                                       pfnGetInfoCb;
    zer_pfnProgramGetBuildInfoCb_t                                  pfnGetBuildInfoCb;
    zer_pfnProgramSetSpecializationConstantCb_t                     pfnSetSpecializationConstantCb;
    zer_pfnProgramGetNativeHandleCb_t                               pfnGetNativeHandleCb;
    zer_pfnProgramCreateWithNativeHandleCb_t                        pfnCreateWithNativeHandleCb;
} zer_program_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerModuleCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_module_create_params_t
{
    zer_context_handle_t* phContext;
    const void** ppIL;
    uint32_t* plength;
    const char** ppOptions;
    void*** ppfnNotify;
    void** ppUserData;
    zer_module_handle_t** pphModule;
} zer_module_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerModuleCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnModuleCreateCb_t)(
    zer_module_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerModuleGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_module_get_reference_params_t
{
    zer_module_handle_t* phModule;
} zer_module_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerModuleGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnModuleGetReferenceCb_t)(
    zer_module_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerModuleRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_module_release_params_t
{
    zer_module_handle_t* phModule;
} zer_module_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerModuleRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnModuleReleaseCb_t)(
    zer_module_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerModuleGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_module_get_native_handle_params_t
{
    zer_module_handle_t* phModule;
    zer_native_handle_t** pphNativeModule;
} zer_module_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerModuleGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnModuleGetNativeHandleCb_t)(
    zer_module_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerModuleCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_module_create_with_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t* phNativeModule;
    zer_module_handle_t** pphModule;
} zer_module_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerModuleCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnModuleCreateWithNativeHandleCb_t)(
    zer_module_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Module callback functions pointers
typedef struct _zer_module_callbacks_t
{
    zer_pfnModuleCreateCb_t                                         pfnCreateCb;
    zer_pfnModuleGetReferenceCb_t                                   pfnGetReferenceCb;
    zer_pfnModuleReleaseCb_t                                        pfnReleaseCb;
    zer_pfnModuleGetNativeHandleCb_t                                pfnGetNativeHandleCb;
    zer_pfnModuleCreateWithNativeHandleCb_t                         pfnCreateWithNativeHandleCb;
} zer_module_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_create_params_t
{
    zer_program_handle_t* phProgram;
    const char** ppKernelName;
    zer_kernel_handle_t** pphKernel;
} zer_kernel_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelCreateCb_t)(
    zer_kernel_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_get_info_params_t
{
    zer_kernel_handle_t* phKernel;
    zer_kernel_info_t* ppropName;
    size_t** ppropSize;
    void** ppropValue;
} zer_kernel_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelGetInfoCb_t)(
    zer_kernel_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelGetGroupInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_get_group_info_params_t
{
    zer_kernel_handle_t* phKernel;
    zer_device_handle_t* phDevice;
    zer_kernel_group_info_t* ppropName;
    size_t* ppropSize;
    void** ppropValue;
} zer_kernel_get_group_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelGetGroupInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelGetGroupInfoCb_t)(
    zer_kernel_get_group_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelGetSubGroupInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_get_sub_group_info_params_t
{
    zer_kernel_handle_t* phKernel;
    zer_device_handle_t* phDevice;
    zer_kernel_sub_group_info_t* ppropName;
    size_t* ppropSize;
    void** ppropValue;
} zer_kernel_get_sub_group_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelGetSubGroupInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelGetSubGroupInfoCb_t)(
    zer_kernel_get_sub_group_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_get_reference_params_t
{
    zer_kernel_handle_t* phKernel;
} zer_kernel_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelGetReferenceCb_t)(
    zer_kernel_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_release_params_t
{
    zer_kernel_handle_t* phKernel;
} zer_kernel_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelReleaseCb_t)(
    zer_kernel_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_get_native_handle_params_t
{
    zer_kernel_handle_t* phKernel;
    zer_native_handle_t** pphNativeKernel;
} zer_kernel_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelGetNativeHandleCb_t)(
    zer_kernel_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_create_with_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t* phNativeKernel;
    zer_kernel_handle_t** pphKernel;
} zer_kernel_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelCreateWithNativeHandleCb_t)(
    zer_kernel_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelSetArg 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_set_arg_params_t
{
    zer_kernel_handle_t* phKernel;
    uint32_t* pargIndex;
    size_t* pargSize;
    const void** ppArgValue;
} zer_kernel_set_arg_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelSetArg 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelSetArgCb_t)(
    zer_kernel_set_arg_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelSetArgPointer 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_set_arg_pointer_params_t
{
    zer_kernel_handle_t* phKernel;
    uint32_t* pargIndex;
    size_t* pargSize;
    const void** ppArgValue;
} zer_kernel_set_arg_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelSetArgPointer 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelSetArgPointerCb_t)(
    zer_kernel_set_arg_pointer_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelSetExecInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_set_exec_info_params_t
{
    zer_kernel_handle_t* phKernel;
    zer_kernel_exec_info_t* ppropName;
    size_t* ppropSize;
    const void** ppropValue;
} zer_kernel_set_exec_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelSetExecInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelSetExecInfoCb_t)(
    zer_kernel_set_exec_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelSetArgSampler 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_set_arg_sampler_params_t
{
    zer_kernel_handle_t* phKernel;
    uint32_t* pargIndex;
    zer_sampler_handle_t* ppArgValue;
} zer_kernel_set_arg_sampler_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelSetArgSampler 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelSetArgSamplerCb_t)(
    zer_kernel_set_arg_sampler_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerKernelSetArgMemObj 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_kernel_set_arg_mem_obj_params_t
{
    zer_kernel_handle_t* phKernel;
    uint32_t* pargIndex;
    zer_mem_handle_t* ppArgValue;
} zer_kernel_set_arg_mem_obj_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerKernelSetArgMemObj 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnKernelSetArgMemObjCb_t)(
    zer_kernel_set_arg_mem_obj_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Kernel callback functions pointers
typedef struct _zer_kernel_callbacks_t
{
    zer_pfnKernelCreateCb_t                                         pfnCreateCb;
    zer_pfnKernelGetInfoCb_t                                        pfnGetInfoCb;
    zer_pfnKernelGetGroupInfoCb_t                                   pfnGetGroupInfoCb;
    zer_pfnKernelGetSubGroupInfoCb_t                                pfnGetSubGroupInfoCb;
    zer_pfnKernelGetReferenceCb_t                                   pfnGetReferenceCb;
    zer_pfnKernelReleaseCb_t                                        pfnReleaseCb;
    zer_pfnKernelGetNativeHandleCb_t                                pfnGetNativeHandleCb;
    zer_pfnKernelCreateWithNativeHandleCb_t                         pfnCreateWithNativeHandleCb;
    zer_pfnKernelSetArgCb_t                                         pfnSetArgCb;
    zer_pfnKernelSetArgPointerCb_t                                  pfnSetArgPointerCb;
    zer_pfnKernelSetExecInfoCb_t                                    pfnSetExecInfoCb;
    zer_pfnKernelSetArgSamplerCb_t                                  pfnSetArgSamplerCb;
    zer_pfnKernelSetArgMemObjCb_t                                   pfnSetArgMemObjCb;
} zer_kernel_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerSamplerCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_sampler_create_params_t
{
    zer_context_handle_t* phContext;
    const zer_sampler_property_value_t** pprops;
    zer_sampler_handle_t** pphSampler;
} zer_sampler_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerSamplerCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnSamplerCreateCb_t)(
    zer_sampler_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerSamplerGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_sampler_get_reference_params_t
{
    zer_sampler_handle_t* phSampler;
} zer_sampler_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerSamplerGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnSamplerGetReferenceCb_t)(
    zer_sampler_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerSamplerRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_sampler_release_params_t
{
    zer_sampler_handle_t* phSampler;
} zer_sampler_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerSamplerRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnSamplerReleaseCb_t)(
    zer_sampler_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerSamplerGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_sampler_get_info_params_t
{
    zer_sampler_handle_t* phSampler;
    zer_sampler_info_t* ppropName;
    size_t* ppropValueSize;
    void** ppropValue;
    size_t** ppSize;
} zer_sampler_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerSamplerGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnSamplerGetInfoCb_t)(
    zer_sampler_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerSamplerGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_sampler_get_native_handle_params_t
{
    zer_sampler_handle_t* phSampler;
    zer_native_handle_t** pphNativeSampler;
} zer_sampler_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerSamplerGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnSamplerGetNativeHandleCb_t)(
    zer_sampler_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerSamplerCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_sampler_create_with_native_handle_params_t
{
    zer_sampler_handle_t* phSampler;
    zer_native_handle_t* phNativeSampler;
    zer_sampler_handle_t** pphSampler;
} zer_sampler_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerSamplerCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnSamplerCreateWithNativeHandleCb_t)(
    zer_sampler_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Sampler callback functions pointers
typedef struct _zer_sampler_callbacks_t
{
    zer_pfnSamplerCreateCb_t                                        pfnCreateCb;
    zer_pfnSamplerGetReferenceCb_t                                  pfnGetReferenceCb;
    zer_pfnSamplerReleaseCb_t                                       pfnReleaseCb;
    zer_pfnSamplerGetInfoCb_t                                       pfnGetInfoCb;
    zer_pfnSamplerGetNativeHandleCb_t                               pfnGetNativeHandleCb;
    zer_pfnSamplerCreateWithNativeHandleCb_t                        pfnCreateWithNativeHandleCb;
} zer_sampler_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemImageCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_image_create_params_t
{
    zer_context_handle_t* phContext;
    zer_mem_flags_t* pflags;
    const zer_image_format_t** pimageFormat;
    const zer_image_desc_t** pimageDesc;
    void** phostPtr;
    zer_mem_handle_t** pphMem;
} zer_mem_image_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemImageCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemImageCreateCb_t)(
    zer_mem_image_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemBufferCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_buffer_create_params_t
{
    zer_context_handle_t* phContext;
    zer_mem_flags_t* pflags;
    size_t* psize;
    void** phostPtr;
    zer_mem_handle_t** pphBuffer;
} zer_mem_buffer_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemBufferCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemBufferCreateCb_t)(
    zer_mem_buffer_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_get_reference_params_t
{
    zer_mem_handle_t* phMem;
} zer_mem_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemGetReferenceCb_t)(
    zer_mem_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_release_params_t
{
    zer_mem_handle_t* phMem;
} zer_mem_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemReleaseCb_t)(
    zer_mem_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemBufferPartition 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_buffer_partition_params_t
{
    zer_mem_handle_t* phBuffer;
    zer_mem_flags_t* pflags;
    zer_buffer_create_type_t* pbufferCreateType;
    zer_buffer_region_t** ppBufferCreateInfo;
    zer_mem_handle_t** pphMem;
} zer_mem_buffer_partition_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemBufferPartition 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemBufferPartitionCb_t)(
    zer_mem_buffer_partition_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_get_native_handle_params_t
{
    zer_mem_handle_t* phMem;
    zer_native_handle_t** pphNativeMem;
} zer_mem_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemGetNativeHandleCb_t)(
    zer_mem_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_create_with_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t* phNativeMem;
    zer_mem_handle_t** pphMem;
} zer_mem_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemCreateWithNativeHandleCb_t)(
    zer_mem_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemFree 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_free_params_t
{
    zer_context_handle_t* pcontext;
    void** pptr;
} zer_mem_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemFree 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemFreeCb_t)(
    zer_mem_free_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerMemGetMemAllocInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_mem_get_mem_alloc_info_params_t
{
    zer_context_handle_t* pcontext;
    const void** pptr;
    zer_mem_info_t* ppropName;
    size_t* ppropValueSize;
    void** ppropValue;
    size_t** ppropValueSizeRet;
} zer_mem_get_mem_alloc_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerMemGetMemAllocInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnMemGetMemAllocInfoCb_t)(
    zer_mem_get_mem_alloc_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Mem callback functions pointers
typedef struct _zer_mem_callbacks_t
{
    zer_pfnMemImageCreateCb_t                                       pfnImageCreateCb;
    zer_pfnMemBufferCreateCb_t                                      pfnBufferCreateCb;
    zer_pfnMemGetReferenceCb_t                                      pfnGetReferenceCb;
    zer_pfnMemReleaseCb_t                                           pfnReleaseCb;
    zer_pfnMemBufferPartitionCb_t                                   pfnBufferPartitionCb;
    zer_pfnMemGetNativeHandleCb_t                                   pfnGetNativeHandleCb;
    zer_pfnMemCreateWithNativeHandleCb_t                            pfnCreateWithNativeHandleCb;
    zer_pfnMemFreeCb_t                                              pfnFreeCb;
    zer_pfnMemGetMemAllocInfoCb_t                                   pfnGetMemAllocInfoCb;
} zer_mem_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueKernelLaunch 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_kernel_launch_params_t
{
    zer_queue_handle_t* phQueue;
    zer_kernel_handle_t* phKernel;
    uint32_t* pworkDim;
    const size_t** pglobalWorkOffset;
    const size_t** pglobalWorkSize;
    const size_t** plocalWorkSize;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_kernel_launch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueKernelLaunch 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueKernelLaunchCb_t)(
    zer_enqueue_kernel_launch_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueEventsWait 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_events_wait_params_t
{
    zer_queue_handle_t* phQueue;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_events_wait_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueEventsWait 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueEventsWaitCb_t)(
    zer_enqueue_events_wait_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueEventsWaitWithBarrier 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_events_wait_with_barrier_params_t
{
    zer_queue_handle_t* phQueue;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_events_wait_with_barrier_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueEventsWaitWithBarrier 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueEventsWaitWithBarrierCb_t)(
    zer_enqueue_events_wait_with_barrier_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferRead 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_read_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phBuffer;
    bool* pblockingRead;
    size_t* poffset;
    size_t* psize;
    void** pdst;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_buffer_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferRead 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferReadCb_t)(
    zer_enqueue_mem_buffer_read_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferWrite 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_write_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phBuffer;
    bool* pblockingWrite;
    size_t* poffset;
    size_t* psize;
    const void** psrc;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_buffer_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferWrite 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferWriteCb_t)(
    zer_enqueue_mem_buffer_write_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferReadRect 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_read_rect_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phBuffer;
    bool* pblockingRead;
    zer_rect_offset_t* pbufferOffset;
    zer_rect_offset_t* phostOffset;
    zer_rect_region_t* pregion;
    size_t* pbufferRowPitch;
    size_t* pbufferSlicePitch;
    size_t* phostRowPitch;
    size_t* phostSlicePitch;
    void** pdst;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_buffer_read_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferReadRect 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferReadRectCb_t)(
    zer_enqueue_mem_buffer_read_rect_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferWriteRect 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_write_rect_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phBuffer;
    bool* pblockingWrite;
    zer_rect_offset_t* pbufferOffset;
    zer_rect_offset_t* phostOffset;
    zer_rect_region_t* pregion;
    size_t* pbufferRowPitch;
    size_t* pbufferSlicePitch;
    size_t* phostRowPitch;
    size_t* phostSlicePitch;
    void** psrc;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_buffer_write_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferWriteRect 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferWriteRectCb_t)(
    zer_enqueue_mem_buffer_write_rect_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferCopy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_copy_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phBufferSrc;
    zer_mem_handle_t* phBufferDst;
    size_t* psize;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_buffer_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferCopy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferCopyCb_t)(
    zer_enqueue_mem_buffer_copy_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferCopyRect 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_copy_rect_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phBufferSrc;
    zer_mem_handle_t* phBufferDst;
    zer_rect_offset_t* psrcOrigin;
    zer_rect_offset_t* pdstOrigin;
    zer_rect_region_t* psrcRegion;
    size_t* psrcRowPitch;
    size_t* psrcSlicePitch;
    size_t* pdstRowPitch;
    size_t* pdstSlicePitch;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_buffer_copy_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferCopyRect 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferCopyRectCb_t)(
    zer_enqueue_mem_buffer_copy_rect_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferFill 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_fill_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phBuffer;
    const void** ppattern;
    size_t* ppatternSize;
    size_t* poffset;
    size_t* psize;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_buffer_fill_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferFill 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferFillCb_t)(
    zer_enqueue_mem_buffer_fill_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemImageRead 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_image_read_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phImage;
    bool* pblockingRead;
    zer_rect_offset_t* porigin;
    zer_rect_region_t* pregion;
    size_t* prowPitch;
    size_t* pslicePitch;
    void** pdst;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_image_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemImageRead 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemImageReadCb_t)(
    zer_enqueue_mem_image_read_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemImageWrite 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_image_write_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phImage;
    bool* pblockingWrite;
    zer_rect_offset_t* porigin;
    zer_rect_region_t* pregion;
    size_t* pinputRowPitch;
    size_t* pinputSlicePitch;
    void** psrc;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_image_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemImageWrite 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemImageWriteCb_t)(
    zer_enqueue_mem_image_write_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemImageCopy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_image_copy_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phImageSrc;
    zer_mem_handle_t* phImageDst;
    zer_rect_offset_t* psrcOrigin;
    zer_rect_offset_t* pdstOrigin;
    zer_rect_region_t* pregion;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_image_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemImageCopy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemImageCopyCb_t)(
    zer_enqueue_mem_image_copy_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemBufferMap 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_buffer_map_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* pbuffer;
    bool* pblockingMap;
    zer_map_flags_t* pmapFlags;
    size_t* poffset;
    size_t* psize;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
    void*** pretMap;
} zer_enqueue_mem_buffer_map_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemBufferMap 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemBufferMapCb_t)(
    zer_enqueue_mem_buffer_map_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueMemUnmap 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_mem_unmap_params_t
{
    zer_queue_handle_t* phQueue;
    zer_mem_handle_t* phMem;
    void** pmappedPtr;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_mem_unmap_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueMemUnmap 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueMemUnmapCb_t)(
    zer_enqueue_mem_unmap_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueUSMMemset 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_usm_memset_params_t
{
    zer_queue_handle_t* phQueue;
    void** pptr;
    int8_t* pbyteValue;
    size_t* pcount;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_usm_memset_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueUSMMemset 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueUSMMemsetCb_t)(
    zer_enqueue_usm_memset_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueUSMMemcpy 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_usm_memcpy_params_t
{
    zer_queue_handle_t* phQueue;
    bool* pblocking;
    void** pdstPrt;
    const void** psrcPrt;
    size_t* psize;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_usm_memcpy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueUSMMemcpy 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueUSMMemcpyCb_t)(
    zer_enqueue_usm_memcpy_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueUSMPrefetch 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_usm_prefetch_params_t
{
    zer_queue_handle_t* phQueue;
    const void** pptr;
    size_t* psize;
    zer_usm_migration_flags_t* pflags;
    uint32_t* pnumEventsInWaitList;
    const zer_event_handle_t** peventWaitList;
    zer_event_handle_t** pevent;
} zer_enqueue_usm_prefetch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueUSMPrefetch 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueUSMPrefetchCb_t)(
    zer_enqueue_usm_prefetch_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerEnqueueUSMMemAdvice 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_enqueue_usm_mem_advice_params_t
{
    zer_queue_handle_t* phQueue;
    const void** pptr;
    size_t* psize;
    zer_mem_advice_t* padvice;
    zer_event_handle_t** pevent;
} zer_enqueue_usm_mem_advice_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerEnqueueUSMMemAdvice 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnEnqueueUSMMemAdviceCb_t)(
    zer_enqueue_usm_mem_advice_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Enqueue callback functions pointers
typedef struct _zer_enqueue_callbacks_t
{
    zer_pfnEnqueueKernelLaunchCb_t                                  pfnKernelLaunchCb;
    zer_pfnEnqueueEventsWaitCb_t                                    pfnEventsWaitCb;
    zer_pfnEnqueueEventsWaitWithBarrierCb_t                         pfnEventsWaitWithBarrierCb;
    zer_pfnEnqueueMemBufferReadCb_t                                 pfnMemBufferReadCb;
    zer_pfnEnqueueMemBufferWriteCb_t                                pfnMemBufferWriteCb;
    zer_pfnEnqueueMemBufferReadRectCb_t                             pfnMemBufferReadRectCb;
    zer_pfnEnqueueMemBufferWriteRectCb_t                            pfnMemBufferWriteRectCb;
    zer_pfnEnqueueMemBufferCopyCb_t                                 pfnMemBufferCopyCb;
    zer_pfnEnqueueMemBufferCopyRectCb_t                             pfnMemBufferCopyRectCb;
    zer_pfnEnqueueMemBufferFillCb_t                                 pfnMemBufferFillCb;
    zer_pfnEnqueueMemImageReadCb_t                                  pfnMemImageReadCb;
    zer_pfnEnqueueMemImageWriteCb_t                                 pfnMemImageWriteCb;
    zer_pfnEnqueueMemImageCopyCb_t                                  pfnMemImageCopyCb;
    zer_pfnEnqueueMemBufferMapCb_t                                  pfnMemBufferMapCb;
    zer_pfnEnqueueMemUnmapCb_t                                      pfnMemUnmapCb;
    zer_pfnEnqueueUSMMemsetCb_t                                     pfnUSMMemsetCb;
    zer_pfnEnqueueUSMMemcpyCb_t                                     pfnUSMMemcpyCb;
    zer_pfnEnqueueUSMPrefetchCb_t                                   pfnUSMPrefetchCb;
    zer_pfnEnqueueUSMMemAdviceCb_t                                  pfnUSMMemAdviceCb;
} zer_enqueue_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerUSMHostAlloc 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_usm_host_alloc_params_t
{
    zer_context_handle_t* pcontext;
    zer_usm_mem_flags_t** ppUSMFlag;
    size_t* psize;
    uint32_t* palign;
    void*** ppptr;
} zer_usm_host_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerUSMHostAlloc 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnUSMHostAllocCb_t)(
    zer_usm_host_alloc_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerUSMDeviceAlloc 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_usm_device_alloc_params_t
{
    zer_context_handle_t* pcontext;
    zer_device_handle_t* pdevice;
    zer_usm_mem_flags_t** ppUSMProp;
    size_t* psize;
    uint32_t* palign;
    void*** ppptr;
} zer_usm_device_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerUSMDeviceAlloc 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnUSMDeviceAllocCb_t)(
    zer_usm_device_alloc_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerUSMSharedAlloc 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_usm_shared_alloc_params_t
{
    zer_context_handle_t* pcontext;
    zer_device_handle_t* pdevice;
    zer_usm_mem_flags_t** ppUSMProp;
    size_t* psize;
    uint32_t* palign;
    void*** ppptr;
} zer_usm_shared_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerUSMSharedAlloc 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnUSMSharedAllocCb_t)(
    zer_usm_shared_alloc_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of USM callback functions pointers
typedef struct _zer_usm_callbacks_t
{
    zer_pfnUSMHostAllocCb_t                                         pfnHostAllocCb;
    zer_pfnUSMDeviceAllocCb_t                                       pfnDeviceAllocCb;
    zer_pfnUSMSharedAllocCb_t                                       pfnSharedAllocCb;
} zer_usm_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerTearDown 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_tear_down_params_t
{
    void** ppParams;
} zer_tear_down_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerTearDown 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnTearDownCb_t)(
    zer_tear_down_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerInit 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_init_params_t
{
    zer_platform_init_flags_t* pplatform_flags;
    zer_device_init_flags_t* pdevice_flags;
} zer_init_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerInit 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnInitCb_t)(
    zer_init_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Global callback functions pointers
typedef struct _zer_global_callbacks_t
{
    zer_pfnTearDownCb_t                                             pfnTearDownCb;
    zer_pfnInitCb_t                                                 pfnInitCb;
} zer_global_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerQueueGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_queue_get_info_params_t
{
    zer_queue_handle_t* phQueue;
    zer_queue_info_t* ppropName;
    size_t* ppropValueSize;
    void** ppropValue;
    size_t** ppSize;
} zer_queue_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerQueueGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnQueueGetInfoCb_t)(
    zer_queue_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerQueueCreate 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_queue_create_params_t
{
    zer_context_handle_t* phContext;
    zer_device_handle_t* phDevice;
    zer_queue_flags_t* pprops;
    zer_queue_handle_t** pphQueue;
} zer_queue_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerQueueCreate 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnQueueCreateCb_t)(
    zer_queue_create_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerQueueGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_queue_get_reference_params_t
{
    zer_queue_handle_t* phQueue;
} zer_queue_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerQueueGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnQueueGetReferenceCb_t)(
    zer_queue_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerQueueRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_queue_release_params_t
{
    zer_queue_handle_t* phQueue;
} zer_queue_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerQueueRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnQueueReleaseCb_t)(
    zer_queue_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerQueueGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_queue_get_native_handle_params_t
{
    zer_queue_handle_t* phQueue;
    zer_native_handle_t** pphNativeQueue;
} zer_queue_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerQueueGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnQueueGetNativeHandleCb_t)(
    zer_queue_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerQueueCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_queue_create_with_native_handle_params_t
{
    zer_queue_handle_t* phQueue;
    zer_native_handle_t* phNativeQueue;
    zer_queue_handle_t** pphQueue;
} zer_queue_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerQueueCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnQueueCreateWithNativeHandleCb_t)(
    zer_queue_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Queue callback functions pointers
typedef struct _zer_queue_callbacks_t
{
    zer_pfnQueueGetInfoCb_t                                         pfnGetInfoCb;
    zer_pfnQueueCreateCb_t                                          pfnCreateCb;
    zer_pfnQueueGetReferenceCb_t                                    pfnGetReferenceCb;
    zer_pfnQueueReleaseCb_t                                         pfnReleaseCb;
    zer_pfnQueueGetNativeHandleCb_t                                 pfnGetNativeHandleCb;
    zer_pfnQueueCreateWithNativeHandleCb_t                          pfnCreateWithNativeHandleCb;
} zer_queue_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDeviceGet 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_get_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_device_type_t* pDevicesType;
    uint32_t** ppCount;
    zer_device_handle_t** pphDevices;
} zer_device_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDeviceGet 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDeviceGetCb_t)(
    zer_device_get_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDeviceGetInfo 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_get_info_params_t
{
    zer_device_handle_t* phDevice;
    zer_device_info_t* pinfoType;
    size_t** ppSize;
    void** ppDeviceInfo;
} zer_device_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDeviceGetInfo 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDeviceGetInfoCb_t)(
    zer_device_get_info_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDeviceGetReference 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_get_reference_params_t
{
    zer_device_handle_t* phDevice;
} zer_device_get_reference_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDeviceGetReference 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDeviceGetReferenceCb_t)(
    zer_device_get_reference_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDeviceRelease 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_release_params_t
{
    zer_device_handle_t* phDevice;
} zer_device_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDeviceRelease 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDeviceReleaseCb_t)(
    zer_device_release_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDevicePartition 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_partition_params_t
{
    zer_device_handle_t* phDevice;
    zer_device_partition_property_value_t** pProperties;
    uint32_t** ppCount;
    zer_device_handle_t** pphSubDevices;
} zer_device_partition_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDevicePartition 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDevicePartitionCb_t)(
    zer_device_partition_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDeviceSelectBinary 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_select_binary_params_t
{
    zer_device_handle_t* phDevice;
    uint32_t* pNumBinaries;
    const uint8_t*** pBinaries;
    uint32_t* pSelectedBinary;
} zer_device_select_binary_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDeviceSelectBinary 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDeviceSelectBinaryCb_t)(
    zer_device_select_binary_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDeviceGetNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_get_native_handle_params_t
{
    zer_device_handle_t* phDevice;
    zer_native_handle_t** pphNativeDevice;
} zer_device_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDeviceGetNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDeviceGetNativeHandleCb_t)(
    zer_device_get_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function parameters for zerDeviceCreateWithNativeHandle 
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct _zer_device_create_with_native_handle_params_t
{
    zer_platform_handle_t* phPlatform;
    zer_native_handle_t* phNativeDevice;
    zer_device_handle_t** pphDevice;
} zer_device_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Callback function-pointer for zerDeviceCreateWithNativeHandle 
/// @param[in] params Parameters passed to this instance
/// @param[in] result Return value
/// @param[in] pTracerUserData Per-Tracer user data
/// @param[in,out] ppTracerInstanceUserData Per-Tracer, Per-Instance user data
typedef void (ZER_APICALL *zer_pfnDeviceCreateWithNativeHandleCb_t)(
    zer_device_create_with_native_handle_params_t* params,
    zer_result_t result,
    void* pTracerUserData,
    void** ppTracerInstanceUserData
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Device callback functions pointers
typedef struct _zer_device_callbacks_t
{
    zer_pfnDeviceGetCb_t                                            pfnGetCb;
    zer_pfnDeviceGetInfoCb_t                                        pfnGetInfoCb;
    zer_pfnDeviceGetReferenceCb_t                                   pfnGetReferenceCb;
    zer_pfnDeviceReleaseCb_t                                        pfnReleaseCb;
    zer_pfnDevicePartitionCb_t                                      pfnPartitionCb;
    zer_pfnDeviceSelectBinaryCb_t                                   pfnSelectBinaryCb;
    zer_pfnDeviceGetNativeHandleCb_t                                pfnGetNativeHandleCb;
    zer_pfnDeviceCreateWithNativeHandleCb_t                         pfnCreateWithNativeHandleCb;
} zer_device_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Container for all callbacks
typedef struct _zer_callbacks_t
{
    zer_platform_callbacks_t            Platform;
    zer_context_callbacks_t             Context;
    zer_event_callbacks_t               Event;
    zer_program_callbacks_t             Program;
    zer_module_callbacks_t              Module;
    zer_kernel_callbacks_t              Kernel;
    zer_sampler_callbacks_t             Sampler;
    zer_mem_callbacks_t                 Mem;
    zer_enqueue_callbacks_t             Enqueue;
    zer_usm_callbacks_t                 USM;
    zer_global_callbacks_t              Global;
    zer_queue_callbacks_t               Queue;
    zer_device_callbacks_t              Device;
} zer_callbacks_t;

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZER_API_H