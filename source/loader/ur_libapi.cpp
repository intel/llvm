/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_libapi.cpp
 *
 * @brief C++ library for ur
 *
 */
#include "ur_lib.h"

extern "C" {

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
ur_result_t UR_APICALL
urContextCreate(
    uint32_t DeviceCount,                           ///< [in] the number of devices given in phDevices
    ur_device_handle_t* phDevices,                  ///< [in][range(0, DeviceCount)] array of handle of devices.
    ur_context_handle_t* phContext                  ///< [out] pointer to handle of context object created
    )
{
    auto pfnCreate = ur_lib::context->urDdiTable.Context.pfnCreate;
    if( nullptr == pfnCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreate( DeviceCount, phDevices, phContext );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
ur_result_t UR_APICALL
urContextRetain(
    ur_context_handle_t hContext                    ///< [in] handle of the context to get a reference of.
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Context.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hContext );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
ur_result_t UR_APICALL
urContextRelease(
    ur_context_handle_t hContext                    ///< [in] handle of the context to release.
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Context.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hContext );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_CONTEXT_INFO_USM_MEMSET2D_SUPPORT < ContextInfoType`
ur_result_t UR_APICALL
urContextGetInfo(
    ur_context_handle_t hContext,                   ///< [in] handle of the context
    ur_context_info_t ContextInfoType,              ///< [in] type of the info to retrieve
    size_t propSize,                                ///< [in] the number of bytes of memory pointed to by pContextInfo.
    void* pContextInfo,                             ///< [out][optional] array of bytes holding the info.
                                                    ///< if propSize is not equal to or greater than the real number of bytes
                                                    ///< needed to return 
                                                    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                                    ///< pContextInfo is not used.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data queried by ContextInfoType.
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Context.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hContext, ContextInfoType, propSize, pContextInfo, pPropSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeContext`
ur_result_t UR_APICALL
urContextGetNativeHandle(
    ur_context_handle_t hContext,                   ///< [in] handle of the context.
    ur_native_handle_t* phNativeContext             ///< [out] a pointer to the native handle of the context.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Context.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hContext, phNativeContext );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phContext`
ur_result_t UR_APICALL
urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext,              ///< [in] the native handle of the context.
    ur_context_handle_t* phContext                  ///< [out] pointer to the handle of the context object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Context.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeContext, phContext );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pUserData`
ur_result_t UR_APICALL
urContextSetExtendedDeleter(
    ur_context_handle_t hContext,                   ///< [in] handle of the context.
    ur_context_extended_deleter_t pfnDeleter,       ///< [in] Function pointer to extended deleter.
    void* pUserData                                 ///< [in][out] pointer to data to be passed to callback.
    )
{
    auto pfnSetExtendedDeleter = ur_lib::context->urDdiTable.Context.pfnSetExtendedDeleter;
    if( nullptr == pfnSetExtendedDeleter )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetExtendedDeleter( hContext, pfnDeleter, pUserData );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGlobalWorkOffset`
///         + `NULL == pGlobalWorkSize`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    uint32_t workDim,                               ///< [in] number of dimensions, from 1 to 3, to specify the global and
                                                    ///< work-group work-items
    const size_t* pGlobalWorkOffset,                ///< [in] pointer to an array of workDim unsigned values that specify the
                                                    ///< offset used to calculate the global ID of a work-item
    const size_t* pGlobalWorkSize,                  ///< [in] pointer to an array of workDim unsigned values that specify the
                                                    ///< number of global work-items in workDim that will execute the kernel
                                                    ///< function
    const size_t* pLocalWorkSize,                   ///< [in][optional] pointer to an array of workDim unsigned values that
                                                    ///< specify the number of local work-items forming a work-group that will
                                                    ///< execute the kernel function.
                                                    ///< If nullptr, the runtime implementation will choose the work-group
                                                    ///< size. 
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before the kernel execution.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                                    ///< event. 
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular kernel execution instance.
    )
{
    auto pfnKernelLaunch = ur_lib::context->urDdiTable.Enqueue.pfnKernelLaunch;
    if( nullptr == pfnKernelLaunch )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnKernelLaunch( hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueEventsWait(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                                    ///< previously enqueued commands
                                                    ///< must be complete. 
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnEventsWait = ur_lib::context->urDdiTable.Enqueue.pfnEventsWait;
    if( nullptr == pfnEventsWait )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnEventsWait( hQueue, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                                    ///< previously enqueued commands
                                                    ///< must be complete. 
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnEventsWaitWithBarrier = ur_lib::context->urDdiTable.Enqueue.pfnEventsWaitWithBarrier;
    if( nullptr == pfnEventsWaitWithBarrier )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnEventsWaitWithBarrier( hQueue, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object
    bool blockingRead,                              ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                                  ///< [in] offset in bytes in the buffer object
    size_t size,                                    ///< [in] size in bytes of data being read
    void* pDst,                                     ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnMemBufferRead = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferRead;
    if( nullptr == pfnMemBufferRead )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferRead( hQueue, hBuffer, blockingRead, offset, size, pDst, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object
    bool blockingWrite,                             ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                                  ///< [in] offset in bytes in the buffer object
    size_t size,                                    ///< [in] size in bytes of data being written
    const void* pSrc,                               ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnMemBufferWrite = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferWrite;
    if( nullptr == pfnMemBufferWrite )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferWrite( hQueue, hBuffer, blockingWrite, offset, size, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object
    bool blockingRead,                              ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOffset,                  ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOffset,                    ///< [in] 3D offset in the host region
    ur_rect_region_t region,                        ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                          ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                        ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t hostRowPitch,                            ///< [in] length of each row in bytes in the host memory region pointed by
                                                    ///< dst
    size_t hostSlicePitch,                          ///< [in] length of each 2D slice in bytes in the host memory region
                                                    ///< pointed by dst
    void* pDst,                                     ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnMemBufferReadRect = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferReadRect;
    if( nullptr == pfnMemBufferReadRect )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferReadRect( hQueue, hBuffer, blockingRead, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object
    bool blockingWrite,                             ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOffset,                  ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOffset,                    ///< [in] 3D offset in the host region
    ur_rect_region_t region,                        ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                          ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                        ///< [in] length of each 2D slice in bytes in the buffer object being
                                                    ///< written
    size_t hostRowPitch,                            ///< [in] length of each row in bytes in the host memory region pointed by
                                                    ///< src
    size_t hostSlicePitch,                          ///< [in] length of each 2D slice in bytes in the host memory region
                                                    ///< pointed by src
    void* pSrc,                                     ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance. 
    )
{
    auto pfnMemBufferWriteRect = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferWriteRect;
    if( nullptr == pfnMemBufferWriteRect )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferWriteRect( hQueue, hBuffer, blockingWrite, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBufferSrc`
///         + `NULL == hBufferDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,                     ///< [in] handle of the src buffer object
    ur_mem_handle_t hBufferDst,                     ///< [in] handle of the dest buffer object
    size_t srcOffset,                               ///< [in] offset into hBufferSrc to begin copying from
    size_t dstOffset,                               ///< [in] offset info hBufferDst to begin copying into
    size_t size,                                    ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance. 
    )
{
    auto pfnMemBufferCopy = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferCopy;
    if( nullptr == pfnMemBufferCopy )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferCopy( hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset, size, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBufferSrc`
///         + `NULL == hBufferDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,                     ///< [in] handle of the source buffer object
    ur_mem_handle_t hBufferDst,                     ///< [in] handle of the dest buffer object
    ur_rect_offset_t srcOrigin,                     ///< [in] 3D offset in the source buffer
    ur_rect_offset_t dstOrigin,                     ///< [in] 3D offset in the destination buffer
    ur_rect_region_t srcRegion,                     ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t srcRowPitch,                             ///< [in] length of each row in bytes in the source buffer object
    size_t srcSlicePitch,                           ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t dstRowPitch,                             ///< [in] length of each row in bytes in the destination buffer object
    size_t dstSlicePitch,                           ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnMemBufferCopyRect = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferCopyRect;
    if( nullptr == pfnMemBufferCopyRect )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferCopyRect( hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, srcRegion, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object
    const void* pPattern,                           ///< [in] pointer to the fill pattern
    size_t patternSize,                             ///< [in] size in bytes of the pattern
    size_t offset,                                  ///< [in] offset into the buffer
    size_t size,                                    ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnMemBufferFill = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferFill;
    if( nullptr == pfnMemBufferFill )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferFill( hQueue, hBuffer, pPattern, patternSize, offset, size, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemImageRead(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                         ///< [in] handle of the image object
    bool blockingRead,                              ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                        ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                        ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                                    ///< image
    size_t rowPitch,                                ///< [in] length of each row in bytes
    size_t slicePitch,                              ///< [in] length of each 2D slice of the 3D image
    void* pDst,                                     ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance. 
    )
{
    auto pfnMemImageRead = ur_lib::context->urDdiTable.Enqueue.pfnMemImageRead;
    if( nullptr == pfnMemImageRead )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemImageRead( hQueue, hImage, blockingRead, origin, region, rowPitch, slicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                         ///< [in] handle of the image object
    bool blockingWrite,                             ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                        ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                        ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                                    ///< image
    size_t inputRowPitch,                           ///< [in] length of each row in bytes
    size_t inputSlicePitch,                         ///< [in] length of each 2D slice of the 3D image
    void* pSrc,                                     ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnMemImageWrite = ur_lib::context->urDdiTable.Enqueue.pfnMemImageWrite;
    if( nullptr == pfnMemImageWrite )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemImageWrite( hQueue, hImage, blockingWrite, origin, region, inputRowPitch, inputSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImageSrc`
///         + `NULL == hImageDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hImageSrc,                      ///< [in] handle of the src image object
    ur_mem_handle_t hImageDst,                      ///< [in] handle of the dest image object
    ur_rect_offset_t srcOrigin,                     ///< [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
                                                    ///< image
    ur_rect_offset_t dstOrigin,                     ///< [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
                                                    ///< or 3D image
    ur_rect_region_t region,                        ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                                    ///< image
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance. 
    )
{
    auto pfnMemImageCopy = ur_lib::context->urDdiTable.Enqueue.pfnMemImageCopy;
    if( nullptr == pfnMemImageCopy )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemImageCopy( hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin, region, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3 < mapFlags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppRetMap`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object
    bool blockingMap,                               ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t mapFlags,                        ///< [in] flags for read, write, readwrite mapping
    size_t offset,                                  ///< [in] offset in bytes of the buffer region being mapped
    size_t size,                                    ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent,                     ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    void** ppRetMap                                 ///< [in,out] return mapped pointer.  TODO: move it before
                                                    ///< numEventsInWaitList?
    )
{
    auto pfnMemBufferMap = ur_lib::context->urDdiTable.Enqueue.pfnMemBufferMap;
    if( nullptr == pfnMemBufferMap )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemBufferMap( hQueue, hBuffer, blockingMap, mapFlags, offset, size, numEventsInWaitList, phEventWaitList, phEvent, ppRetMap );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMappedPtr`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueMemUnmap(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_mem_handle_t hMem,                           ///< [in] handle of the memory (buffer or image) object
    void* pMappedPtr,                               ///< [in] mapped host address
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnMemUnmap = ur_lib::context->urDdiTable.Enqueue.pfnMemUnmap;
    if( nullptr == pfnMemUnmap )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnMemUnmap( hQueue, hMem, pMappedPtr, numEventsInWaitList, phEventWaitList, phEvent );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to set USM memory object value
/// 
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ptr`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueUSMMemset(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    void* ptr,                                      ///< [in] pointer to USM memory object
    int8_t byteValue,                               ///< [in] byte value to fill
    size_t count,                                   ///< [in] size in bytes to be set
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance. 
    )
{
    auto pfnUSMMemset = ur_lib::context->urDdiTable.Enqueue.pfnUSMMemset;
    if( nullptr == pfnUSMMemset )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnUSMMemset( hQueue, ptr, byteValue, count, numEventsInWaitList, phEventWaitList, phEvent );
}

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
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    bool blocking,                                  ///< [in] blocking or non-blocking copy
    void* pDst,                                     ///< [in] pointer to the destination USM memory object
    const void* pSrc,                               ///< [in] pointer to the source USM memory object
    size_t size,                                    ///< [in] size in bytes to be copied
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnUSMMemcpy = ur_lib::context->urDdiTable.Enqueue.pfnUSMMemcpy;
    if( nullptr == pfnUSMMemcpy )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnUSMMemcpy( hQueue, blocking, pDst, pSrc, size, numEventsInWaitList, phEventWaitList, phEvent );
}

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
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    const void* pMem,                               ///< [in] pointer to the USM memory object
    size_t size,                                    ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t flags,                 ///< [in] USM prefetch flags
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before this command can be executed.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                                    ///< command does not wait on any event to complete.
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnUSMPrefetch = ur_lib::context->urDdiTable.Enqueue.pfnUSMPrefetch;
    if( nullptr == pfnUSMPrefetch )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnUSMPrefetch( hQueue, pMem, size, flags, numEventsInWaitList, phEventWaitList, phEvent );
}

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
///         + `::UR_MEM_ADVICE_DEFAULT < advice`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEnqueueUSMMemAdvice(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    const void* pMem,                               ///< [in] pointer to the USM memory object
    size_t size,                                    ///< [in] size in bytes to be adviced
    ur_mem_advice_t advice,                         ///< [in] USM memory advice
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular command instance.
    )
{
    auto pfnUSMMemAdvice = ur_lib::context->urDdiTable.Enqueue.pfnUSMMemAdvice;
    if( nullptr == pfnUSMMemAdvice )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnUSMMemAdvice( hQueue, pMem, size, advice, phEvent );
}

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
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
ur_result_t UR_APICALL
urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue to submit to.
    void* pMem,                                     ///< [in] pointer to memory to be filled.
    size_t pitch,                                   ///< [in] the total width of the destination memory including padding.
    size_t patternSize,                             ///< [in] the size in bytes of the pattern.
    const void* pPattern,                           ///< [in] pointer with the bytes of the pattern to set.
    size_t width,                                   ///< [in] the width in bytes of each row to fill.
    size_t height,                                  ///< [in] the height of the columns to fill.
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before the kernel execution.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                                    ///< event. 
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular kernel execution instance.
    )
{
    auto pfnUSMFill2D = ur_lib::context->urDdiTable.Enqueue.pfnUSMFill2D;
    if( nullptr == pfnUSMFill2D )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnUSMFill2D( hQueue, pMem, pitch, patternSize, pPattern, width, height, numEventsInWaitList, phEventWaitList, phEvent );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to set 2D USM memory.
/// 
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
ur_result_t UR_APICALL
urEnqueueUSMMemset2D(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue to submit to.
    void* pMem,                                     ///< [in] pointer to memory to be filled.
    size_t pitch,                                   ///< [in] the total width of the destination memory including padding.
    int value,                                      ///< [in] the value to fill into the region in pMem.
    size_t width,                                   ///< [in] the width in bytes of each row to set.
    size_t height,                                  ///< [in] the height of the columns to set.
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before the kernel execution.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                                    ///< event. 
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular kernel execution instance.
    )
{
    auto pfnUSMMemset2D = ur_lib::context->urDdiTable.Enqueue.pfnUSMMemset2D;
    if( nullptr == pfnUSMMemset2D )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnUSMMemset2D( hQueue, pMem, pitch, value, width, height, numEventsInWaitList, phEventWaitList, phEvent );
}

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
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
ur_result_t UR_APICALL
urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue to submit to.
    bool blocking,                                  ///< [in] indicates if this operation should block the host.
    void* pDst,                                     ///< [in] pointer to memory where data will be copied.
    size_t dstPitch,                                ///< [in] the total width of the source memory including padding.
    const void* pSrc,                               ///< [in] pointer to memory to be copied.
    size_t srcPitch,                                ///< [in] the total width of the source memory including padding.
    size_t width,                                   ///< [in] the width in bytes of each row to be copied.
    size_t height,                                  ///< [in] the height of columns to be copied.
    uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list
    const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                    ///< events that must be complete before the kernel execution.
                                                    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                                    ///< event. 
    ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                    ///< particular kernel execution instance.
    )
{
    auto pfnUSMMemcpy2D = ur_lib::context->urDdiTable.Enqueue.pfnUSMMemcpy2D;
    if( nullptr == pfnUSMMemcpy2D )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnUSMMemcpy2D( hQueue, blocking, pDst, dstPitch, pSrc, srcPitch, width, height, numEventsInWaitList, phEventWaitList, phEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EVENT_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urEventGetInfo(
    ur_event_handle_t hEvent,                       ///< [in] handle of the event object
    ur_event_info_t propName,                       ///< [in] the name of the event property to query
    size_t propValueSize,                           ///< [in] size in bytes of the event property value
    void* pPropValue,                               ///< [out][optional] value of the event property
    size_t* pPropValueSizeRet                       ///< [out][optional] bytes returned in event property
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Event.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROFILING_INFO_COMMAND_END < propName`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urEventGetProfilingInfo(
    ur_event_handle_t hEvent,                       ///< [in] handle of the event object
    ur_profiling_info_t propName,                   ///< [in] the name of the profiling property to query
    size_t propValueSize,                           ///< [in] size in bytes of the profiling property value
    void* pPropValue,                               ///< [out][optional] value of the profiling property
    size_t* pPropValueSizeRet                       ///< [out][optional] pointer to the actual size in bytes returned in
                                                    ///< propValue
    )
{
    auto pfnGetProfilingInfo = ur_lib::context->urDdiTable.Event.pfnGetProfilingInfo;
    if( nullptr == pfnGetProfilingInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetProfilingInfo( hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEventWaitList`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urEventWait(
    uint32_t numEvents,                             ///< [in] number of events in the event list
    const ur_event_handle_t* phEventWaitList        ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                                                    ///< completion
    )
{
    auto pfnWait = ur_lib::context->urDdiTable.Event.pfnWait;
    if( nullptr == pfnWait )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnWait( numEvents, phEventWaitList );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urEventRetain(
    ur_event_handle_t hEvent                        ///< [in] handle of the event object
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Event.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urEventRelease(
    ur_event_handle_t hEvent                        ///< [in] handle of the event object
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Event.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeEvent`
ur_result_t UR_APICALL
urEventGetNativeHandle(
    ur_event_handle_t hEvent,                       ///< [in] handle of the event.
    ur_native_handle_t* phNativeEvent               ///< [out] a pointer to the native handle of the event.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Event.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hEvent, phNativeEvent );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeEvent`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEvent`
ur_result_t UR_APICALL
urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent,                ///< [in] the native handle of the event.
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_event_handle_t* phEvent                      ///< [out] pointer to the handle of the event object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Event.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeEvent, hContext, phEvent );
}

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
ur_result_t UR_APICALL
urEventSetCallback(
    ur_event_handle_t hEvent,                       ///< [in] handle of the event object
    ur_execution_info_t execStatus,                 ///< [in] execution status of the event
    ur_event_callback_t pfnNotify,                  ///< [in] execution status of the event
    void* pUserData                                 ///< [in][out][optional] pointer to data to be passed to callback.
    )
{
    auto pfnSetCallback = ur_lib::context->urDdiTable.Event.pfnSetCallback;
    if( nullptr == pfnSetCallback )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetCallback( hEvent, execStatus, pfnNotify, pUserData );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
///         + `::UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == pHost`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urMemImageCreate(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_mem_flags_t flags,                           ///< [in] allocation and usage information flags
    const ur_image_format_t* pImageFormat,          ///< [in] pointer to image format specification
    const ur_image_desc_t* pImageDesc,              ///< [in] pointer to image description
    void* pHost,                                    ///< [in] pointer to the buffer data
    ur_mem_handle_t* phMem                          ///< [out] pointer to handle of image object created
    )
{
    auto pfnImageCreate = ur_lib::context->urDdiTable.Mem.pfnImageCreate;
    if( nullptr == pfnImageCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnImageCreate( hContext, flags, pImageFormat, pImageDesc, pHost, phMem );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a memory buffer
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
///         + `NULL == pHost`
///         + `NULL == phBuffer`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_BUFFER_SIZE
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urMemBufferCreate(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_mem_flags_t flags,                           ///< [in] allocation and usage information flags
    size_t size,                                    ///< [in] size in bytes of the memory object to be allocated
    void* pHost,                                    ///< [in] pointer to the buffer data
    ur_mem_handle_t* phBuffer                       ///< [out] pointer to handle of the memory buffer created
    )
{
    auto pfnBufferCreate = ur_lib::context->urDdiTable.Mem.pfnBufferCreate;
    if( nullptr == pfnBufferCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnBufferCreate( hContext, flags, size, pHost, phBuffer );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urMemRetain(
    ur_mem_handle_t hMem                            ///< [in] handle of the memory object to get access
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Mem.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hMem );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urMemRelease(
    ur_mem_handle_t hMem                            ///< [in] handle of the memory object to release
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Mem.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hMem );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x3f < flags`
///         + `::UR_BUFFER_CREATE_TYPE_REGION < bufferCreateType`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pBufferCreateInfo`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_BUFFER_SIZE
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urMemBufferPartition(
    ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t flags,                           ///< [in] allocation and usage information flags
    ur_buffer_create_type_t bufferCreateType,       ///< [in] buffer creation type
    ur_buffer_region_t* pBufferCreateInfo,          ///< [in] pointer to buffer create region information
    ur_mem_handle_t* phMem                          ///< [out] pointer to the handle of sub buffer created
    )
{
    auto pfnBufferPartition = ur_lib::context->urDdiTable.Mem.pfnBufferPartition;
    if( nullptr == pfnBufferPartition )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnBufferPartition( hBuffer, flags, bufferCreateType, pBufferCreateInfo, phMem );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeMem`
ur_result_t UR_APICALL
urMemGetNativeHandle(
    ur_mem_handle_t hMem,                           ///< [in] handle of the mem.
    ur_native_handle_t* phNativeMem                 ///< [out] a pointer to the native handle of the mem.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Mem.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hMem, phNativeMem );
}

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
ur_result_t UR_APICALL
urMemCreateWithNativeHandle(
    ur_native_handle_t hNativeMem,                  ///< [in] the native handle of the mem.
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_mem_handle_t* phMem                          ///< [out] pointer to the handle of the mem object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Mem.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeMem, hContext, phMem );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_INFO_CONTEXT < MemInfoType`
ur_result_t UR_APICALL
urMemGetInfo(
    ur_mem_handle_t hMemory,                        ///< [in] handle to the memory object being queried.
    ur_mem_info_t MemInfoType,                      ///< [in] type of the info to retrieve.
    size_t propSize,                                ///< [in] the number of bytes of memory pointed to by pMemInfo.
    void* pMemInfo,                                 ///< [out][optional] array of bytes holding the info.
                                                    ///< If propSize is less than the real number of bytes needed to return 
                                                    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                                    ///< pMemInfo is not used.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data queried by pMemInfo.
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Mem.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hMemory, MemInfoType, propSize, pMemInfo, pPropSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_IMAGE_INFO_DEPTH < ImgInfoType`
ur_result_t UR_APICALL
urMemImageGetInfo(
    ur_mem_handle_t hMemory,                        ///< [in] handle to the image object being queried.
    ur_image_info_t ImgInfoType,                    ///< [in] type of image info to retrieve.
    size_t propSize,                                ///< [in] the number of bytes of memory pointer to by pImgInfo.
    void* pImgInfo,                                 ///< [out][optional] array of bytes holding the info.
                                                    ///< If propSize is less than the real number of bytes needed to return
                                                    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                                    ///< pImgInfo is not used.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data queried by pImgInfo.
    )
{
    auto pfnImageGetInfo = ur_lib::context->urDdiTable.Mem.pfnImageGetInfo;
    if( nullptr == pfnImageGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnImageGetInfo( hMemory, ImgInfoType, propSize, pImgInfo, pPropSizeRet );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Tear down L0 runtime instance and release all its resources
/// 
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pParams`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urTearDown(
    void* pParams                                   ///< [in] pointer to tear down parameters
    )
{
    auto pfnTearDown = ur_lib::context->urDdiTable.Global.pfnTearDown;
    if( nullptr == pfnTearDown )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnTearDown( pParams );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_QUEUE_INFO_SIZE < propName`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPropValue`
///         + `NULL == pPropSizeRet`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urQueueGetInfo(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
    ur_queue_info_t propName,                       ///< [in] name of the queue property to query
    size_t propValueSize,                           ///< [in] size in bytes of the queue property value provided
    void* pPropValue,                               ///< [out] value of the queue property
    size_t* pPropSizeRet                            ///< [out] size in bytes returned in queue property value
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Queue.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hQueue, propName, propValueSize, pPropValue, pPropSizeRet );
}

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
///         + `NULL == pProps`
///         + `NULL == phQueue`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urQueueCreate(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_device_handle_t hDevice,                     ///< [in] handle of the device object
    ur_queue_property_value_t* pProps,              ///< [in] specifies a list of queue properties and their corresponding values.
                                                    ///< Each property name is immediately followed by the corresponding
                                                    ///< desired value.
                                                    ///< The list is terminated with a 0. 
                                                    ///< If a property value is not specified, then its default value will be used.
    ur_queue_handle_t* phQueue                      ///< [out] pointer to handle of queue object created
    )
{
    auto pfnCreate = ur_lib::context->urDdiTable.Queue.pfnCreate;
    if( nullptr == pfnCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreate( hContext, hDevice, pProps, phQueue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urQueueRetain(
    ur_queue_handle_t hQueue                        ///< [in] handle of the queue object to get access
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Queue.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hQueue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urQueueRelease(
    ur_queue_handle_t hQueue                        ///< [in] handle of the queue object to release
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Queue.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hQueue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeQueue`
ur_result_t UR_APICALL
urQueueGetNativeHandle(
    ur_queue_handle_t hQueue,                       ///< [in] handle of the queue.
    ur_native_handle_t* phNativeQueue               ///< [out] a pointer to the native handle of the queue.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Queue.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hQueue, phNativeQueue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeQueue`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phQueue`
ur_result_t UR_APICALL
urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue,                ///< [in] the native handle of the queue.
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_queue_handle_t* phQueue                      ///< [out] pointer to the handle of the queue object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Queue.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeQueue, hContext, phQueue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urQueueFinish(
    ur_queue_handle_t hQueue                        ///< [in] handle of the queue to be finished.
    )
{
    auto pfnFinish = ur_lib::context->urDdiTable.Queue.pfnFinish;
    if( nullptr == pfnFinish )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnFinish( hQueue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urQueueFlush(
    ur_queue_handle_t hQueue                        ///< [in] handle of the queue to be flushed.
    )
{
    auto pfnFlush = ur_lib::context->urDdiTable.Queue.pfnFlush;
    if( nullptr == pfnFlush )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnFlush( hQueue );
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
ur_result_t UR_APICALL
urSamplerCreate(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    const ur_sampler_property_value_t* pProps,      ///< [in] specifies a list of sampler property names and their
                                                    ///< corresponding values.
    ur_sampler_handle_t* phSampler                  ///< [out] pointer to handle of sampler object created
    )
{
    auto pfnCreate = ur_lib::context->urDdiTable.Sampler.pfnCreate;
    if( nullptr == pfnCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreate( hContext, pProps, phSampler );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urSamplerRetain(
    ur_sampler_handle_t hSampler                    ///< [in] handle of the sampler object to get access
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Sampler.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hSampler );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urSamplerRelease(
    ur_sampler_handle_t hSampler                    ///< [in] handle of the sampler object to release
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Sampler.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hSampler );
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
ur_result_t UR_APICALL
urSamplerGetInfo(
    ur_sampler_handle_t hSampler,                   ///< [in] handle of the sampler object
    ur_sampler_info_t propName,                     ///< [in] name of the sampler property to query
    size_t propValueSize,                           ///< [in] size in bytes of the sampler property value provided
    void* pPropValue,                               ///< [out] value of the sampler property
    size_t* pPropSizeRet                            ///< [out] size in bytes returned in sampler property value
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Sampler.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hSampler, propName, propValueSize, pPropValue, pPropSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeSampler`
ur_result_t UR_APICALL
urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler,                   ///< [in] handle of the sampler.
    ur_native_handle_t* phNativeSampler             ///< [out] a pointer to the native handle of the sampler.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Sampler.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hSampler, phNativeSampler );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeSampler`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phSampler`
ur_result_t UR_APICALL
urSamplerCreateWithNativeHandle(
    ur_native_handle_t hNativeSampler,              ///< [in] the native handle of the sampler.
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_sampler_handle_t* phSampler                  ///< [out] pointer to the handle of the sampler object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Sampler.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeSampler, hContext, phSampler );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate host memory
/// 
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pUSMFlag`
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urUSMHostAlloc(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_usm_mem_flags_t* pUSMFlag,                   ///< [in] USM memory allocation flags
    size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,                                 ///< [in] alignment of the USM memory object
    void** ppMem                                    ///< [out] pointer to USM host memory object
    )
{
    auto pfnHostAlloc = ur_lib::context->urDdiTable.USM.pfnHostAlloc;
    if( nullptr == pfnHostAlloc )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnHostAlloc( hContext, pUSMFlag, size, align, ppMem );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate device memory
/// 
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pUSMProp`
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urUSMDeviceAlloc(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_device_handle_t hDevice,                     ///< [in] handle of the device object
    ur_usm_mem_flags_t* pUSMProp,                   ///< [in] USM memory properties
    size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,                                 ///< [in] alignment of the USM memory object
    void** ppMem                                    ///< [out] pointer to USM device memory object
    )
{
    auto pfnDeviceAlloc = ur_lib::context->urDdiTable.USM.pfnDeviceAlloc;
    if( nullptr == pfnDeviceAlloc )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnDeviceAlloc( hContext, hDevice, pUSMProp, size, align, ppMem );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate shared memory
/// 
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pUSMProp`
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
ur_result_t UR_APICALL
urUSMSharedAlloc(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_device_handle_t hDevice,                     ///< [in] handle of the device object
    ur_usm_mem_flags_t* pUSMProp,                   ///< [in] USM memory properties
    size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
    uint32_t align,                                 ///< [in] alignment of the USM memory object
    void** ppMem                                    ///< [out] pointer to USM shared memory object
    )
{
    auto pfnSharedAlloc = ur_lib::context->urDdiTable.USM.pfnSharedAlloc;
    if( nullptr == pfnSharedAlloc )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSharedAlloc( hContext, hDevice, pUSMProp, size, align, ppMem );
}

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
ur_result_t UR_APICALL
urMemFree(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    void* pMem                                      ///< [in] pointer to USM memory object
    )
{
    auto pfnFree = ur_lib::context->urDdiTable.Mem.pfnFree;
    if( nullptr == pfnFree )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnFree( hContext, pMem );
}

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
///         + `NULL == pPropValue`
///         + `NULL == pPropValueSizeRet`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_ALLOC_INFO_ALLOC_DEVICE < propName`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urMemGetMemAllocInfo(
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    const void* pMem,                               ///< [in] pointer to USM memory object
    ur_mem_alloc_info_t propName,                   ///< [in] the name of the USM allocation property to query
    size_t propValueSize,                           ///< [in] size in bytes of the USM allocation property value
    void* pPropValue,                               ///< [out] value of the USM allocation property
    size_t* pPropValueSizeRet                       ///< [out] bytes returned in USM allocation property
    )
{
    auto pfnGetMemAllocInfo = ur_lib::context->urDdiTable.Mem.pfnGetMemAllocInfo;
    if( nullptr == pfnGetMemAllocInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetMemAllocInfo( hContext, pMem, propName, propValueSize, pPropValue, pPropValueSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_TYPE_VPU < DeviceType`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
ur_result_t UR_APICALL
urDeviceGet(
    ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform instance
    ur_device_type_t DeviceType,                    ///< [in] the type of the devices.
    uint32_t NumEntries,                            ///< [in] the number of devices to be added to phDevices.
                                                    ///< If phDevices in not NULL then NumEntries should be greater than zero,
                                                    ///< otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
                                                    ///< will be returned.
    ur_device_handle_t* phDevices,                  ///< [out][optional][range(0, NumEntries)] array of handle of devices.
                                                    ///< If NumEntries is less than the number of devices available, then
                                                    ///< platform shall only retrieve that number of devices.
    uint32_t* pNumDevices                           ///< [out][optional] pointer to the number of devices.
                                                    ///< pNumDevices will be updated with the total number of devices available.
    )
{
    auto pfnGet = ur_lib::context->urDdiTable.Device.pfnGet;
    if( nullptr == pfnGet )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGet( hPlatform, DeviceType, NumEntries, phDevices, pNumDevices );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES < infoType`
ur_result_t UR_APICALL
urDeviceGetInfo(
    ur_device_handle_t hDevice,                     ///< [in] handle of the device instance
    ur_device_info_t infoType,                      ///< [in] type of the info to retrieve
    size_t propSize,                                ///< [in] the number of bytes pointed to by pDeviceInfo.
    void* pDeviceInfo,                              ///< [out][optional] array of bytes holding the info.
                                                    ///< If propSize is not equal to or greater than the real number of bytes
                                                    ///< needed to return the info
                                                    ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                                    ///< pDeviceInfo is not used.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of the queried infoType.
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Device.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hDevice, infoType, propSize, pDeviceInfo, pPropSizeRet );
}

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
ur_result_t UR_APICALL
urDeviceRetain(
    ur_device_handle_t hDevice                      ///< [in] handle of the device to get a reference of.
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Device.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hDevice );
}

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
ur_result_t UR_APICALL
urDeviceRelease(
    ur_device_handle_t hDevice                      ///< [in] handle of the device to release.
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Device.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hDevice );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == Properties`
ur_result_t UR_APICALL
urDevicePartition(
    ur_device_handle_t hDevice,                     ///< [in] handle of the device to partition.
    ur_device_partition_property_value_t* Properties,   ///< [in] null-terminated array of <property, value> pair of the requested partitioning.
    uint32_t NumDevices,                            ///< [in] the number of sub-devices.
    ur_device_handle_t* phSubDevices,               ///< [out][optional][range(0, NumDevices)] array of handle of devices.
                                                    ///< If NumDevices is less than the number of sub-devices available, then
                                                    ///< the function shall only retrieve that number of sub-devices.
    uint32_t* pNumDevicesRet                        ///< [out][optional] pointer to the number of sub-devices the device can be
                                                    ///< partitioned into according to the partitioning property.
    )
{
    auto pfnPartition = ur_lib::context->urDdiTable.Device.pfnPartition;
    if( nullptr == pfnPartition )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnPartition( hDevice, Properties, NumDevices, phSubDevices, pNumDevicesRet );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppBinaries`
///         + `NULL == pSelectedBinary`
ur_result_t UR_APICALL
urDeviceSelectBinary(
    ur_device_handle_t hDevice,                     ///< [in] handle of the device to select binary for.
    const uint8_t** ppBinaries,                     ///< [in] the array of binaries to select from.
    uint32_t NumBinaries,                           ///< [in] the number of binaries passed in ppBinaries. Must greater than or
                                                    ///< equal to zero.
    uint32_t* pSelectedBinary                       ///< [out] the index of the selected binary in the input array of binaries.
                                                    ///< If a suitable binary was not found the function returns ${X}_INVALID_BINARY.
    )
{
    auto pfnSelectBinary = ur_lib::context->urDdiTable.Device.pfnSelectBinary;
    if( nullptr == pfnSelectBinary )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSelectBinary( hDevice, ppBinaries, NumBinaries, pSelectedBinary );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeDevice`
ur_result_t UR_APICALL
urDeviceGetNativeHandle(
    ur_device_handle_t hDevice,                     ///< [in] handle of the device.
    ur_native_handle_t* phNativeDevice              ///< [out] a pointer to the native handle of the device.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Device.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hDevice, phNativeDevice );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeDevice`
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevice`
ur_result_t UR_APICALL
urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice,               ///< [in] the native handle of the device.
    ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform instance
    ur_device_handle_t* phDevice                    ///< [out] pointer to the handle of the device object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Device.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeDevice, hPlatform, phDevice );
}

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
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDeviceTimestamp`
///         + `NULL == pHostTimestamp`
ur_result_t UR_APICALL
urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice,                     ///< [in] handle of the device instance
    uint64_t* pDeviceTimestamp,                     ///< [out] pointer to the Device's global timestamp that 
                                                    ///< correlates with the Host's global timestamp value
    uint64_t* pHostTimestamp                        ///< [out] pointer to the Host's global timestamp that 
                                                    ///< correlates with the Device's global timestamp value
    )
{
    auto pfnGetGlobalTimestamps = ur_lib::context->urDdiTable.Device.pfnGetGlobalTimestamps;
    if( nullptr == pfnGetGlobalTimestamps )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetGlobalTimestamps( hDevice, pDeviceTimestamp, pHostTimestamp );
}

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
ur_result_t UR_APICALL
urKernelCreate(
    ur_program_handle_t hProgram,                   ///< [in] handle of the program instance
    const char* pKernelName,                        ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t* phKernel                    ///< [out] pointer to handle of kernel object created.
    )
{
    auto pfnCreate = ur_lib::context->urDdiTable.Kernel.pfnCreate;
    if( nullptr == pfnCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreate( hProgram, pKernelName, phKernel );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
ur_result_t UR_APICALL
urKernelSetArgValue(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                 ///< [in] size of argument type
    const void* pArgValue                           ///< [in] argument value represented as matching arg type.
    )
{
    auto pfnSetArgValue = ur_lib::context->urDdiTable.Kernel.pfnSetArgValue;
    if( nullptr == pfnSetArgValue )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetArgValue( hKernel, argIndex, argSize, pArgValue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
ur_result_t UR_APICALL
urKernelSetArgLocal(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    size_t argSize                                  ///< [in] size of the local buffer to be allocated by the runtime
    )
{
    auto pfnSetArgLocal = ur_lib::context->urDdiTable.Kernel.pfnSetArgLocal;
    if( nullptr == pfnSetArgLocal )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetArgLocal( hKernel, argIndex, argSize );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_INFO_ATTRIBUTES < propName`
ur_result_t UR_APICALL
urKernelGetInfo(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the Kernel object
    ur_kernel_info_t propName,                      ///< [in] name of the Kernel property to query
    size_t propSize,                                ///< [in] the size of the Kernel property value.           
    void* pKernelInfo,                              ///< [in,out][optional] array of bytes holding the kernel info property.
                                                    ///< If propSize is not equal to or greater than the real number of bytes
                                                    ///< needed to return 
                                                    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                                    ///< pKernelInfo is not used.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data being
                                                    ///< queried by propName.
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Kernel.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hKernel, propName, propSize, pKernelInfo, pPropSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE < propName`
ur_result_t UR_APICALL
urKernelGetGroupInfo(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,                     ///< [in] handle of the Device object
    ur_kernel_group_info_t propName,                ///< [in] name of the work Group property to query
    size_t propSize,                                ///< [in] size of the Kernel Work Group property value
    void* pPropValue,                               ///< [in,out][optional][range(0, propSize)] value of the Kernel Work Group
                                                    ///< property.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data being
                                                    ///< queried by propName.
    )
{
    auto pfnGetGroupInfo = ur_lib::context->urDdiTable.Kernel.pfnGetGroupInfo;
    if( nullptr == pfnGetGroupInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetGroupInfo( hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet );
}

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
ur_result_t UR_APICALL
urKernelGetSubGroupInfo(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,                     ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t propName,            ///< [in] name of the SubGroup property to query
    size_t propSize,                                ///< [in] size of the Kernel SubGroup property value
    void* pPropValue,                               ///< [in,out][range(0, propSize)][optional] value of the Kernel SubGroup
                                                    ///< property.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data being
                                                    ///< queried by propName.
    )
{
    auto pfnGetSubGroupInfo = ur_lib::context->urDdiTable.Kernel.pfnGetSubGroupInfo;
    if( nullptr == pfnGetSubGroupInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetSubGroupInfo( hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
ur_result_t UR_APICALL
urKernelRetain(
    ur_kernel_handle_t hKernel                      ///< [in] handle for the Kernel to retain
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Kernel.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hKernel );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
ur_result_t UR_APICALL
urKernelRelease(
    ur_kernel_handle_t hKernel                      ///< [in] handle for the Kernel to release
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Kernel.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hKernel );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
ur_result_t UR_APICALL
urKernelSetArgPointer(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                 ///< [in] size of argument type
    const void* pArgValue                           ///< [in][optional] SVM pointer to memory location holding the argument
                                                    ///< value. If null then argument value is considered null.
    )
{
    auto pfnSetArgPointer = ur_lib::context->urDdiTable.Kernel.pfnSetArgPointer;
    if( nullptr == pfnSetArgPointer )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetArgPointer( hKernel, argIndex, argSize, pArgValue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_EXEC_INFO_USM_PTRS < propName`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPropValue`
ur_result_t UR_APICALL
urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName,                 ///< [in] name of the execution attribute
    size_t propSize,                                ///< [in] size in byte the attribute value
    const void* pPropValue                          ///< [in][range(0, propSize)] pointer to memory location holding the
                                                    ///< property value.
    )
{
    auto pfnSetExecInfo = ur_lib::context->urDdiTable.Kernel.pfnSetExecInfo;
    if( nullptr == pfnSetExecInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetExecInfo( hKernel, propName, propSize, pPropValue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
ur_result_t UR_APICALL
urKernelSetArgSampler(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    ur_sampler_handle_t hArgValue                   ///< [in] handle of Sampler object.
    )
{
    auto pfnSetArgSampler = ur_lib::context->urDdiTable.Kernel.pfnSetArgSampler;
    if( nullptr == pfnSetArgSampler )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetArgSampler( hKernel, argIndex, hArgValue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
ur_result_t UR_APICALL
urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    ur_mem_handle_t hArgValue                       ///< [in][optional] handle of Memory object.
    )
{
    auto pfnSetArgMemObj = ur_lib::context->urDdiTable.Kernel.pfnSetArgMemObj;
    if( nullptr == pfnSetArgMemObj )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetArgMemObj( hKernel, argIndex, hArgValue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeKernel`
ur_result_t UR_APICALL
urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel.
    ur_native_handle_t* phNativeKernel              ///< [out] a pointer to the native handle of the kernel.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Kernel.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hKernel, phNativeKernel );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeKernel`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phKernel`
ur_result_t UR_APICALL
urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel,               ///< [in] the native handle of the kernel.
    ur_context_handle_t hContext,                   ///< [in] handle of the context object
    ur_kernel_handle_t* phKernel                    ///< [out] pointer to the handle of the kernel object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Kernel.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeKernel, hContext, phKernel );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pIL`
///         + `NULL == pOptions`
///         + `NULL == phModule`
ur_result_t UR_APICALL
urModuleCreate(
    ur_context_handle_t hContext,                   ///< [in] handle of the context instance.
    const void* pIL,                                ///< [in] pointer to IL string.
    size_t length,                                  ///< [in] length of IL in bytes.
    const char* pOptions,                           ///< [in] pointer to compiler options null-terminated string.
    ur_modulecreate_callback_t pfnNotify,           ///< [in][optional] A function pointer to a notification routine that is
                                                    ///< called when program compilation is complete.
    void* pUserData,                                ///< [in][optional] Passed as an argument when pfnNotify is called.
    ur_module_handle_t* phModule                    ///< [out] pointer to handle of Module object created.
    )
{
    auto pfnCreate = ur_lib::context->urDdiTable.Module.pfnCreate;
    if( nullptr == pfnCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreate( hContext, pIL, length, pOptions, pfnNotify, pUserData, phModule );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hModule`
ur_result_t UR_APICALL
urModuleRetain(
    ur_module_handle_t hModule                      ///< [in] handle for the Module to retain
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Module.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hModule );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hModule`
ur_result_t UR_APICALL
urModuleRelease(
    ur_module_handle_t hModule                      ///< [in] handle for the Module to release
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Module.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hModule );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hModule`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeModule`
ur_result_t UR_APICALL
urModuleGetNativeHandle(
    ur_module_handle_t hModule,                     ///< [in] handle of the module.
    ur_native_handle_t* phNativeModule              ///< [out] a pointer to the native handle of the module.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Module.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hModule, phNativeModule );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeModule`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phModule`
ur_result_t UR_APICALL
urModuleCreateWithNativeHandle(
    ur_native_handle_t hNativeModule,               ///< [in] the native handle of the module.
    ur_context_handle_t hContext,                   ///< [in] handle of the context instance.
    ur_module_handle_t* phModule                    ///< [out] pointer to the handle of the module object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Module.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeModule, hContext, phModule );
}

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
ur_result_t UR_APICALL
urPlatformGet(
    uint32_t NumEntries,                            ///< [in] the number of platforms to be added to phPlatforms. 
                                                    ///< If phPlatforms is not NULL, then NumEntries should be greater than
                                                    ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
                                                    ///< will be returned.
    ur_platform_handle_t* phPlatforms,              ///< [out][optional][range(0, NumEntries)] array of handle of platforms.
                                                    ///< If NumEntries is less than the number of platforms available, then
                                                    ///< ::urPlatformGet shall only retrieve that number of platforms.
    uint32_t* pNumPlatforms                         ///< [out][optional] returns the total number of platforms available. 
    )
{
    auto pfnGet = ur_lib::context->urDdiTable.Platform.pfnGet;
    if( nullptr == pfnGet )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGet( NumEntries, phPlatforms, pNumPlatforms );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PLATFORM_INFO_PROFILE < PlatformInfoType`
ur_result_t UR_APICALL
urPlatformGetInfo(
    ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform
    ur_platform_info_t PlatformInfoType,            ///< [in] type of the info to retrieve
    size_t Size,                                    ///< [in] the number of bytes pointed to by pPlatformInfo.
    void* pPlatformInfo,                            ///< [out][optional] array of bytes holding the info.
                                                    ///< If Size is not equal to or greater to the real number of bytes needed
                                                    ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                                                    ///< returned and pPlatformInfo is not used.
    size_t* pSizeRet                                ///< [out][optional] pointer to the actual number of bytes being queried by pPlatformInfo.
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Platform.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hPlatform, PlatformInfoType, Size, pPlatformInfo, pSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDriver`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pVersion`
ur_result_t UR_APICALL
urPlatformGetApiVersion(
    ur_platform_handle_t hDriver,                   ///< [in] handle of the platform
    ur_api_version_t* pVersion                      ///< [out] api version
    )
{
    auto pfnGetApiVersion = ur_lib::context->urDdiTable.Platform.pfnGetApiVersion;
    if( nullptr == pfnGetApiVersion )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetApiVersion( hDriver, pVersion );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativePlatform`
ur_result_t UR_APICALL
urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform.
    ur_native_handle_t* phNativePlatform            ///< [out] a pointer to the native handle of the platform.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Platform.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hPlatform, phNativePlatform );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativePlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPlatform`
ur_result_t UR_APICALL
urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform,             ///< [in] the native handle of the platform.
    ur_platform_handle_t* phPlatform                ///< [out] pointer to the handle of the platform object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Platform.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativePlatform, phPlatform );
}

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
ur_result_t UR_APICALL
urGetLastResult(
    ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform instance
    const char** ppMessage                          ///< [out] pointer to a string containing adapter specific result in string
                                                    ///< representation.
    )
{
    auto pfnGetLastResult = ur_lib::context->urDdiTable.Global.pfnGetLastResult;
    if( nullptr == pfnGetLastResult )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetLastResult( hPlatform, ppMessage );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phModules`
///         + `NULL == phProgram`
ur_result_t UR_APICALL
urProgramCreate(
    ur_context_handle_t hContext,                   ///< [in] handle of the context instance
    uint32_t count,                                 ///< [in] number of module handles in module list.
    const ur_module_handle_t* phModules,            ///< [in][range(0, count)] pointer to array of modules.
    const char* pOptions,                           ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t* phProgram                  ///< [out] pointer to handle of program object created.
    )
{
    auto pfnCreate = ur_lib::context->urDdiTable.Program.pfnCreate;
    if( nullptr == pfnCreate )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreate( hContext, count, phModules, pOptions, phProgram );
}

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
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pBinary`
///         + `NULL == phProgram`
ur_result_t UR_APICALL
urProgramCreateWithBinary(
    ur_context_handle_t hContext,                   ///< [in] handle of the context instance
    ur_device_handle_t hDevice,                     ///< [in] handle to device associated with binary.
    size_t size,                                    ///< [in] size in bytes.
    const uint8_t* pBinary,                         ///< [in] pointer to binary.
    ur_program_handle_t* phProgram                  ///< [out] pointer to handle of Program object created.
    )
{
    auto pfnCreateWithBinary = ur_lib::context->urDdiTable.Program.pfnCreateWithBinary;
    if( nullptr == pfnCreateWithBinary )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithBinary( hContext, hDevice, size, pBinary, phProgram );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
ur_result_t UR_APICALL
urProgramRetain(
    ur_program_handle_t hProgram                    ///< [in] handle for the Program to retain
    )
{
    auto pfnRetain = ur_lib::context->urDdiTable.Program.pfnRetain;
    if( nullptr == pfnRetain )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRetain( hProgram );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
ur_result_t UR_APICALL
urProgramRelease(
    ur_program_handle_t hProgram                    ///< [in] handle for the Program to release
    )
{
    auto pfnRelease = ur_lib::context->urDdiTable.Program.pfnRelease;
    if( nullptr == pfnRelease )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnRelease( hProgram );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pFunctionName`
///         + `NULL == ppFunctionPointer`
ur_result_t UR_APICALL
urProgramGetFunctionPointer(
    ur_device_handle_t hDevice,                     ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t hProgram,                   ///< [in] handle of the program to search for function in.
                                                    ///< The program must already be built to the specified device, or
                                                    ///< otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char* pFunctionName,                      ///< [in] A null-terminates string denoting the mangled function name.
    void** ppFunctionPointer                        ///< [out] Returns the pointer to the function if it is found in the program.
    )
{
    auto pfnGetFunctionPointer = ur_lib::context->urDdiTable.Program.pfnGetFunctionPointer;
    if( nullptr == pfnGetFunctionPointer )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetFunctionPointer( hDevice, hProgram, pFunctionName, ppFunctionPointer );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_INFO_KERNEL_NAMES < propName`
ur_result_t UR_APICALL
urProgramGetInfo(
    ur_program_handle_t hProgram,                   ///< [in] handle of the Program object
    ur_program_info_t propName,                     ///< [in] name of the Program property to query
    size_t propSize,                                ///< [in] the size of the Program property.
    void* pProgramInfo,                             ///< [in,out][optional] array of bytes of holding the program info property.
                                                    ///< If propSize is not equal to or greater than the real number of bytes
                                                    ///< needed to return 
                                                    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                                    ///< pProgramInfo is not used.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data copied to propName.
    )
{
    auto pfnGetInfo = ur_lib::context->urDdiTable.Program.pfnGetInfo;
    if( nullptr == pfnGetInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetInfo( hProgram, propName, propSize, pProgramInfo, pPropSizeRet );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_BUILD_INFO_BINARY_TYPE < propName`
ur_result_t UR_APICALL
urProgramGetBuildInfo(
    ur_program_handle_t hProgram,                   ///< [in] handle of the Program object
    ur_device_handle_t hDevice,                     ///< [in] handle of the Device object
    ur_program_build_info_t propName,               ///< [in] name of the Program build info to query
    size_t propSize,                                ///< [in] size of the Program build info property.
    void* pPropValue,                               ///< [in,out][optional] value of the Program build property.
                                                    ///< If propSize is not equal to or greater than the real number of bytes
                                                    ///< needed to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE
                                                    ///< error is returned and pKernelInfo is not used.
    size_t* pPropSizeRet                            ///< [out][optional] pointer to the actual size in bytes of data being
                                                    ///< queried by propName.
    )
{
    auto pfnGetBuildInfo = ur_lib::context->urDdiTable.Program.pfnGetBuildInfo;
    if( nullptr == pfnGetBuildInfo )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetBuildInfo( hProgram, hDevice, propName, propSize, pPropValue, pPropSizeRet );
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Program object specialization constant to a specific value
/// 
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSpecValue`
ur_result_t UR_APICALL
urProgramSetSpecializationConstant(
    ur_program_handle_t hProgram,                   ///< [in] handle of the Program object
    uint32_t specId,                                ///< [in] specification constant Id
    size_t specSize,                                ///< [in] size of the specialization constant value
    const void* pSpecValue                          ///< [in] pointer to the specialization value bytes
    )
{
    auto pfnSetSpecializationConstant = ur_lib::context->urDdiTable.Program.pfnSetSpecializationConstant;
    if( nullptr == pfnSetSpecializationConstant )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnSetSpecializationConstant( hProgram, specId, specSize, pSpecValue );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeProgram`
ur_result_t UR_APICALL
urProgramGetNativeHandle(
    ur_program_handle_t hProgram,                   ///< [in] handle of the program.
    ur_native_handle_t* phNativeProgram             ///< [out] a pointer to the native handle of the program.
    )
{
    auto pfnGetNativeHandle = ur_lib::context->urDdiTable.Program.pfnGetNativeHandle;
    if( nullptr == pfnGetNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnGetNativeHandle( hProgram, phNativeProgram );
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
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hNativeProgram`
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phProgram`
ur_result_t UR_APICALL
urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram,              ///< [in] the native handle of the program.
    ur_context_handle_t hContext,                   ///< [in] handle of the context instance
    ur_program_handle_t* phProgram                  ///< [out] pointer to the handle of the program object created.
    )
{
    auto pfnCreateWithNativeHandle = ur_lib::context->urDdiTable.Program.pfnCreateWithNativeHandle;
    if( nullptr == pfnCreateWithNativeHandle )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnCreateWithNativeHandle( hNativeProgram, hContext, phProgram );
}

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
///         + `0x1 < platform_flags`
///         + `0x1 < device_flags`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
ur_result_t UR_APICALL
urInit(
    ur_platform_init_flags_t platform_flags,        ///< [in] platform initialization flags.
                                                    ///< must be 0 (default) or a combination of ::ur_platform_init_flag_t.
    ur_device_init_flags_t device_flags             ///< [in] device initialization flags.
                                                    ///< must be 0 (default) or a combination of ::ur_device_init_flag_t.
    )
{
    static ur_result_t result = UR_RESULT_SUCCESS;
    std::call_once(ur_lib::context->initOnce, [platform_flags, device_flags]() {
        result = ur_lib::context->Init(platform_flags, device_flags);
    });

    if( UR_RESULT_SUCCESS != result )
        return result;

    auto pfnInit = ur_lib::context->urDdiTable.Global.pfnInit;
    if( nullptr == pfnInit )
        return UR_RESULT_ERROR_UNINITIALIZED;

    return pfnInit( platform_flags, device_flags );
}

} // extern "C"
