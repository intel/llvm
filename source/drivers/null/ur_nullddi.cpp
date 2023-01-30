/*
 *
 * Copyright (C) 2019-2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_nullddi.cpp
 *
 */
#include "ur_null.h"

namespace driver
{
    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urContextCreate
    __urdlllocal ur_result_t UR_APICALL
    urContextCreate(
        uint32_t DeviceCount,                           ///< [in] the number of devices given in phDevices
        ur_device_handle_t* phDevices,                  ///< [in][range(0, DeviceCount)] array of handle of devices.
        ur_context_handle_t* phContext                  ///< [out] pointer to handle of context object created
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreate = d_context.urDdiTable.Context.pfnCreate;
        if( nullptr != pfnCreate )
        {
            result = pfnCreate( DeviceCount, phDevices, phContext );
        }
        else
        {
            // generic implementation
            *phContext = reinterpret_cast<ur_context_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urContextRetain
    __urdlllocal ur_result_t UR_APICALL
    urContextRetain(
        ur_context_handle_t hContext                    ///< [in] handle of the context to get a reference of.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Context.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hContext );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urContextRelease
    __urdlllocal ur_result_t UR_APICALL
    urContextRelease(
        ur_context_handle_t hContext                    ///< [in] handle of the context to release.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Context.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hContext );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urContextGetInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Context.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hContext, ContextInfoType, propSize, pContextInfo, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urContextGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urContextGetNativeHandle(
        ur_context_handle_t hContext,                   ///< [in] handle of the context.
        ur_native_handle_t* phNativeContext             ///< [out] a pointer to the native handle of the context.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Context.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hContext, phNativeContext );
        }
        else
        {
            // generic implementation
            *phNativeContext = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urContextCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urContextCreateWithNativeHandle(
        ur_native_handle_t hNativeContext,              ///< [in] the native handle of the context.
        ur_context_handle_t* phContext                  ///< [out] pointer to the handle of the context object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Context.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeContext, phContext );
        }
        else
        {
            // generic implementation
            *phContext = reinterpret_cast<ur_context_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urContextSetExtendedDeleter
    __urdlllocal ur_result_t UR_APICALL
    urContextSetExtendedDeleter(
        ur_context_handle_t hContext,                   ///< [in] handle of the context.
        ur_context_extended_deleter_t pfnDeleter,       ///< [in] Function pointer to extended deleter.
        void* pUserData                                 ///< [in][out] pointer to data to be passed to callback.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetExtendedDeleter = d_context.urDdiTable.Context.pfnSetExtendedDeleter;
        if( nullptr != pfnSetExtendedDeleter )
        {
            result = pfnSetExtendedDeleter( hContext, pfnDeleter, pUserData );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueKernelLaunch
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnKernelLaunch = d_context.urDdiTable.Enqueue.pfnKernelLaunch;
        if( nullptr != pfnKernelLaunch )
        {
            result = pfnKernelLaunch( hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueEventsWait
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnEventsWait = d_context.urDdiTable.Enqueue.pfnEventsWait;
        if( nullptr != pfnEventsWait )
        {
            result = pfnEventsWait( hQueue, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueEventsWaitWithBarrier
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnEventsWaitWithBarrier = d_context.urDdiTable.Enqueue.pfnEventsWaitWithBarrier;
        if( nullptr != pfnEventsWaitWithBarrier )
        {
            result = pfnEventsWaitWithBarrier( hQueue, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferRead
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferRead = d_context.urDdiTable.Enqueue.pfnMemBufferRead;
        if( nullptr != pfnMemBufferRead )
        {
            result = pfnMemBufferRead( hQueue, hBuffer, blockingRead, offset, size, pDst, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferWrite
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferWrite = d_context.urDdiTable.Enqueue.pfnMemBufferWrite;
        if( nullptr != pfnMemBufferWrite )
        {
            result = pfnMemBufferWrite( hQueue, hBuffer, blockingWrite, offset, size, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferReadRect
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferReadRect = d_context.urDdiTable.Enqueue.pfnMemBufferReadRect;
        if( nullptr != pfnMemBufferReadRect )
        {
            result = pfnMemBufferReadRect( hQueue, hBuffer, blockingRead, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferWriteRect
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferWriteRect = d_context.urDdiTable.Enqueue.pfnMemBufferWriteRect;
        if( nullptr != pfnMemBufferWriteRect )
        {
            result = pfnMemBufferWriteRect( hQueue, hBuffer, blockingWrite, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferCopy
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferCopy = d_context.urDdiTable.Enqueue.pfnMemBufferCopy;
        if( nullptr != pfnMemBufferCopy )
        {
            result = pfnMemBufferCopy( hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset, size, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferCopyRect
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferCopyRect = d_context.urDdiTable.Enqueue.pfnMemBufferCopyRect;
        if( nullptr != pfnMemBufferCopyRect )
        {
            result = pfnMemBufferCopyRect( hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, srcRegion, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferFill
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferFill = d_context.urDdiTable.Enqueue.pfnMemBufferFill;
        if( nullptr != pfnMemBufferFill )
        {
            result = pfnMemBufferFill( hQueue, hBuffer, pPattern, patternSize, offset, size, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemImageRead
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemImageRead = d_context.urDdiTable.Enqueue.pfnMemImageRead;
        if( nullptr != pfnMemImageRead )
        {
            result = pfnMemImageRead( hQueue, hImage, blockingRead, origin, region, rowPitch, slicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemImageWrite
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemImageWrite = d_context.urDdiTable.Enqueue.pfnMemImageWrite;
        if( nullptr != pfnMemImageWrite )
        {
            result = pfnMemImageWrite( hQueue, hImage, blockingWrite, origin, region, inputRowPitch, inputSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemImageCopy
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemImageCopy = d_context.urDdiTable.Enqueue.pfnMemImageCopy;
        if( nullptr != pfnMemImageCopy )
        {
            result = pfnMemImageCopy( hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin, region, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemBufferMap
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemBufferMap = d_context.urDdiTable.Enqueue.pfnMemBufferMap;
        if( nullptr != pfnMemBufferMap )
        {
            result = pfnMemBufferMap( hQueue, hBuffer, blockingMap, mapFlags, offset, size, numEventsInWaitList, phEventWaitList, phEvent, ppRetMap );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueMemUnmap
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnMemUnmap = d_context.urDdiTable.Enqueue.pfnMemUnmap;
        if( nullptr != pfnMemUnmap )
        {
            result = pfnMemUnmap( hQueue, hMem, pMappedPtr, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMMemset
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnUSMMemset = d_context.urDdiTable.Enqueue.pfnUSMMemset;
        if( nullptr != pfnUSMMemset )
        {
            result = pfnUSMMemset( hQueue, ptr, byteValue, count, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMMemcpy
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnUSMMemcpy = d_context.urDdiTable.Enqueue.pfnUSMMemcpy;
        if( nullptr != pfnUSMMemcpy )
        {
            result = pfnUSMMemcpy( hQueue, blocking, pDst, pSrc, size, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMPrefetch
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnUSMPrefetch = d_context.urDdiTable.Enqueue.pfnUSMPrefetch;
        if( nullptr != pfnUSMPrefetch )
        {
            result = pfnUSMPrefetch( hQueue, pMem, size, flags, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMMemAdvise
    __urdlllocal ur_result_t UR_APICALL
    urEnqueueUSMMemAdvise(
        ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
        const void* pMem,                               ///< [in] pointer to the USM memory object
        size_t size,                                    ///< [in] size in bytes to be advised
        ur_mem_advice_t advice,                         ///< [in] USM memory advice
        ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                        ///< particular command instance.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnUSMMemAdvise = d_context.urDdiTable.Enqueue.pfnUSMMemAdvise;
        if( nullptr != pfnUSMMemAdvise )
        {
            result = pfnUSMMemAdvise( hQueue, pMem, size, advice, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMFill2D
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnUSMFill2D = d_context.urDdiTable.Enqueue.pfnUSMFill2D;
        if( nullptr != pfnUSMFill2D )
        {
            result = pfnUSMFill2D( hQueue, pMem, pitch, patternSize, pPattern, width, height, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMMemset2D
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnUSMMemset2D = d_context.urDdiTable.Enqueue.pfnUSMMemset2D;
        if( nullptr != pfnUSMMemset2D )
        {
            result = pfnUSMMemset2D( hQueue, pMem, pitch, value, width, height, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMMemcpy2D
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnUSMMemcpy2D = d_context.urDdiTable.Enqueue.pfnUSMMemcpy2D;
        if( nullptr != pfnUSMMemcpy2D )
        {
            result = pfnUSMMemcpy2D( hQueue, blocking, pDst, dstPitch, pSrc, srcPitch, width, height, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueDeviceGlobalVariableWrite
    __urdlllocal ur_result_t UR_APICALL
    urEnqueueDeviceGlobalVariableWrite(
        ur_queue_handle_t hQueue,                       ///< [in] handle of the queue to submit to.
        ur_program_handle_t hProgram,                   ///< [in] handle of the program containing the device global variable.
        const char* name,                               ///< [in] the unique identifier for the device global variable.
        bool blockingWrite,                             ///< [in] indicates if this operation should block.
        size_t count,                                   ///< [in] the number of bytes to copy.
        size_t offset,                                  ///< [in] the byte offset into the device global variable to start copying.
        const void* pSrc,                               ///< [in] pointer to where the data must be copied from.
        uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list.
        const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                        ///< events that must be complete before the kernel execution.
                                                        ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                                        ///< event. 
        ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                        ///< particular kernel execution instance.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnDeviceGlobalVariableWrite = d_context.urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite;
        if( nullptr != pfnDeviceGlobalVariableWrite )
        {
            result = pfnDeviceGlobalVariableWrite( hQueue, hProgram, name, blockingWrite, count, offset, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueDeviceGlobalVariableRead
    __urdlllocal ur_result_t UR_APICALL
    urEnqueueDeviceGlobalVariableRead(
        ur_queue_handle_t hQueue,                       ///< [in] handle of the queue to submit to.
        ur_program_handle_t hProgram,                   ///< [in] handle of the program containing the device global variable.
        const char* name,                               ///< [in] the unique identifier for the device global variable.
        bool blockingRead,                              ///< [in] indicates if this operation should block.
        size_t count,                                   ///< [in] the number of bytes to copy.
        size_t offset,                                  ///< [in] the byte offset into the device global variable to start copying.
        void* pDst,                                     ///< [in] pointer to where the data must be copied to.
        uint32_t numEventsInWaitList,                   ///< [in] size of the event wait list.
        const ur_event_handle_t* phEventWaitList,       ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                        ///< events that must be complete before the kernel execution.
                                                        ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                                        ///< event. 
        ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                        ///< particular kernel execution instance.    
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnDeviceGlobalVariableRead = d_context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead;
        if( nullptr != pfnDeviceGlobalVariableRead )
        {
            result = pfnDeviceGlobalVariableRead( hQueue, hProgram, name, blockingRead, count, offset, pDst, numEventsInWaitList, phEventWaitList, phEvent );
        }
        else
        {
            // generic implementation
            if( nullptr != phEvent ) *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventGetInfo
    __urdlllocal ur_result_t UR_APICALL
    urEventGetInfo(
        ur_event_handle_t hEvent,                       ///< [in] handle of the event object
        ur_event_info_t propName,                       ///< [in] the name of the event property to query
        size_t propValueSize,                           ///< [in] size in bytes of the event property value
        void* pPropValue,                               ///< [out][optional] value of the event property
        size_t* pPropValueSizeRet                       ///< [out][optional] bytes returned in event property
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Event.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventGetProfilingInfo
    __urdlllocal ur_result_t UR_APICALL
    urEventGetProfilingInfo(
        ur_event_handle_t hEvent,                       ///< [in] handle of the event object
        ur_profiling_info_t propName,                   ///< [in] the name of the profiling property to query
        size_t propValueSize,                           ///< [in] size in bytes of the profiling property value
        void* pPropValue,                               ///< [out][optional] value of the profiling property
        size_t* pPropValueSizeRet                       ///< [out][optional] pointer to the actual size in bytes returned in
                                                        ///< propValue
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetProfilingInfo = d_context.urDdiTable.Event.pfnGetProfilingInfo;
        if( nullptr != pfnGetProfilingInfo )
        {
            result = pfnGetProfilingInfo( hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventWait
    __urdlllocal ur_result_t UR_APICALL
    urEventWait(
        uint32_t numEvents,                             ///< [in] number of events in the event list
        const ur_event_handle_t* phEventWaitList        ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                                                        ///< completion
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnWait = d_context.urDdiTable.Event.pfnWait;
        if( nullptr != pfnWait )
        {
            result = pfnWait( numEvents, phEventWaitList );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventRetain
    __urdlllocal ur_result_t UR_APICALL
    urEventRetain(
        ur_event_handle_t hEvent                        ///< [in] handle of the event object
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Event.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hEvent );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventRelease
    __urdlllocal ur_result_t UR_APICALL
    urEventRelease(
        ur_event_handle_t hEvent                        ///< [in] handle of the event object
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Event.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hEvent );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urEventGetNativeHandle(
        ur_event_handle_t hEvent,                       ///< [in] handle of the event.
        ur_native_handle_t* phNativeEvent               ///< [out] a pointer to the native handle of the event.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Event.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hEvent, phNativeEvent );
        }
        else
        {
            // generic implementation
            *phNativeEvent = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urEventCreateWithNativeHandle(
        ur_native_handle_t hNativeEvent,                ///< [in] the native handle of the event.
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_event_handle_t* phEvent                      ///< [out] pointer to the handle of the event object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Event.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeEvent, hContext, phEvent );
        }
        else
        {
            // generic implementation
            *phEvent = reinterpret_cast<ur_event_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEventSetCallback
    __urdlllocal ur_result_t UR_APICALL
    urEventSetCallback(
        ur_event_handle_t hEvent,                       ///< [in] handle of the event object
        ur_execution_info_t execStatus,                 ///< [in] execution status of the event
        ur_event_callback_t pfnNotify,                  ///< [in] execution status of the event
        void* pUserData                                 ///< [in][out][optional] pointer to data to be passed to callback.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetCallback = d_context.urDdiTable.Event.pfnSetCallback;
        if( nullptr != pfnSetCallback )
        {
            result = pfnSetCallback( hEvent, execStatus, pfnNotify, pUserData );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemImageCreate
    __urdlllocal ur_result_t UR_APICALL
    urMemImageCreate(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_mem_flags_t flags,                           ///< [in] allocation and usage information flags
        const ur_image_format_t* pImageFormat,          ///< [in] pointer to image format specification
        const ur_image_desc_t* pImageDesc,              ///< [in] pointer to image description
        void* pHost,                                    ///< [in] pointer to the buffer data
        ur_mem_handle_t* phMem                          ///< [out] pointer to handle of image object created
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnImageCreate = d_context.urDdiTable.Mem.pfnImageCreate;
        if( nullptr != pfnImageCreate )
        {
            result = pfnImageCreate( hContext, flags, pImageFormat, pImageDesc, pHost, phMem );
        }
        else
        {
            // generic implementation
            *phMem = reinterpret_cast<ur_mem_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemBufferCreate
    __urdlllocal ur_result_t UR_APICALL
    urMemBufferCreate(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_mem_flags_t flags,                           ///< [in] allocation and usage information flags
        size_t size,                                    ///< [in] size in bytes of the memory object to be allocated
        void* pHost,                                    ///< [in][optional] pointer to the buffer data
        ur_mem_handle_t* phBuffer                       ///< [out] pointer to handle of the memory buffer created
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnBufferCreate = d_context.urDdiTable.Mem.pfnBufferCreate;
        if( nullptr != pfnBufferCreate )
        {
            result = pfnBufferCreate( hContext, flags, size, pHost, phBuffer );
        }
        else
        {
            // generic implementation
            *phBuffer = reinterpret_cast<ur_mem_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemRetain
    __urdlllocal ur_result_t UR_APICALL
    urMemRetain(
        ur_mem_handle_t hMem                            ///< [in] handle of the memory object to get access
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Mem.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hMem );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemRelease
    __urdlllocal ur_result_t UR_APICALL
    urMemRelease(
        ur_mem_handle_t hMem                            ///< [in] handle of the memory object to release
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Mem.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hMem );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemBufferPartition
    __urdlllocal ur_result_t UR_APICALL
    urMemBufferPartition(
        ur_mem_handle_t hBuffer,                        ///< [in] handle of the buffer object to allocate from
        ur_mem_flags_t flags,                           ///< [in] allocation and usage information flags
        ur_buffer_create_type_t bufferCreateType,       ///< [in] buffer creation type
        ur_buffer_region_t* pBufferCreateInfo,          ///< [in] pointer to buffer create region information
        ur_mem_handle_t* phMem                          ///< [out] pointer to the handle of sub buffer created
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnBufferPartition = d_context.urDdiTable.Mem.pfnBufferPartition;
        if( nullptr != pfnBufferPartition )
        {
            result = pfnBufferPartition( hBuffer, flags, bufferCreateType, pBufferCreateInfo, phMem );
        }
        else
        {
            // generic implementation
            *phMem = reinterpret_cast<ur_mem_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urMemGetNativeHandle(
        ur_mem_handle_t hMem,                           ///< [in] handle of the mem.
        ur_native_handle_t* phNativeMem                 ///< [out] a pointer to the native handle of the mem.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Mem.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hMem, phNativeMem );
        }
        else
        {
            // generic implementation
            *phNativeMem = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urMemCreateWithNativeHandle(
        ur_native_handle_t hNativeMem,                  ///< [in] the native handle of the mem.
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_mem_handle_t* phMem                          ///< [out] pointer to the handle of the mem object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Mem.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeMem, hContext, phMem );
        }
        else
        {
            // generic implementation
            *phMem = reinterpret_cast<ur_mem_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemGetInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Mem.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hMemory, MemInfoType, propSize, pMemInfo, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemImageGetInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnImageGetInfo = d_context.urDdiTable.Mem.pfnImageGetInfo;
        if( nullptr != pfnImageGetInfo )
        {
            result = pfnImageGetInfo( hMemory, ImgInfoType, propSize, pImgInfo, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urTearDown
    __urdlllocal ur_result_t UR_APICALL
    urTearDown(
        void* pParams                                   ///< [in] pointer to tear down parameters
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnTearDown = d_context.urDdiTable.Global.pfnTearDown;
        if( nullptr != pfnTearDown )
        {
            result = pfnTearDown( pParams );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueGetInfo
    __urdlllocal ur_result_t UR_APICALL
    urQueueGetInfo(
        ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
        ur_queue_info_t propName,                       ///< [in] name of the queue property to query
        size_t propValueSize,                           ///< [in] size in bytes of the queue property value provided
        void* pPropValue,                               ///< [out] value of the queue property
        size_t* pPropSizeRet                            ///< [out] size in bytes returned in queue property value
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Queue.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hQueue, propName, propValueSize, pPropValue, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueCreate
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreate = d_context.urDdiTable.Queue.pfnCreate;
        if( nullptr != pfnCreate )
        {
            result = pfnCreate( hContext, hDevice, pProps, phQueue );
        }
        else
        {
            // generic implementation
            *phQueue = reinterpret_cast<ur_queue_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueRetain
    __urdlllocal ur_result_t UR_APICALL
    urQueueRetain(
        ur_queue_handle_t hQueue                        ///< [in] handle of the queue object to get access
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Queue.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hQueue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueRelease
    __urdlllocal ur_result_t UR_APICALL
    urQueueRelease(
        ur_queue_handle_t hQueue                        ///< [in] handle of the queue object to release
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Queue.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hQueue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urQueueGetNativeHandle(
        ur_queue_handle_t hQueue,                       ///< [in] handle of the queue.
        ur_native_handle_t* phNativeQueue               ///< [out] a pointer to the native handle of the queue.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Queue.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hQueue, phNativeQueue );
        }
        else
        {
            // generic implementation
            *phNativeQueue = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urQueueCreateWithNativeHandle(
        ur_native_handle_t hNativeQueue,                ///< [in] the native handle of the queue.
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_queue_handle_t* phQueue                      ///< [out] pointer to the handle of the queue object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Queue.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeQueue, hContext, phQueue );
        }
        else
        {
            // generic implementation
            *phQueue = reinterpret_cast<ur_queue_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueFinish
    __urdlllocal ur_result_t UR_APICALL
    urQueueFinish(
        ur_queue_handle_t hQueue                        ///< [in] handle of the queue to be finished.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnFinish = d_context.urDdiTable.Queue.pfnFinish;
        if( nullptr != pfnFinish )
        {
            result = pfnFinish( hQueue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urQueueFlush
    __urdlllocal ur_result_t UR_APICALL
    urQueueFlush(
        ur_queue_handle_t hQueue                        ///< [in] handle of the queue to be flushed.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnFlush = d_context.urDdiTable.Queue.pfnFlush;
        if( nullptr != pfnFlush )
        {
            result = pfnFlush( hQueue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urSamplerCreate
    __urdlllocal ur_result_t UR_APICALL
    urSamplerCreate(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        const ur_sampler_property_value_t* pProps,      ///< [in] specifies a list of sampler property names and their
                                                        ///< corresponding values.
        ur_sampler_handle_t* phSampler                  ///< [out] pointer to handle of sampler object created
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreate = d_context.urDdiTable.Sampler.pfnCreate;
        if( nullptr != pfnCreate )
        {
            result = pfnCreate( hContext, pProps, phSampler );
        }
        else
        {
            // generic implementation
            *phSampler = reinterpret_cast<ur_sampler_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urSamplerRetain
    __urdlllocal ur_result_t UR_APICALL
    urSamplerRetain(
        ur_sampler_handle_t hSampler                    ///< [in] handle of the sampler object to get access
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Sampler.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hSampler );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urSamplerRelease
    __urdlllocal ur_result_t UR_APICALL
    urSamplerRelease(
        ur_sampler_handle_t hSampler                    ///< [in] handle of the sampler object to release
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Sampler.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hSampler );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urSamplerGetInfo
    __urdlllocal ur_result_t UR_APICALL
    urSamplerGetInfo(
        ur_sampler_handle_t hSampler,                   ///< [in] handle of the sampler object
        ur_sampler_info_t propName,                     ///< [in] name of the sampler property to query
        size_t propValueSize,                           ///< [in] size in bytes of the sampler property value provided
        void* pPropValue,                               ///< [out] value of the sampler property
        size_t* pPropSizeRet                            ///< [out] size in bytes returned in sampler property value
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Sampler.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hSampler, propName, propValueSize, pPropValue, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urSamplerGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urSamplerGetNativeHandle(
        ur_sampler_handle_t hSampler,                   ///< [in] handle of the sampler.
        ur_native_handle_t* phNativeSampler             ///< [out] a pointer to the native handle of the sampler.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Sampler.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hSampler, phNativeSampler );
        }
        else
        {
            // generic implementation
            *phNativeSampler = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urSamplerCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urSamplerCreateWithNativeHandle(
        ur_native_handle_t hNativeSampler,              ///< [in] the native handle of the sampler.
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_sampler_handle_t* phSampler                  ///< [out] pointer to the handle of the sampler object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Sampler.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeSampler, hContext, phSampler );
        }
        else
        {
            // generic implementation
            *phSampler = reinterpret_cast<ur_sampler_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urUSMHostAlloc
    __urdlllocal ur_result_t UR_APICALL
    urUSMHostAlloc(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_usm_mem_flags_t* pUSMFlag,                   ///< [in] USM memory allocation flags
        size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
        uint32_t align,                                 ///< [in] alignment of the USM memory object
        void** ppMem                                    ///< [out] pointer to USM host memory object
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnHostAlloc = d_context.urDdiTable.USM.pfnHostAlloc;
        if( nullptr != pfnHostAlloc )
        {
            result = pfnHostAlloc( hContext, pUSMFlag, size, align, ppMem );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urUSMDeviceAlloc
    __urdlllocal ur_result_t UR_APICALL
    urUSMDeviceAlloc(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_device_handle_t hDevice,                     ///< [in] handle of the device object
        ur_usm_mem_flags_t* pUSMProp,                   ///< [in] USM memory properties
        size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
        uint32_t align,                                 ///< [in] alignment of the USM memory object
        void** ppMem                                    ///< [out] pointer to USM device memory object
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnDeviceAlloc = d_context.urDdiTable.USM.pfnDeviceAlloc;
        if( nullptr != pfnDeviceAlloc )
        {
            result = pfnDeviceAlloc( hContext, hDevice, pUSMProp, size, align, ppMem );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urUSMSharedAlloc
    __urdlllocal ur_result_t UR_APICALL
    urUSMSharedAlloc(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_device_handle_t hDevice,                     ///< [in] handle of the device object
        ur_usm_mem_flags_t* pUSMProp,                   ///< [in] USM memory properties
        size_t size,                                    ///< [in] size in bytes of the USM memory object to be allocated
        uint32_t align,                                 ///< [in] alignment of the USM memory object
        void** ppMem                                    ///< [out] pointer to USM shared memory object
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSharedAlloc = d_context.urDdiTable.USM.pfnSharedAlloc;
        if( nullptr != pfnSharedAlloc )
        {
            result = pfnSharedAlloc( hContext, hDevice, pUSMProp, size, align, ppMem );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemFree
    __urdlllocal ur_result_t UR_APICALL
    urMemFree(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        void* pMem                                      ///< [in] pointer to USM memory object
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnFree = d_context.urDdiTable.Mem.pfnFree;
        if( nullptr != pfnFree )
        {
            result = pfnFree( hContext, pMem );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urMemGetMemAllocInfo
    __urdlllocal ur_result_t UR_APICALL
    urMemGetMemAllocInfo(
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        const void* pMem,                               ///< [in] pointer to USM memory object
        ur_mem_alloc_info_t propName,                   ///< [in] the name of the USM allocation property to query
        size_t propValueSize,                           ///< [in] size in bytes of the USM allocation property value
        void* pPropValue,                               ///< [out] value of the USM allocation property
        size_t* pPropValueSizeRet                       ///< [out] bytes returned in USM allocation property
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetMemAllocInfo = d_context.urDdiTable.Mem.pfnGetMemAllocInfo;
        if( nullptr != pfnGetMemAllocInfo )
        {
            result = pfnGetMemAllocInfo( hContext, pMem, propName, propValueSize, pPropValue, pPropValueSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceGet
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGet = d_context.urDdiTable.Device.pfnGet;
        if( nullptr != pfnGet )
        {
            result = pfnGet( hPlatform, DeviceType, NumEntries, phDevices, pNumDevices );
        }
        else
        {
            // generic implementation
            for( size_t i = 0; ( nullptr != phDevices ) && ( i < NumEntries ); ++i )
                phDevices[ i ] = reinterpret_cast<ur_device_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceGetInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Device.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hDevice, infoType, propSize, pDeviceInfo, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceRetain
    __urdlllocal ur_result_t UR_APICALL
    urDeviceRetain(
        ur_device_handle_t hDevice                      ///< [in] handle of the device to get a reference of.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Device.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hDevice );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceRelease
    __urdlllocal ur_result_t UR_APICALL
    urDeviceRelease(
        ur_device_handle_t hDevice                      ///< [in] handle of the device to release.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Device.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hDevice );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDevicePartition
    __urdlllocal ur_result_t UR_APICALL
    urDevicePartition(
        ur_device_handle_t hDevice,                     ///< [in] handle of the device to partition.
        const ur_device_partition_property_t* pProperties,  ///< [in] null-terminated array of <$_device_partition_t enum, value> pairs.
        uint32_t NumDevices,                            ///< [in] the number of sub-devices.
        ur_device_handle_t* phSubDevices,               ///< [out][optional][range(0, NumDevices)] array of handle of devices.
                                                        ///< If NumDevices is less than the number of sub-devices available, then
                                                        ///< the function shall only retrieve that number of sub-devices.
        uint32_t* pNumDevicesRet                        ///< [out][optional] pointer to the number of sub-devices the device can be
                                                        ///< partitioned into according to the partitioning property.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnPartition = d_context.urDdiTable.Device.pfnPartition;
        if( nullptr != pfnPartition )
        {
            result = pfnPartition( hDevice, pProperties, NumDevices, phSubDevices, pNumDevicesRet );
        }
        else
        {
            // generic implementation
            for( size_t i = 0; ( nullptr != phSubDevices ) && ( i < NumDevices ); ++i )
                phSubDevices[ i ] = reinterpret_cast<ur_device_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceSelectBinary
    __urdlllocal ur_result_t UR_APICALL
    urDeviceSelectBinary(
        ur_device_handle_t hDevice,                     ///< [in] handle of the device to select binary for.
        const uint8_t** ppBinaries,                     ///< [in] the array of binaries to select from.
        uint32_t NumBinaries,                           ///< [in] the number of binaries passed in ppBinaries. Must greater than or
                                                        ///< equal to zero.
        uint32_t* pSelectedBinary                       ///< [out] the index of the selected binary in the input array of binaries.
                                                        ///< If a suitable binary was not found the function returns ${X}_INVALID_BINARY.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSelectBinary = d_context.urDdiTable.Device.pfnSelectBinary;
        if( nullptr != pfnSelectBinary )
        {
            result = pfnSelectBinary( hDevice, ppBinaries, NumBinaries, pSelectedBinary );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urDeviceGetNativeHandle(
        ur_device_handle_t hDevice,                     ///< [in] handle of the device.
        ur_native_handle_t* phNativeDevice              ///< [out] a pointer to the native handle of the device.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Device.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hDevice, phNativeDevice );
        }
        else
        {
            // generic implementation
            *phNativeDevice = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urDeviceCreateWithNativeHandle(
        ur_native_handle_t hNativeDevice,               ///< [in] the native handle of the device.
        ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform instance
        ur_device_handle_t* phDevice                    ///< [out] pointer to the handle of the device object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Device.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeDevice, hPlatform, phDevice );
        }
        else
        {
            // generic implementation
            *phDevice = reinterpret_cast<ur_device_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceGetGlobalTimestamps
    __urdlllocal ur_result_t UR_APICALL
    urDeviceGetGlobalTimestamps(
        ur_device_handle_t hDevice,                     ///< [in] handle of the device instance
        uint64_t* pDeviceTimestamp,                     ///< [out][optional] pointer to the Device's global timestamp that 
                                                        ///< correlates with the Host's global timestamp value
        uint64_t* pHostTimestamp                        ///< [out][optional] pointer to the Host's global timestamp that 
                                                        ///< correlates with the Device's global timestamp value
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetGlobalTimestamps = d_context.urDdiTable.Device.pfnGetGlobalTimestamps;
        if( nullptr != pfnGetGlobalTimestamps )
        {
            result = pfnGetGlobalTimestamps( hDevice, pDeviceTimestamp, pHostTimestamp );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelCreate
    __urdlllocal ur_result_t UR_APICALL
    urKernelCreate(
        ur_program_handle_t hProgram,                   ///< [in] handle of the program instance
        const char* pKernelName,                        ///< [in] pointer to null-terminated string.
        ur_kernel_handle_t* phKernel                    ///< [out] pointer to handle of kernel object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreate = d_context.urDdiTable.Kernel.pfnCreate;
        if( nullptr != pfnCreate )
        {
            result = pfnCreate( hProgram, pKernelName, phKernel );
        }
        else
        {
            // generic implementation
            *phKernel = reinterpret_cast<ur_kernel_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelSetArgValue
    __urdlllocal ur_result_t UR_APICALL
    urKernelSetArgValue(
        ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
        uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
        size_t argSize,                                 ///< [in] size of argument type
        const void* pArgValue                           ///< [in] argument value represented as matching arg type.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetArgValue = d_context.urDdiTable.Kernel.pfnSetArgValue;
        if( nullptr != pfnSetArgValue )
        {
            result = pfnSetArgValue( hKernel, argIndex, argSize, pArgValue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelSetArgLocal
    __urdlllocal ur_result_t UR_APICALL
    urKernelSetArgLocal(
        ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
        uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
        size_t argSize                                  ///< [in] size of the local buffer to be allocated by the runtime
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetArgLocal = d_context.urDdiTable.Kernel.pfnSetArgLocal;
        if( nullptr != pfnSetArgLocal )
        {
            result = pfnSetArgLocal( hKernel, argIndex, argSize );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelGetInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Kernel.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hKernel, propName, propSize, pKernelInfo, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelGetGroupInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetGroupInfo = d_context.urDdiTable.Kernel.pfnGetGroupInfo;
        if( nullptr != pfnGetGroupInfo )
        {
            result = pfnGetGroupInfo( hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelGetSubGroupInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetSubGroupInfo = d_context.urDdiTable.Kernel.pfnGetSubGroupInfo;
        if( nullptr != pfnGetSubGroupInfo )
        {
            result = pfnGetSubGroupInfo( hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelRetain
    __urdlllocal ur_result_t UR_APICALL
    urKernelRetain(
        ur_kernel_handle_t hKernel                      ///< [in] handle for the Kernel to retain
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Kernel.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hKernel );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelRelease
    __urdlllocal ur_result_t UR_APICALL
    urKernelRelease(
        ur_kernel_handle_t hKernel                      ///< [in] handle for the Kernel to release
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Kernel.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hKernel );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelSetArgPointer
    __urdlllocal ur_result_t UR_APICALL
    urKernelSetArgPointer(
        ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
        uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
        size_t argSize,                                 ///< [in] size of argument type
        const void* pArgValue                           ///< [in][optional] SVM pointer to memory location holding the argument
                                                        ///< value. If null then argument value is considered null.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetArgPointer = d_context.urDdiTable.Kernel.pfnSetArgPointer;
        if( nullptr != pfnSetArgPointer )
        {
            result = pfnSetArgPointer( hKernel, argIndex, argSize, pArgValue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelSetExecInfo
    __urdlllocal ur_result_t UR_APICALL
    urKernelSetExecInfo(
        ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
        ur_kernel_exec_info_t propName,                 ///< [in] name of the execution attribute
        size_t propSize,                                ///< [in] size in byte the attribute value
        const void* pPropValue                          ///< [in][range(0, propSize)] pointer to memory location holding the
                                                        ///< property value.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetExecInfo = d_context.urDdiTable.Kernel.pfnSetExecInfo;
        if( nullptr != pfnSetExecInfo )
        {
            result = pfnSetExecInfo( hKernel, propName, propSize, pPropValue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelSetArgSampler
    __urdlllocal ur_result_t UR_APICALL
    urKernelSetArgSampler(
        ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
        uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
        ur_sampler_handle_t hArgValue                   ///< [in] handle of Sampler object.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetArgSampler = d_context.urDdiTable.Kernel.pfnSetArgSampler;
        if( nullptr != pfnSetArgSampler )
        {
            result = pfnSetArgSampler( hKernel, argIndex, hArgValue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelSetArgMemObj
    __urdlllocal ur_result_t UR_APICALL
    urKernelSetArgMemObj(
        ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
        uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
        ur_mem_handle_t hArgValue                       ///< [in][optional] handle of Memory object.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetArgMemObj = d_context.urDdiTable.Kernel.pfnSetArgMemObj;
        if( nullptr != pfnSetArgMemObj )
        {
            result = pfnSetArgMemObj( hKernel, argIndex, hArgValue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urKernelGetNativeHandle(
        ur_kernel_handle_t hKernel,                     ///< [in] handle of the kernel.
        ur_native_handle_t* phNativeKernel              ///< [out] a pointer to the native handle of the kernel.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Kernel.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hKernel, phNativeKernel );
        }
        else
        {
            // generic implementation
            *phNativeKernel = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urKernelCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urKernelCreateWithNativeHandle(
        ur_native_handle_t hNativeKernel,               ///< [in] the native handle of the kernel.
        ur_context_handle_t hContext,                   ///< [in] handle of the context object
        ur_kernel_handle_t* phKernel                    ///< [out] pointer to the handle of the kernel object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Kernel.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeKernel, hContext, phKernel );
        }
        else
        {
            // generic implementation
            *phKernel = reinterpret_cast<ur_kernel_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urModuleCreate
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreate = d_context.urDdiTable.Module.pfnCreate;
        if( nullptr != pfnCreate )
        {
            result = pfnCreate( hContext, pIL, length, pOptions, pfnNotify, pUserData, phModule );
        }
        else
        {
            // generic implementation
            *phModule = reinterpret_cast<ur_module_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urModuleRetain
    __urdlllocal ur_result_t UR_APICALL
    urModuleRetain(
        ur_module_handle_t hModule                      ///< [in] handle for the Module to retain
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Module.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hModule );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urModuleRelease
    __urdlllocal ur_result_t UR_APICALL
    urModuleRelease(
        ur_module_handle_t hModule                      ///< [in] handle for the Module to release
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Module.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hModule );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urModuleGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urModuleGetNativeHandle(
        ur_module_handle_t hModule,                     ///< [in] handle of the module.
        ur_native_handle_t* phNativeModule              ///< [out] a pointer to the native handle of the module.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Module.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hModule, phNativeModule );
        }
        else
        {
            // generic implementation
            *phNativeModule = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urModuleCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urModuleCreateWithNativeHandle(
        ur_native_handle_t hNativeModule,               ///< [in] the native handle of the module.
        ur_context_handle_t hContext,                   ///< [in] handle of the context instance.
        ur_module_handle_t* phModule                    ///< [out] pointer to the handle of the module object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Module.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeModule, hContext, phModule );
        }
        else
        {
            // generic implementation
            *phModule = reinterpret_cast<ur_module_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urPlatformGet
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGet = d_context.urDdiTable.Platform.pfnGet;
        if( nullptr != pfnGet )
        {
            result = pfnGet( NumEntries, phPlatforms, pNumPlatforms );
        }
        else
        {
            // generic implementation
            for( size_t i = 0; ( nullptr != phPlatforms ) && ( i < NumEntries ); ++i )
                phPlatforms[ i ] = reinterpret_cast<ur_platform_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urPlatformGetInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Platform.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hPlatform, PlatformInfoType, Size, pPlatformInfo, pSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urPlatformGetApiVersion
    __urdlllocal ur_result_t UR_APICALL
    urPlatformGetApiVersion(
        ur_platform_handle_t hDriver,                   ///< [in] handle of the platform
        ur_api_version_t* pVersion                      ///< [out] api version
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetApiVersion = d_context.urDdiTable.Platform.pfnGetApiVersion;
        if( nullptr != pfnGetApiVersion )
        {
            result = pfnGetApiVersion( hDriver, pVersion );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urPlatformGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urPlatformGetNativeHandle(
        ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform.
        ur_native_handle_t* phNativePlatform            ///< [out] a pointer to the native handle of the platform.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Platform.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hPlatform, phNativePlatform );
        }
        else
        {
            // generic implementation
            *phNativePlatform = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urPlatformCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urPlatformCreateWithNativeHandle(
        ur_native_handle_t hNativePlatform,             ///< [in] the native handle of the platform.
        ur_platform_handle_t* phPlatform                ///< [out] pointer to the handle of the platform object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Platform.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativePlatform, phPlatform );
        }
        else
        {
            // generic implementation
            *phPlatform = reinterpret_cast<ur_platform_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urGetLastResult
    __urdlllocal ur_result_t UR_APICALL
    urGetLastResult(
        ur_platform_handle_t hPlatform,                 ///< [in] handle of the platform instance
        const char** ppMessage                          ///< [out] pointer to a string containing adapter specific result in string
                                                        ///< representation.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetLastResult = d_context.urDdiTable.Global.pfnGetLastResult;
        if( nullptr != pfnGetLastResult )
        {
            result = pfnGetLastResult( hPlatform, ppMessage );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramCreate
    __urdlllocal ur_result_t UR_APICALL
    urProgramCreate(
        ur_context_handle_t hContext,                   ///< [in] handle of the context instance
        uint32_t count,                                 ///< [in] number of module handles in module list.
        const ur_module_handle_t* phModules,            ///< [in][range(0, count)] pointer to array of modules.
        const char* pOptions,                           ///< [in][optional] pointer to linker options null-terminated string.
        ur_program_handle_t* phProgram                  ///< [out] pointer to handle of program object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreate = d_context.urDdiTable.Program.pfnCreate;
        if( nullptr != pfnCreate )
        {
            result = pfnCreate( hContext, count, phModules, pOptions, phProgram );
        }
        else
        {
            // generic implementation
            *phProgram = reinterpret_cast<ur_program_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramCreateWithBinary
    __urdlllocal ur_result_t UR_APICALL
    urProgramCreateWithBinary(
        ur_context_handle_t hContext,                   ///< [in] handle of the context instance
        ur_device_handle_t hDevice,                     ///< [in] handle to device associated with binary.
        size_t size,                                    ///< [in] size in bytes.
        const uint8_t* pBinary,                         ///< [in] pointer to binary.
        ur_program_handle_t* phProgram                  ///< [out] pointer to handle of Program object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithBinary = d_context.urDdiTable.Program.pfnCreateWithBinary;
        if( nullptr != pfnCreateWithBinary )
        {
            result = pfnCreateWithBinary( hContext, hDevice, size, pBinary, phProgram );
        }
        else
        {
            // generic implementation
            *phProgram = reinterpret_cast<ur_program_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramRetain
    __urdlllocal ur_result_t UR_APICALL
    urProgramRetain(
        ur_program_handle_t hProgram                    ///< [in] handle for the Program to retain
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRetain = d_context.urDdiTable.Program.pfnRetain;
        if( nullptr != pfnRetain )
        {
            result = pfnRetain( hProgram );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramRelease
    __urdlllocal ur_result_t UR_APICALL
    urProgramRelease(
        ur_program_handle_t hProgram                    ///< [in] handle for the Program to release
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnRelease = d_context.urDdiTable.Program.pfnRelease;
        if( nullptr != pfnRelease )
        {
            result = pfnRelease( hProgram );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramGetFunctionPointer
    __urdlllocal ur_result_t UR_APICALL
    urProgramGetFunctionPointer(
        ur_device_handle_t hDevice,                     ///< [in] handle of the device to retrieve pointer for.
        ur_program_handle_t hProgram,                   ///< [in] handle of the program to search for function in.
                                                        ///< The program must already be built to the specified device, or
                                                        ///< otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
        const char* pFunctionName,                      ///< [in] A null-terminates string denoting the mangled function name.
        void** ppFunctionPointer                        ///< [out] Returns the pointer to the function if it is found in the program.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetFunctionPointer = d_context.urDdiTable.Program.pfnGetFunctionPointer;
        if( nullptr != pfnGetFunctionPointer )
        {
            result = pfnGetFunctionPointer( hDevice, hProgram, pFunctionName, ppFunctionPointer );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramGetInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetInfo = d_context.urDdiTable.Program.pfnGetInfo;
        if( nullptr != pfnGetInfo )
        {
            result = pfnGetInfo( hProgram, propName, propSize, pProgramInfo, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramGetBuildInfo
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetBuildInfo = d_context.urDdiTable.Program.pfnGetBuildInfo;
        if( nullptr != pfnGetBuildInfo )
        {
            result = pfnGetBuildInfo( hProgram, hDevice, propName, propSize, pPropValue, pPropSizeRet );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramSetSpecializationConstant
    __urdlllocal ur_result_t UR_APICALL
    urProgramSetSpecializationConstant(
        ur_program_handle_t hProgram,                   ///< [in] handle of the Program object
        uint32_t specId,                                ///< [in] specification constant Id
        size_t specSize,                                ///< [in] size of the specialization constant value
        const void* pSpecValue                          ///< [in] pointer to the specialization value bytes
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnSetSpecializationConstant = d_context.urDdiTable.Program.pfnSetSpecializationConstant;
        if( nullptr != pfnSetSpecializationConstant )
        {
            result = pfnSetSpecializationConstant( hProgram, specId, specSize, pSpecValue );
        }
        else
        {
            // generic implementation
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramGetNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urProgramGetNativeHandle(
        ur_program_handle_t hProgram,                   ///< [in] handle of the program.
        ur_native_handle_t* phNativeProgram             ///< [out] a pointer to the native handle of the program.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnGetNativeHandle = d_context.urDdiTable.Program.pfnGetNativeHandle;
        if( nullptr != pfnGetNativeHandle )
        {
            result = pfnGetNativeHandle( hProgram, phNativeProgram );
        }
        else
        {
            // generic implementation
            *phNativeProgram = reinterpret_cast<ur_native_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urProgramCreateWithNativeHandle
    __urdlllocal ur_result_t UR_APICALL
    urProgramCreateWithNativeHandle(
        ur_native_handle_t hNativeProgram,              ///< [in] the native handle of the program.
        ur_context_handle_t hContext,                   ///< [in] handle of the context instance
        ur_program_handle_t* phProgram                  ///< [out] pointer to the handle of the program object created.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnCreateWithNativeHandle = d_context.urDdiTable.Program.pfnCreateWithNativeHandle;
        if( nullptr != pfnCreateWithNativeHandle )
        {
            result = pfnCreateWithNativeHandle( hNativeProgram, hContext, phProgram );
        }
        else
        {
            // generic implementation
            *phProgram = reinterpret_cast<ur_program_handle_t>( d_context.get() );

        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urInit
    __urdlllocal ur_result_t UR_APICALL
    urInit(
        ur_platform_init_flags_t platform_flags,        ///< [in] platform initialization flags.
                                                        ///< must be 0 (default) or a combination of ::ur_platform_init_flag_t.
        ur_device_init_flags_t device_flags             ///< [in] device initialization flags.
                                                        ///< must be 0 (default) or a combination of ::ur_device_init_flag_t.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // if the driver has created a custom function, then call it instead of using the generic path
        auto pfnInit = d_context.urDdiTable.Global.pfnInit;
        if( nullptr != pfnInit )
        {
            result = pfnInit( platform_flags, device_flags );
        }
        else
        {
            // generic implementation
        }

        return result;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetGlobalProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_global_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnTearDown                               = driver::urTearDown;

    pDdiTable->pfnGetLastResult                          = driver::urGetLastResult;

    pDdiTable->pfnInit                                   = driver::urInit;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetContextProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_context_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate                                 = driver::urContextCreate;

    pDdiTable->pfnRetain                                 = driver::urContextRetain;

    pDdiTable->pfnRelease                                = driver::urContextRelease;

    pDdiTable->pfnGetInfo                                = driver::urContextGetInfo;

    pDdiTable->pfnGetNativeHandle                        = driver::urContextGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urContextCreateWithNativeHandle;

    pDdiTable->pfnSetExtendedDeleter                     = driver::urContextSetExtendedDeleter;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetEnqueueProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_enqueue_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnKernelLaunch                           = driver::urEnqueueKernelLaunch;

    pDdiTable->pfnEventsWait                             = driver::urEnqueueEventsWait;

    pDdiTable->pfnEventsWaitWithBarrier                  = driver::urEnqueueEventsWaitWithBarrier;

    pDdiTable->pfnMemBufferRead                          = driver::urEnqueueMemBufferRead;

    pDdiTable->pfnMemBufferWrite                         = driver::urEnqueueMemBufferWrite;

    pDdiTable->pfnMemBufferReadRect                      = driver::urEnqueueMemBufferReadRect;

    pDdiTable->pfnMemBufferWriteRect                     = driver::urEnqueueMemBufferWriteRect;

    pDdiTable->pfnMemBufferCopy                          = driver::urEnqueueMemBufferCopy;

    pDdiTable->pfnMemBufferCopyRect                      = driver::urEnqueueMemBufferCopyRect;

    pDdiTable->pfnMemBufferFill                          = driver::urEnqueueMemBufferFill;

    pDdiTable->pfnMemImageRead                           = driver::urEnqueueMemImageRead;

    pDdiTable->pfnMemImageWrite                          = driver::urEnqueueMemImageWrite;

    pDdiTable->pfnMemImageCopy                           = driver::urEnqueueMemImageCopy;

    pDdiTable->pfnMemBufferMap                           = driver::urEnqueueMemBufferMap;

    pDdiTable->pfnMemUnmap                               = driver::urEnqueueMemUnmap;

    pDdiTable->pfnUSMMemset                              = driver::urEnqueueUSMMemset;

    pDdiTable->pfnUSMMemcpy                              = driver::urEnqueueUSMMemcpy;

    pDdiTable->pfnUSMPrefetch                            = driver::urEnqueueUSMPrefetch;

    pDdiTable->pfnUSMMemAdvise                           = driver::urEnqueueUSMMemAdvise;

    pDdiTable->pfnUSMFill2D                              = driver::urEnqueueUSMFill2D;

    pDdiTable->pfnUSMMemset2D                            = driver::urEnqueueUSMMemset2D;

    pDdiTable->pfnUSMMemcpy2D                            = driver::urEnqueueUSMMemcpy2D;

    pDdiTable->pfnDeviceGlobalVariableWrite              = driver::urEnqueueDeviceGlobalVariableWrite;

    pDdiTable->pfnDeviceGlobalVariableRead               = driver::urEnqueueDeviceGlobalVariableRead;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetEventProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_event_dditable_t* pDdiTable                  ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGetInfo                                = driver::urEventGetInfo;

    pDdiTable->pfnGetProfilingInfo                       = driver::urEventGetProfilingInfo;

    pDdiTable->pfnWait                                   = driver::urEventWait;

    pDdiTable->pfnRetain                                 = driver::urEventRetain;

    pDdiTable->pfnRelease                                = driver::urEventRelease;

    pDdiTable->pfnGetNativeHandle                        = driver::urEventGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urEventCreateWithNativeHandle;

    pDdiTable->pfnSetCallback                            = driver::urEventSetCallback;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetKernelProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_kernel_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate                                 = driver::urKernelCreate;

    pDdiTable->pfnGetInfo                                = driver::urKernelGetInfo;

    pDdiTable->pfnGetGroupInfo                           = driver::urKernelGetGroupInfo;

    pDdiTable->pfnGetSubGroupInfo                        = driver::urKernelGetSubGroupInfo;

    pDdiTable->pfnRetain                                 = driver::urKernelRetain;

    pDdiTable->pfnRelease                                = driver::urKernelRelease;

    pDdiTable->pfnGetNativeHandle                        = driver::urKernelGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urKernelCreateWithNativeHandle;

    pDdiTable->pfnSetArgValue                            = driver::urKernelSetArgValue;

    pDdiTable->pfnSetArgLocal                            = driver::urKernelSetArgLocal;

    pDdiTable->pfnSetArgPointer                          = driver::urKernelSetArgPointer;

    pDdiTable->pfnSetExecInfo                            = driver::urKernelSetExecInfo;

    pDdiTable->pfnSetArgSampler                          = driver::urKernelSetArgSampler;

    pDdiTable->pfnSetArgMemObj                           = driver::urKernelSetArgMemObj;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_mem_dditable_t* pDdiTable                    ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnImageCreate                            = driver::urMemImageCreate;

    pDdiTable->pfnBufferCreate                           = driver::urMemBufferCreate;

    pDdiTable->pfnRetain                                 = driver::urMemRetain;

    pDdiTable->pfnRelease                                = driver::urMemRelease;

    pDdiTable->pfnBufferPartition                        = driver::urMemBufferPartition;

    pDdiTable->pfnGetNativeHandle                        = driver::urMemGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urMemCreateWithNativeHandle;

    pDdiTable->pfnGetInfo                                = driver::urMemGetInfo;

    pDdiTable->pfnImageGetInfo                           = driver::urMemImageGetInfo;

    pDdiTable->pfnFree                                   = driver::urMemFree;

    pDdiTable->pfnGetMemAllocInfo                        = driver::urMemGetMemAllocInfo;

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Module table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL
urGetModuleProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_module_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate                                 = driver::urModuleCreate;

    pDdiTable->pfnRetain                                 = driver::urModuleRetain;

    pDdiTable->pfnRelease                                = driver::urModuleRelease;

    pDdiTable->pfnGetNativeHandle                        = driver::urModuleGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urModuleCreateWithNativeHandle;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetPlatformProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_platform_dditable_t* pDdiTable               ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGet                                    = driver::urPlatformGet;

    pDdiTable->pfnGetInfo                                = driver::urPlatformGetInfo;

    pDdiTable->pfnGetNativeHandle                        = driver::urPlatformGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urPlatformCreateWithNativeHandle;

    pDdiTable->pfnGetApiVersion                          = driver::urPlatformGetApiVersion;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetProgramProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_program_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate                                 = driver::urProgramCreate;

    pDdiTable->pfnCreateWithBinary                       = driver::urProgramCreateWithBinary;

    pDdiTable->pfnRetain                                 = driver::urProgramRetain;

    pDdiTable->pfnRelease                                = driver::urProgramRelease;

    pDdiTable->pfnGetFunctionPointer                     = driver::urProgramGetFunctionPointer;

    pDdiTable->pfnGetInfo                                = driver::urProgramGetInfo;

    pDdiTable->pfnGetBuildInfo                           = driver::urProgramGetBuildInfo;

    pDdiTable->pfnSetSpecializationConstant              = driver::urProgramSetSpecializationConstant;

    pDdiTable->pfnGetNativeHandle                        = driver::urProgramGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urProgramCreateWithNativeHandle;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetQueueProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_queue_dditable_t* pDdiTable                  ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGetInfo                                = driver::urQueueGetInfo;

    pDdiTable->pfnCreate                                 = driver::urQueueCreate;

    pDdiTable->pfnRetain                                 = driver::urQueueRetain;

    pDdiTable->pfnRelease                                = driver::urQueueRelease;

    pDdiTable->pfnGetNativeHandle                        = driver::urQueueGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urQueueCreateWithNativeHandle;

    pDdiTable->pfnFinish                                 = driver::urQueueFinish;

    pDdiTable->pfnFlush                                  = driver::urQueueFlush;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetSamplerProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_sampler_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate                                 = driver::urSamplerCreate;

    pDdiTable->pfnRetain                                 = driver::urSamplerRetain;

    pDdiTable->pfnRelease                                = driver::urSamplerRelease;

    pDdiTable->pfnGetInfo                                = driver::urSamplerGetInfo;

    pDdiTable->pfnGetNativeHandle                        = driver::urSamplerGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urSamplerCreateWithNativeHandle;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_usm_dditable_t* pDdiTable                    ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnHostAlloc                              = driver::urUSMHostAlloc;

    pDdiTable->pfnDeviceAlloc                            = driver::urUSMDeviceAlloc;

    pDdiTable->pfnSharedAlloc                            = driver::urUSMSharedAlloc;

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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetDeviceProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_device_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( driver::d_context.version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnGet                                    = driver::urDeviceGet;

    pDdiTable->pfnGetInfo                                = driver::urDeviceGetInfo;

    pDdiTable->pfnRetain                                 = driver::urDeviceRetain;

    pDdiTable->pfnRelease                                = driver::urDeviceRelease;

    pDdiTable->pfnPartition                              = driver::urDevicePartition;

    pDdiTable->pfnSelectBinary                           = driver::urDeviceSelectBinary;

    pDdiTable->pfnGetNativeHandle                        = driver::urDeviceGetNativeHandle;

    pDdiTable->pfnCreateWithNativeHandle                 = driver::urDeviceCreateWithNativeHandle;

    pDdiTable->pfnGetGlobalTimestamps                    = driver::urDeviceGetGlobalTimestamps;

    return result;
}

#if defined(__cplusplus)
};
#endif
