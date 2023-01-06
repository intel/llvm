/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_ldrddi.cpp
 *
 */
#include "ur_loader.h"

namespace loader
{
    ///////////////////////////////////////////////////////////////////////////////
    ur_platform_factory_t               ur_platform_factory;
    ur_device_factory_t                 ur_device_factory;
    ur_context_factory_t                ur_context_factory;
    ur_event_factory_t                  ur_event_factory;
    ur_program_factory_t                ur_program_factory;
    ur_module_factory_t                 ur_module_factory;
    ur_kernel_factory_t                 ur_kernel_factory;
    ur_queue_factory_t                  ur_queue_factory;
    ur_native_factory_t                 ur_native_factory;
    ur_sampler_factory_t                ur_sampler_factory;
    ur_mem_factory_t                    ur_mem_factory;

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( *phDevices )->dditable;
        auto pfnCreate = dditable->ur.Context.pfnCreate;
        if( nullptr == pfnCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handles to platform handles
        auto phDevicesLocal = new ur_device_handle_t [DeviceCount];
        for( size_t i = 0; ( nullptr != phDevices ) && ( i < DeviceCount ); ++i )
            phDevicesLocal[ i ] = reinterpret_cast<ur_device_object_t*>( phDevices[ i ] )->handle;

        // forward to device-platform
        result = pfnCreate( DeviceCount, phDevices, phContext );
        delete []phDevicesLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phContext = reinterpret_cast<ur_context_handle_t>(
                ur_context_factory.getInstance( *phContext, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnRetain = dditable->ur.Context.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnRetain( hContext );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnRelease = dditable->ur.Context.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnRelease( hContext );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnGetInfo = dditable->ur.Context.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnGetInfo( hContext, ContextInfoType, propSize, pContextInfo, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Context.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hContext, phNativeContext );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeContext = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeContext, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeContext )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Context.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeContext = reinterpret_cast<ur_native_object_t*>( hNativeContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeContext, phContext );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phContext = reinterpret_cast<ur_context_handle_t>(
                ur_context_factory.getInstance( *phContext, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnSetExtendedDeleter = dditable->ur.Context.pfnSetExtendedDeleter;
        if( nullptr == pfnSetExtendedDeleter )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnSetExtendedDeleter( hContext, pfnDeleter, pUserData );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnKernelLaunch = dditable->ur.Enqueue.pfnKernelLaunch;
        if( nullptr == pfnKernelLaunch )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnKernelLaunch( hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnEventsWait = dditable->ur.Enqueue.pfnEventsWait;
        if( nullptr == pfnEventsWait )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnEventsWait( hQueue, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnEventsWaitWithBarrier = dditable->ur.Enqueue.pfnEventsWaitWithBarrier;
        if( nullptr == pfnEventsWaitWithBarrier )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnEventsWaitWithBarrier( hQueue, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferRead = dditable->ur.Enqueue.pfnMemBufferRead;
        if( nullptr == pfnMemBufferRead )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBuffer = reinterpret_cast<ur_mem_object_t*>( hBuffer )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferRead( hQueue, hBuffer, blockingRead, offset, size, pDst, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferWrite = dditable->ur.Enqueue.pfnMemBufferWrite;
        if( nullptr == pfnMemBufferWrite )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBuffer = reinterpret_cast<ur_mem_object_t*>( hBuffer )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferWrite( hQueue, hBuffer, blockingWrite, offset, size, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferReadRect = dditable->ur.Enqueue.pfnMemBufferReadRect;
        if( nullptr == pfnMemBufferReadRect )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBuffer = reinterpret_cast<ur_mem_object_t*>( hBuffer )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferReadRect( hQueue, hBuffer, blockingRead, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferWriteRect = dditable->ur.Enqueue.pfnMemBufferWriteRect;
        if( nullptr == pfnMemBufferWriteRect )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBuffer = reinterpret_cast<ur_mem_object_t*>( hBuffer )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferWriteRect( hQueue, hBuffer, blockingWrite, bufferOffset, hostOffset, region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferCopy = dditable->ur.Enqueue.pfnMemBufferCopy;
        if( nullptr == pfnMemBufferCopy )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBufferSrc = reinterpret_cast<ur_mem_object_t*>( hBufferSrc )->handle;

        // convert loader handle to platform handle
        hBufferDst = reinterpret_cast<ur_mem_object_t*>( hBufferDst )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferCopy( hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset, size, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferCopyRect = dditable->ur.Enqueue.pfnMemBufferCopyRect;
        if( nullptr == pfnMemBufferCopyRect )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBufferSrc = reinterpret_cast<ur_mem_object_t*>( hBufferSrc )->handle;

        // convert loader handle to platform handle
        hBufferDst = reinterpret_cast<ur_mem_object_t*>( hBufferDst )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferCopyRect( hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, srcRegion, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferFill = dditable->ur.Enqueue.pfnMemBufferFill;
        if( nullptr == pfnMemBufferFill )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBuffer = reinterpret_cast<ur_mem_object_t*>( hBuffer )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferFill( hQueue, hBuffer, pPattern, patternSize, offset, size, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemImageRead = dditable->ur.Enqueue.pfnMemImageRead;
        if( nullptr == pfnMemImageRead )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hImage = reinterpret_cast<ur_mem_object_t*>( hImage )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemImageRead( hQueue, hImage, blockingRead, origin, region, rowPitch, slicePitch, pDst, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemImageWrite = dditable->ur.Enqueue.pfnMemImageWrite;
        if( nullptr == pfnMemImageWrite )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hImage = reinterpret_cast<ur_mem_object_t*>( hImage )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemImageWrite( hQueue, hImage, blockingWrite, origin, region, inputRowPitch, inputSlicePitch, pSrc, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemImageCopy = dditable->ur.Enqueue.pfnMemImageCopy;
        if( nullptr == pfnMemImageCopy )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hImageSrc = reinterpret_cast<ur_mem_object_t*>( hImageSrc )->handle;

        // convert loader handle to platform handle
        hImageDst = reinterpret_cast<ur_mem_object_t*>( hImageDst )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemImageCopy( hQueue, hImageSrc, hImageDst, srcOrigin, dstOrigin, region, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemBufferMap = dditable->ur.Enqueue.pfnMemBufferMap;
        if( nullptr == pfnMemBufferMap )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hBuffer = reinterpret_cast<ur_mem_object_t*>( hBuffer )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemBufferMap( hQueue, hBuffer, blockingMap, mapFlags, offset, size, numEventsInWaitList, phEventWaitList, phEvent, ppRetMap );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnMemUnmap = dditable->ur.Enqueue.pfnMemUnmap;
        if( nullptr == pfnMemUnmap )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handle to platform handle
        hMem = reinterpret_cast<ur_mem_object_t*>( hMem )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnMemUnmap( hQueue, hMem, pMappedPtr, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnUSMMemset = dditable->ur.Enqueue.pfnUSMMemset;
        if( nullptr == pfnUSMMemset )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnUSMMemset( hQueue, ptr, byteValue, count, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnUSMMemcpy = dditable->ur.Enqueue.pfnUSMMemcpy;
        if( nullptr == pfnUSMMemcpy )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnUSMMemcpy( hQueue, blocking, pDst, pSrc, size, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnUSMPrefetch = dditable->ur.Enqueue.pfnUSMPrefetch;
        if( nullptr == pfnUSMPrefetch )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnUSMPrefetch( hQueue, pMem, size, flags, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urEnqueueUSMMemAdvice
    __urdlllocal ur_result_t UR_APICALL
    urEnqueueUSMMemAdvice(
        ur_queue_handle_t hQueue,                       ///< [in] handle of the queue object
        const void* pMem,                               ///< [in] pointer to the USM memory object
        size_t size,                                    ///< [in] size in bytes to be adviced
        ur_mem_advice_t advice,                         ///< [in] USM memory advice
        ur_event_handle_t* phEvent                      ///< [in,out][optional] return an event object that identifies this
                                                        ///< particular command instance.
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnUSMMemAdvice = dditable->ur.Enqueue.pfnUSMMemAdvice;
        if( nullptr == pfnUSMMemAdvice )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // forward to device-platform
        result = pfnUSMMemAdvice( hQueue, pMem, size, advice, phEvent );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnUSMFill2D = dditable->ur.Enqueue.pfnUSMFill2D;
        if( nullptr == pfnUSMFill2D )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnUSMFill2D( hQueue, pMem, pitch, patternSize, pPattern, width, height, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnUSMMemset2D = dditable->ur.Enqueue.pfnUSMMemset2D;
        if( nullptr == pfnUSMMemset2D )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnUSMMemset2D( hQueue, pMem, pitch, value, width, height, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnUSMMemcpy2D = dditable->ur.Enqueue.pfnUSMMemcpy2D;
        if( nullptr == pfnUSMMemcpy2D )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEventsInWaitList];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEventsInWaitList ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnUSMMemcpy2D( hQueue, blocking, pDst, dstPitch, pSrc, srcPitch, width, height, numEventsInWaitList, phEventWaitList, phEvent );
        delete []phEventWaitListLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            if( nullptr != phEvent )
                *phEvent = reinterpret_cast<ur_event_handle_t>(
                    ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_event_object_t*>( hEvent )->dditable;
        auto pfnGetInfo = dditable->ur.Event.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hEvent = reinterpret_cast<ur_event_object_t*>( hEvent )->handle;

        // forward to device-platform
        result = pfnGetInfo( hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_event_object_t*>( hEvent )->dditable;
        auto pfnGetProfilingInfo = dditable->ur.Event.pfnGetProfilingInfo;
        if( nullptr == pfnGetProfilingInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hEvent = reinterpret_cast<ur_event_object_t*>( hEvent )->handle;

        // forward to device-platform
        result = pfnGetProfilingInfo( hEvent, propName, propValueSize, pPropValue, pPropValueSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_event_object_t*>( *phEventWaitList )->dditable;
        auto pfnWait = dditable->ur.Event.pfnWait;
        if( nullptr == pfnWait )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handles to platform handles
        auto phEventWaitListLocal = new ur_event_handle_t [numEvents];
        for( size_t i = 0; ( nullptr != phEventWaitList ) && ( i < numEvents ); ++i )
            phEventWaitListLocal[ i ] = reinterpret_cast<ur_event_object_t*>( phEventWaitList[ i ] )->handle;

        // forward to device-platform
        result = pfnWait( numEvents, phEventWaitList );
        delete []phEventWaitListLocal;

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_event_object_t*>( hEvent )->dditable;
        auto pfnRetain = dditable->ur.Event.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hEvent = reinterpret_cast<ur_event_object_t*>( hEvent )->handle;

        // forward to device-platform
        result = pfnRetain( hEvent );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_event_object_t*>( hEvent )->dditable;
        auto pfnRelease = dditable->ur.Event.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hEvent = reinterpret_cast<ur_event_object_t*>( hEvent )->handle;

        // forward to device-platform
        result = pfnRelease( hEvent );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_event_object_t*>( hEvent )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Event.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hEvent = reinterpret_cast<ur_event_object_t*>( hEvent )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hEvent, phNativeEvent );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeEvent = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeEvent )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Event.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeEvent = reinterpret_cast<ur_native_object_t*>( hNativeEvent )->handle;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeEvent, hContext, phEvent );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phEvent = reinterpret_cast<ur_event_handle_t>(
                ur_event_factory.getInstance( *phEvent, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_event_object_t*>( hEvent )->dditable;
        auto pfnSetCallback = dditable->ur.Event.pfnSetCallback;
        if( nullptr == pfnSetCallback )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hEvent = reinterpret_cast<ur_event_object_t*>( hEvent )->handle;

        // forward to device-platform
        result = pfnSetCallback( hEvent, execStatus, pfnNotify, pUserData );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnImageCreate = dditable->ur.Mem.pfnImageCreate;
        if( nullptr == pfnImageCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnImageCreate( hContext, flags, pImageFormat, pImageDesc, pHost, phMem );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phMem = reinterpret_cast<ur_mem_handle_t>(
                ur_mem_factory.getInstance( *phMem, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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
        void* pHost,                                    ///< [in] pointer to the buffer data
        ur_mem_handle_t* phBuffer                       ///< [out] pointer to handle of the memory buffer created
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnBufferCreate = dditable->ur.Mem.pfnBufferCreate;
        if( nullptr == pfnBufferCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnBufferCreate( hContext, flags, size, pHost, phBuffer );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phBuffer = reinterpret_cast<ur_mem_handle_t>(
                ur_mem_factory.getInstance( *phBuffer, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_mem_object_t*>( hMem )->dditable;
        auto pfnRetain = dditable->ur.Mem.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hMem = reinterpret_cast<ur_mem_object_t*>( hMem )->handle;

        // forward to device-platform
        result = pfnRetain( hMem );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_mem_object_t*>( hMem )->dditable;
        auto pfnRelease = dditable->ur.Mem.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hMem = reinterpret_cast<ur_mem_object_t*>( hMem )->handle;

        // forward to device-platform
        result = pfnRelease( hMem );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_mem_object_t*>( hBuffer )->dditable;
        auto pfnBufferPartition = dditable->ur.Mem.pfnBufferPartition;
        if( nullptr == pfnBufferPartition )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hBuffer = reinterpret_cast<ur_mem_object_t*>( hBuffer )->handle;

        // forward to device-platform
        result = pfnBufferPartition( hBuffer, flags, bufferCreateType, pBufferCreateInfo, phMem );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phMem = reinterpret_cast<ur_mem_handle_t>(
                ur_mem_factory.getInstance( *phMem, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_mem_object_t*>( hMem )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Mem.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hMem = reinterpret_cast<ur_mem_object_t*>( hMem )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hMem, phNativeMem );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeMem = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeMem, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeMem )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Mem.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeMem = reinterpret_cast<ur_native_object_t*>( hNativeMem )->handle;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeMem, hContext, phMem );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phMem = reinterpret_cast<ur_mem_handle_t>(
                ur_mem_factory.getInstance( *phMem, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_mem_object_t*>( hMemory )->dditable;
        auto pfnGetInfo = dditable->ur.Mem.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hMemory = reinterpret_cast<ur_mem_object_t*>( hMemory )->handle;

        // forward to device-platform
        result = pfnGetInfo( hMemory, MemInfoType, propSize, pMemInfo, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_mem_object_t*>( hMemory )->dditable;
        auto pfnImageGetInfo = dditable->ur.Mem.pfnImageGetInfo;
        if( nullptr == pfnImageGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hMemory = reinterpret_cast<ur_mem_object_t*>( hMemory )->handle;

        // forward to device-platform
        result = pfnImageGetInfo( hMemory, ImgInfoType, propSize, pImgInfo, pPropSizeRet );

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


        for( auto& platform : context->platforms )
        {
            platform.dditable.ur.Global.pfnTearDown( pParams );
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnGetInfo = dditable->ur.Queue.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // forward to device-platform
        result = pfnGetInfo( hQueue, propName, propValueSize, pPropValue, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnCreate = dditable->ur.Queue.pfnCreate;
        if( nullptr == pfnCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnCreate( hContext, hDevice, pProps, phQueue );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phQueue = reinterpret_cast<ur_queue_handle_t>(
                ur_queue_factory.getInstance( *phQueue, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnRetain = dditable->ur.Queue.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // forward to device-platform
        result = pfnRetain( hQueue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnRelease = dditable->ur.Queue.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // forward to device-platform
        result = pfnRelease( hQueue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Queue.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hQueue, phNativeQueue );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeQueue = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeQueue, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeQueue )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Queue.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeQueue = reinterpret_cast<ur_native_object_t*>( hNativeQueue )->handle;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeQueue, hContext, phQueue );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phQueue = reinterpret_cast<ur_queue_handle_t>(
                ur_queue_factory.getInstance( *phQueue, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnFinish = dditable->ur.Queue.pfnFinish;
        if( nullptr == pfnFinish )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // forward to device-platform
        result = pfnFinish( hQueue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_queue_object_t*>( hQueue )->dditable;
        auto pfnFlush = dditable->ur.Queue.pfnFlush;
        if( nullptr == pfnFlush )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hQueue = reinterpret_cast<ur_queue_object_t*>( hQueue )->handle;

        // forward to device-platform
        result = pfnFlush( hQueue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnCreate = dditable->ur.Sampler.pfnCreate;
        if( nullptr == pfnCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreate( hContext, pProps, phSampler );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phSampler = reinterpret_cast<ur_sampler_handle_t>(
                ur_sampler_factory.getInstance( *phSampler, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_sampler_object_t*>( hSampler )->dditable;
        auto pfnRetain = dditable->ur.Sampler.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hSampler = reinterpret_cast<ur_sampler_object_t*>( hSampler )->handle;

        // forward to device-platform
        result = pfnRetain( hSampler );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_sampler_object_t*>( hSampler )->dditable;
        auto pfnRelease = dditable->ur.Sampler.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hSampler = reinterpret_cast<ur_sampler_object_t*>( hSampler )->handle;

        // forward to device-platform
        result = pfnRelease( hSampler );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_sampler_object_t*>( hSampler )->dditable;
        auto pfnGetInfo = dditable->ur.Sampler.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hSampler = reinterpret_cast<ur_sampler_object_t*>( hSampler )->handle;

        // forward to device-platform
        result = pfnGetInfo( hSampler, propName, propValueSize, pPropValue, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_sampler_object_t*>( hSampler )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Sampler.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hSampler = reinterpret_cast<ur_sampler_object_t*>( hSampler )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hSampler, phNativeSampler );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeSampler = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeSampler, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeSampler )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Sampler.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeSampler = reinterpret_cast<ur_native_object_t*>( hNativeSampler )->handle;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeSampler, hContext, phSampler );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phSampler = reinterpret_cast<ur_sampler_handle_t>(
                ur_sampler_factory.getInstance( *phSampler, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnHostAlloc = dditable->ur.USM.pfnHostAlloc;
        if( nullptr == pfnHostAlloc )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnHostAlloc( hContext, pUSMFlag, size, align, ppMem );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnDeviceAlloc = dditable->ur.USM.pfnDeviceAlloc;
        if( nullptr == pfnDeviceAlloc )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnDeviceAlloc( hContext, hDevice, pUSMProp, size, align, ppMem );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnSharedAlloc = dditable->ur.USM.pfnSharedAlloc;
        if( nullptr == pfnSharedAlloc )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnSharedAlloc( hContext, hDevice, pUSMProp, size, align, ppMem );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnFree = dditable->ur.Mem.pfnFree;
        if( nullptr == pfnFree )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnFree( hContext, pMem );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnGetMemAllocInfo = dditable->ur.Mem.pfnGetMemAllocInfo;
        if( nullptr == pfnGetMemAllocInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnGetMemAllocInfo( hContext, pMem, propName, propValueSize, pPropValue, pPropValueSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_platform_object_t*>( hPlatform )->dditable;
        auto pfnGet = dditable->ur.Device.pfnGet;
        if( nullptr == pfnGet )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hPlatform = reinterpret_cast<ur_platform_object_t*>( hPlatform )->handle;

        // forward to device-platform
        result = pfnGet( hPlatform, DeviceType, NumEntries, phDevices, pNumDevices );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handles to loader handles
            for( size_t i = 0; ( nullptr != phDevices ) && ( i < NumEntries ); ++i )
                phDevices[ i ] = reinterpret_cast<ur_device_handle_t>(
                    ur_device_factory.getInstance( phDevices[ i ], dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnGetInfo = dditable->ur.Device.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnGetInfo( hDevice, infoType, propSize, pDeviceInfo, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnRetain = dditable->ur.Device.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnRetain( hDevice );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnRelease = dditable->ur.Device.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnRelease( hDevice );

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDevicePartition
    __urdlllocal ur_result_t UR_APICALL
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
        ur_result_t result = UR_RESULT_SUCCESS;

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnPartition = dditable->ur.Device.pfnPartition;
        if( nullptr == pfnPartition )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnPartition( hDevice, Properties, NumDevices, phSubDevices, pNumDevicesRet );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handles to loader handles
            for( size_t i = 0; ( nullptr != phSubDevices ) && ( i < NumDevices ); ++i )
                phSubDevices[ i ] = reinterpret_cast<ur_device_handle_t>(
                    ur_device_factory.getInstance( phSubDevices[ i ], dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnSelectBinary = dditable->ur.Device.pfnSelectBinary;
        if( nullptr == pfnSelectBinary )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnSelectBinary( hDevice, ppBinaries, NumBinaries, pSelectedBinary );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Device.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hDevice, phNativeDevice );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeDevice = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeDevice, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeDevice )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Device.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeDevice = reinterpret_cast<ur_native_object_t*>( hNativeDevice )->handle;

        // convert loader handle to platform handle
        hPlatform = reinterpret_cast<ur_platform_object_t*>( hPlatform )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeDevice, hPlatform, phDevice );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phDevice = reinterpret_cast<ur_device_handle_t>(
                ur_device_factory.getInstance( *phDevice, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Intercept function for urDeviceGetGlobalTimestamps
    __urdlllocal ur_result_t UR_APICALL
    urDeviceGetGlobalTimestamps(
        ur_device_handle_t hDevice,                     ///< [in] handle of the device instance
        uint64_t* pDeviceTimestamp,                     ///< [out] pointer to the Device's global timestamp that 
                                                        ///< correlates with the Host's global timestamp value
        uint64_t* pHostTimestamp                        ///< [out] pointer to the Host's global timestamp that 
                                                        ///< correlates with the Device's global timestamp value
        )
    {
        ur_result_t result = UR_RESULT_SUCCESS;

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnGetGlobalTimestamps = dditable->ur.Device.pfnGetGlobalTimestamps;
        if( nullptr == pfnGetGlobalTimestamps )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnGetGlobalTimestamps( hDevice, pDeviceTimestamp, pHostTimestamp );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_program_object_t*>( hProgram )->dditable;
        auto pfnCreate = dditable->ur.Kernel.pfnCreate;
        if( nullptr == pfnCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // forward to device-platform
        result = pfnCreate( hProgram, pKernelName, phKernel );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phKernel = reinterpret_cast<ur_kernel_handle_t>(
                ur_kernel_factory.getInstance( *phKernel, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnSetArgValue = dditable->ur.Kernel.pfnSetArgValue;
        if( nullptr == pfnSetArgValue )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnSetArgValue( hKernel, argIndex, argSize, pArgValue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnSetArgLocal = dditable->ur.Kernel.pfnSetArgLocal;
        if( nullptr == pfnSetArgLocal )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnSetArgLocal( hKernel, argIndex, argSize );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnGetInfo = dditable->ur.Kernel.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnGetInfo( hKernel, propName, propSize, pKernelInfo, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnGetGroupInfo = dditable->ur.Kernel.pfnGetGroupInfo;
        if( nullptr == pfnGetGroupInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnGetGroupInfo( hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnGetSubGroupInfo = dditable->ur.Kernel.pfnGetSubGroupInfo;
        if( nullptr == pfnGetSubGroupInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnGetSubGroupInfo( hKernel, hDevice, propName, propSize, pPropValue, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnRetain = dditable->ur.Kernel.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnRetain( hKernel );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnRelease = dditable->ur.Kernel.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnRelease( hKernel );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnSetArgPointer = dditable->ur.Kernel.pfnSetArgPointer;
        if( nullptr == pfnSetArgPointer )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnSetArgPointer( hKernel, argIndex, argSize, pArgValue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnSetExecInfo = dditable->ur.Kernel.pfnSetExecInfo;
        if( nullptr == pfnSetExecInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnSetExecInfo( hKernel, propName, propSize, pPropValue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnSetArgSampler = dditable->ur.Kernel.pfnSetArgSampler;
        if( nullptr == pfnSetArgSampler )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // convert loader handle to platform handle
        hArgValue = reinterpret_cast<ur_sampler_object_t*>( hArgValue )->handle;

        // forward to device-platform
        result = pfnSetArgSampler( hKernel, argIndex, hArgValue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnSetArgMemObj = dditable->ur.Kernel.pfnSetArgMemObj;
        if( nullptr == pfnSetArgMemObj )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // convert loader handle to platform handle
        hArgValue = ( hArgValue ) ? reinterpret_cast<ur_mem_object_t*>( hArgValue )->handle : nullptr;

        // forward to device-platform
        result = pfnSetArgMemObj( hKernel, argIndex, hArgValue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_kernel_object_t*>( hKernel )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Kernel.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hKernel = reinterpret_cast<ur_kernel_object_t*>( hKernel )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hKernel, phNativeKernel );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeKernel = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeKernel, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeKernel )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Kernel.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeKernel = reinterpret_cast<ur_native_object_t*>( hNativeKernel )->handle;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeKernel, hContext, phKernel );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phKernel = reinterpret_cast<ur_kernel_handle_t>(
                ur_kernel_factory.getInstance( *phKernel, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnCreate = dditable->ur.Module.pfnCreate;
        if( nullptr == pfnCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreate( hContext, pIL, length, pOptions, pfnNotify, pUserData, phModule );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phModule = reinterpret_cast<ur_module_handle_t>(
                ur_module_factory.getInstance( *phModule, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_module_object_t*>( hModule )->dditable;
        auto pfnRetain = dditable->ur.Module.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hModule = reinterpret_cast<ur_module_object_t*>( hModule )->handle;

        // forward to device-platform
        result = pfnRetain( hModule );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_module_object_t*>( hModule )->dditable;
        auto pfnRelease = dditable->ur.Module.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hModule = reinterpret_cast<ur_module_object_t*>( hModule )->handle;

        // forward to device-platform
        result = pfnRelease( hModule );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_module_object_t*>( hModule )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Module.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hModule = reinterpret_cast<ur_module_object_t*>( hModule )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hModule, phNativeModule );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeModule = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeModule, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeModule )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Module.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeModule = reinterpret_cast<ur_native_object_t*>( hNativeModule )->handle;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeModule, hContext, phModule );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phModule = reinterpret_cast<ur_module_handle_t>(
                ur_module_factory.getInstance( *phModule, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        uint32_t total_platform_handle_count = 0;

        for( auto& platform : context->platforms )
        {
            if(platform.initStatus != UR_RESULT_SUCCESS)
                continue;

            if( ( 0 < NumEntries ) && ( NumEntries == total_platform_handle_count))
                break;

            uint32_t library_platform_handle_count = 0;

            result = platform.dditable.ur.Platform.pfnGet( 0, nullptr, &library_platform_handle_count );
            if( UR_RESULT_SUCCESS != result ) break;

            if( nullptr != phPlatforms && NumEntries !=0)
            {
                if( total_platform_handle_count + library_platform_handle_count > NumEntries) {
                    library_platform_handle_count = NumEntries - total_platform_handle_count;
                }
                result = platform.dditable.ur.Platform.pfnGet( library_platform_handle_count, &phPlatforms[ total_platform_handle_count ], nullptr );
                if( UR_RESULT_SUCCESS != result ) break;

                try
                {
                    for( uint32_t i = 0; i < library_platform_handle_count; ++i ) {
                        uint32_t platform_index = total_platform_handle_count + i;
                        phPlatforms[ platform_index ] = reinterpret_cast<ur_platform_handle_t>(
                            ur_platform_factory.getInstance( phPlatforms[ platform_index ], &platform.dditable ) );
                    }
                }
                catch( std::bad_alloc& )
                {
                    result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
                }
            }

            total_platform_handle_count += library_platform_handle_count;
        }

        if( UR_RESULT_SUCCESS == result && pNumPlatforms != nullptr )
            *pNumPlatforms = total_platform_handle_count;

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_platform_object_t*>( hPlatform )->dditable;
        auto pfnGetInfo = dditable->ur.Platform.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hPlatform = reinterpret_cast<ur_platform_object_t*>( hPlatform )->handle;

        // forward to device-platform
        result = pfnGetInfo( hPlatform, PlatformInfoType, Size, pPlatformInfo, pSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_platform_object_t*>( hDriver )->dditable;
        auto pfnGetApiVersion = dditable->ur.Platform.pfnGetApiVersion;
        if( nullptr == pfnGetApiVersion )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDriver = reinterpret_cast<ur_platform_object_t*>( hDriver )->handle;

        // forward to device-platform
        result = pfnGetApiVersion( hDriver, pVersion );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_platform_object_t*>( hPlatform )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Platform.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hPlatform = reinterpret_cast<ur_platform_object_t*>( hPlatform )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hPlatform, phNativePlatform );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativePlatform = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativePlatform, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativePlatform )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Platform.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativePlatform = reinterpret_cast<ur_native_object_t*>( hNativePlatform )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativePlatform, phPlatform );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phPlatform = reinterpret_cast<ur_platform_handle_t>(
                ur_platform_factory.getInstance( *phPlatform, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_platform_object_t*>( hPlatform )->dditable;
        auto pfnGetLastResult = dditable->ur.Global.pfnGetLastResult;
        if( nullptr == pfnGetLastResult )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hPlatform = reinterpret_cast<ur_platform_object_t*>( hPlatform )->handle;

        // forward to device-platform
        result = pfnGetLastResult( hPlatform, ppMessage );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnCreate = dditable->ur.Program.pfnCreate;
        if( nullptr == pfnCreate )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // convert loader handles to platform handles
        auto phModulesLocal = new ur_module_handle_t [count];
        for( size_t i = 0; ( nullptr != phModules ) && ( i < count ); ++i )
            phModulesLocal[ i ] = reinterpret_cast<ur_module_object_t*>( phModules[ i ] )->handle;

        // forward to device-platform
        result = pfnCreate( hContext, count, phModules, pOptions, phProgram );
        delete []phModulesLocal;

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phProgram = reinterpret_cast<ur_program_handle_t>(
                ur_program_factory.getInstance( *phProgram, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_context_object_t*>( hContext )->dditable;
        auto pfnCreateWithBinary = dditable->ur.Program.pfnCreateWithBinary;
        if( nullptr == pfnCreateWithBinary )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnCreateWithBinary( hContext, hDevice, size, pBinary, phProgram );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phProgram = reinterpret_cast<ur_program_handle_t>(
                ur_program_factory.getInstance( *phProgram, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_program_object_t*>( hProgram )->dditable;
        auto pfnRetain = dditable->ur.Program.pfnRetain;
        if( nullptr == pfnRetain )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // forward to device-platform
        result = pfnRetain( hProgram );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_program_object_t*>( hProgram )->dditable;
        auto pfnRelease = dditable->ur.Program.pfnRelease;
        if( nullptr == pfnRelease )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // forward to device-platform
        result = pfnRelease( hProgram );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_device_object_t*>( hDevice )->dditable;
        auto pfnGetFunctionPointer = dditable->ur.Program.pfnGetFunctionPointer;
        if( nullptr == pfnGetFunctionPointer )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // forward to device-platform
        result = pfnGetFunctionPointer( hDevice, hProgram, pFunctionName, ppFunctionPointer );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_program_object_t*>( hProgram )->dditable;
        auto pfnGetInfo = dditable->ur.Program.pfnGetInfo;
        if( nullptr == pfnGetInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // forward to device-platform
        result = pfnGetInfo( hProgram, propName, propSize, pProgramInfo, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_program_object_t*>( hProgram )->dditable;
        auto pfnGetBuildInfo = dditable->ur.Program.pfnGetBuildInfo;
        if( nullptr == pfnGetBuildInfo )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // convert loader handle to platform handle
        hDevice = reinterpret_cast<ur_device_object_t*>( hDevice )->handle;

        // forward to device-platform
        result = pfnGetBuildInfo( hProgram, hDevice, propName, propSize, pPropValue, pPropSizeRet );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_program_object_t*>( hProgram )->dditable;
        auto pfnSetSpecializationConstant = dditable->ur.Program.pfnSetSpecializationConstant;
        if( nullptr == pfnSetSpecializationConstant )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // forward to device-platform
        result = pfnSetSpecializationConstant( hProgram, specId, specSize, pSpecValue );

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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_program_object_t*>( hProgram )->dditable;
        auto pfnGetNativeHandle = dditable->ur.Program.pfnGetNativeHandle;
        if( nullptr == pfnGetNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hProgram = reinterpret_cast<ur_program_object_t*>( hProgram )->handle;

        // forward to device-platform
        result = pfnGetNativeHandle( hProgram, phNativeProgram );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phNativeProgram = reinterpret_cast<ur_native_handle_t>(
                ur_native_factory.getInstance( *phNativeProgram, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        // extract platform's function pointer table
        auto dditable = reinterpret_cast<ur_native_object_t*>( hNativeProgram )->dditable;
        auto pfnCreateWithNativeHandle = dditable->ur.Program.pfnCreateWithNativeHandle;
        if( nullptr == pfnCreateWithNativeHandle )
            return UR_RESULT_ERROR_UNINITIALIZED;

        // convert loader handle to platform handle
        hNativeProgram = reinterpret_cast<ur_native_object_t*>( hNativeProgram )->handle;

        // convert loader handle to platform handle
        hContext = reinterpret_cast<ur_context_object_t*>( hContext )->handle;

        // forward to device-platform
        result = pfnCreateWithNativeHandle( hNativeProgram, hContext, phProgram );

        if( UR_RESULT_SUCCESS != result )
            return result;

        try
        {
            // convert platform handle to loader handle
            *phProgram = reinterpret_cast<ur_program_handle_t>(
                ur_program_factory.getInstance( *phProgram, dditable ) );
        }
        catch( std::bad_alloc& )
        {
            result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
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

        bool atLeastOneplatformValid = false;
        for( auto& platform : context->platforms )
        {
            if(platform.initStatus != UR_RESULT_SUCCESS)
                continue;
            platform.initStatus = platform.dditable.ur.Global.pfnInit( platform_flags, device_flags );
            if(platform.initStatus == UR_RESULT_SUCCESS)
                atLeastOneplatformValid = true;
        }

        if(!atLeastOneplatformValid)
            result=UR_RESULT_ERROR_UNINITIALIZED;

        return result;
    }

} // namespace loader

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL
urGetGlobalProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_global_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetGlobalProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetGlobalProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Global);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnTearDown                                 = loader::urTearDown;
            pDdiTable->pfnGetLastResult                            = loader::urGetLastResult;
            pDdiTable->pfnInit                                     = loader::urInit;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Global;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetContextProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_context_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetContextProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetContextProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Context);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate                                   = loader::urContextCreate;
            pDdiTable->pfnRetain                                   = loader::urContextRetain;
            pDdiTable->pfnRelease                                  = loader::urContextRelease;
            pDdiTable->pfnGetInfo                                  = loader::urContextGetInfo;
            pDdiTable->pfnGetNativeHandle                          = loader::urContextGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urContextCreateWithNativeHandle;
            pDdiTable->pfnSetExtendedDeleter                       = loader::urContextSetExtendedDeleter;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Context;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetEnqueueProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_enqueue_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetEnqueueProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetEnqueueProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Enqueue);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnKernelLaunch                             = loader::urEnqueueKernelLaunch;
            pDdiTable->pfnEventsWait                               = loader::urEnqueueEventsWait;
            pDdiTable->pfnEventsWaitWithBarrier                    = loader::urEnqueueEventsWaitWithBarrier;
            pDdiTable->pfnMemBufferRead                            = loader::urEnqueueMemBufferRead;
            pDdiTable->pfnMemBufferWrite                           = loader::urEnqueueMemBufferWrite;
            pDdiTable->pfnMemBufferReadRect                        = loader::urEnqueueMemBufferReadRect;
            pDdiTable->pfnMemBufferWriteRect                       = loader::urEnqueueMemBufferWriteRect;
            pDdiTable->pfnMemBufferCopy                            = loader::urEnqueueMemBufferCopy;
            pDdiTable->pfnMemBufferCopyRect                        = loader::urEnqueueMemBufferCopyRect;
            pDdiTable->pfnMemBufferFill                            = loader::urEnqueueMemBufferFill;
            pDdiTable->pfnMemImageRead                             = loader::urEnqueueMemImageRead;
            pDdiTable->pfnMemImageWrite                            = loader::urEnqueueMemImageWrite;
            pDdiTable->pfnMemImageCopy                             = loader::urEnqueueMemImageCopy;
            pDdiTable->pfnMemBufferMap                             = loader::urEnqueueMemBufferMap;
            pDdiTable->pfnMemUnmap                                 = loader::urEnqueueMemUnmap;
            pDdiTable->pfnUSMMemset                                = loader::urEnqueueUSMMemset;
            pDdiTable->pfnUSMMemcpy                                = loader::urEnqueueUSMMemcpy;
            pDdiTable->pfnUSMPrefetch                              = loader::urEnqueueUSMPrefetch;
            pDdiTable->pfnUSMMemAdvice                             = loader::urEnqueueUSMMemAdvice;
            pDdiTable->pfnUSMFill2D                                = loader::urEnqueueUSMFill2D;
            pDdiTable->pfnUSMMemset2D                              = loader::urEnqueueUSMMemset2D;
            pDdiTable->pfnUSMMemcpy2D                              = loader::urEnqueueUSMMemcpy2D;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Enqueue;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetEventProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_event_dditable_t* pDdiTable                  ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetEventProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetEventProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Event);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnGetInfo                                  = loader::urEventGetInfo;
            pDdiTable->pfnGetProfilingInfo                         = loader::urEventGetProfilingInfo;
            pDdiTable->pfnWait                                     = loader::urEventWait;
            pDdiTable->pfnRetain                                   = loader::urEventRetain;
            pDdiTable->pfnRelease                                  = loader::urEventRelease;
            pDdiTable->pfnGetNativeHandle                          = loader::urEventGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urEventCreateWithNativeHandle;
            pDdiTable->pfnSetCallback                              = loader::urEventSetCallback;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Event;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetKernelProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_kernel_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetKernelProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetKernelProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Kernel);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate                                   = loader::urKernelCreate;
            pDdiTable->pfnGetInfo                                  = loader::urKernelGetInfo;
            pDdiTable->pfnGetGroupInfo                             = loader::urKernelGetGroupInfo;
            pDdiTable->pfnGetSubGroupInfo                          = loader::urKernelGetSubGroupInfo;
            pDdiTable->pfnRetain                                   = loader::urKernelRetain;
            pDdiTable->pfnRelease                                  = loader::urKernelRelease;
            pDdiTable->pfnGetNativeHandle                          = loader::urKernelGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urKernelCreateWithNativeHandle;
            pDdiTable->pfnSetArgValue                              = loader::urKernelSetArgValue;
            pDdiTable->pfnSetArgLocal                              = loader::urKernelSetArgLocal;
            pDdiTable->pfnSetArgPointer                            = loader::urKernelSetArgPointer;
            pDdiTable->pfnSetExecInfo                              = loader::urKernelSetExecInfo;
            pDdiTable->pfnSetArgSampler                            = loader::urKernelSetArgSampler;
            pDdiTable->pfnSetArgMemObj                             = loader::urKernelSetArgMemObj;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Kernel;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetMemProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_mem_dditable_t* pDdiTable                    ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetMemProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetMemProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Mem);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnImageCreate                              = loader::urMemImageCreate;
            pDdiTable->pfnBufferCreate                             = loader::urMemBufferCreate;
            pDdiTable->pfnRetain                                   = loader::urMemRetain;
            pDdiTable->pfnRelease                                  = loader::urMemRelease;
            pDdiTable->pfnBufferPartition                          = loader::urMemBufferPartition;
            pDdiTable->pfnGetNativeHandle                          = loader::urMemGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urMemCreateWithNativeHandle;
            pDdiTable->pfnGetInfo                                  = loader::urMemGetInfo;
            pDdiTable->pfnImageGetInfo                             = loader::urMemImageGetInfo;
            pDdiTable->pfnFree                                     = loader::urMemFree;
            pDdiTable->pfnGetMemAllocInfo                          = loader::urMemGetMemAllocInfo;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Mem;
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Module table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
UR_DLLEXPORT ur_result_t UR_APICALL
urGetModuleProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_module_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetModuleProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetModuleProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Module);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate                                   = loader::urModuleCreate;
            pDdiTable->pfnRetain                                   = loader::urModuleRetain;
            pDdiTable->pfnRelease                                  = loader::urModuleRelease;
            pDdiTable->pfnGetNativeHandle                          = loader::urModuleGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urModuleCreateWithNativeHandle;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Module;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetPlatformProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_platform_dditable_t* pDdiTable               ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetPlatformProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetPlatformProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Platform);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnGet                                      = loader::urPlatformGet;
            pDdiTable->pfnGetInfo                                  = loader::urPlatformGetInfo;
            pDdiTable->pfnGetNativeHandle                          = loader::urPlatformGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urPlatformCreateWithNativeHandle;
            pDdiTable->pfnGetApiVersion                            = loader::urPlatformGetApiVersion;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Platform;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetProgramProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_program_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetProgramProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetProgramProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Program);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate                                   = loader::urProgramCreate;
            pDdiTable->pfnCreateWithBinary                         = loader::urProgramCreateWithBinary;
            pDdiTable->pfnRetain                                   = loader::urProgramRetain;
            pDdiTable->pfnRelease                                  = loader::urProgramRelease;
            pDdiTable->pfnGetFunctionPointer                       = loader::urProgramGetFunctionPointer;
            pDdiTable->pfnGetInfo                                  = loader::urProgramGetInfo;
            pDdiTable->pfnGetBuildInfo                             = loader::urProgramGetBuildInfo;
            pDdiTable->pfnSetSpecializationConstant                = loader::urProgramSetSpecializationConstant;
            pDdiTable->pfnGetNativeHandle                          = loader::urProgramGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urProgramCreateWithNativeHandle;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Program;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetQueueProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_queue_dditable_t* pDdiTable                  ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetQueueProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetQueueProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Queue);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnGetInfo                                  = loader::urQueueGetInfo;
            pDdiTable->pfnCreate                                   = loader::urQueueCreate;
            pDdiTable->pfnRetain                                   = loader::urQueueRetain;
            pDdiTable->pfnRelease                                  = loader::urQueueRelease;
            pDdiTable->pfnGetNativeHandle                          = loader::urQueueGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urQueueCreateWithNativeHandle;
            pDdiTable->pfnFinish                                   = loader::urQueueFinish;
            pDdiTable->pfnFlush                                    = loader::urQueueFlush;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Queue;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetSamplerProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_sampler_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetSamplerProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetSamplerProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Sampler);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnCreate                                   = loader::urSamplerCreate;
            pDdiTable->pfnRetain                                   = loader::urSamplerRetain;
            pDdiTable->pfnRelease                                  = loader::urSamplerRelease;
            pDdiTable->pfnGetInfo                                  = loader::urSamplerGetInfo;
            pDdiTable->pfnGetNativeHandle                          = loader::urSamplerGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urSamplerCreateWithNativeHandle;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Sampler;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetUSMProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_usm_dditable_t* pDdiTable                    ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetUSMProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetUSMProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.USM);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnHostAlloc                                = loader::urUSMHostAlloc;
            pDdiTable->pfnDeviceAlloc                              = loader::urUSMDeviceAlloc;
            pDdiTable->pfnSharedAlloc                              = loader::urUSMSharedAlloc;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.USM;
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
UR_DLLEXPORT ur_result_t UR_APICALL
urGetDeviceProcAddrTable(
    ur_api_version_t version,                       ///< [in] API version requested
    ur_device_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    )
{
    if( loader::context->platforms.size() < 1 )
        return UR_RESULT_ERROR_UNINITIALIZED;

    if( nullptr == pDdiTable )
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;

    if( loader::context->version < version )
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;

    ur_result_t result = UR_RESULT_SUCCESS;

    bool atLeastOneplatformValid = false;
    // Load the device-platform DDI tables
    for( auto& platform : loader::context->platforms )
    {
        if(platform.initStatus != UR_RESULT_SUCCESS)
            continue;
        auto getTable = reinterpret_cast<ur_pfnGetDeviceProcAddrTable_t>(
            GET_FUNCTION_PTR( platform.handle, "urGetDeviceProcAddrTable") );
        if(!getTable) 
            continue; 
        auto getTableResult = getTable( version, &platform.dditable.ur.Device);
        if(getTableResult == UR_RESULT_SUCCESS) 
            atLeastOneplatformValid = true;
        else
            platform.initStatus = getTableResult;
    }

    if(!atLeastOneplatformValid)
        result = UR_RESULT_ERROR_UNINITIALIZED;
    else
        result = UR_RESULT_SUCCESS;

    if( UR_RESULT_SUCCESS == result )
    {
        if( ( loader::context->platforms.size() > 1 ) || loader::context->forceIntercept )
        {
            // return pointers to loader's DDIs
            pDdiTable->pfnGet                                      = loader::urDeviceGet;
            pDdiTable->pfnGetInfo                                  = loader::urDeviceGetInfo;
            pDdiTable->pfnRetain                                   = loader::urDeviceRetain;
            pDdiTable->pfnRelease                                  = loader::urDeviceRelease;
            pDdiTable->pfnPartition                                = loader::urDevicePartition;
            pDdiTable->pfnSelectBinary                             = loader::urDeviceSelectBinary;
            pDdiTable->pfnGetNativeHandle                          = loader::urDeviceGetNativeHandle;
            pDdiTable->pfnCreateWithNativeHandle                   = loader::urDeviceCreateWithNativeHandle;
            pDdiTable->pfnGetGlobalTimestamps                      = loader::urDeviceGetGlobalTimestamps;
        }
        else
        {
            // return pointers directly to platform's DDIs
            *pDdiTable = loader::context->platforms.front().dditable.ur.Device;
        }
    }

    return result;
}


#if defined(__cplusplus)
};
#endif
