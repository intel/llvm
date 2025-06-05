/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_ddi.cpp
 *
 */

#include "tsan_ddi.hpp"
#include "sanitizer_common/sanitizer_common.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "tsan_interceptor.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace tsan {

namespace {

ur_result_t setupContext(ur_context_handle_t Context, uint32_t numDevices,
                         const ur_device_handle_t *phDevices) {
  std::shared_ptr<ContextInfo> CI;
  UR_CALL(getTsanInterceptor()->insertContext(Context, CI));
  for (uint32_t i = 0; i < numDevices; i++) {
    std::shared_ptr<DeviceInfo> DI;
    UR_CALL(getTsanInterceptor()->insertDevice(phDevices[i], DI));
    DI->Type = GetDeviceType(Context, DI->Handle);
    if (DI->Type == DeviceType::UNKNOWN) {
      UR_LOG_L(getContext()->logger, ERR, "Unsupport device");
      return UR_RESULT_ERROR_INVALID_DEVICE;
    }
    if (!DI->Shadow)
      UR_CALL(DI->allocShadowMemory());
    CI->DeviceList.emplace_back(DI->Handle);
  }
  return UR_RESULT_SUCCESS;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreate
__urdlllocal ur_result_t UR_APICALL urContextCreate(
    /// [in] the number of devices given in phDevices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] array of handle of devices.
    const ur_device_handle_t *phDevices,
    /// [in][optional] pointer to context creation properties.
    const ur_context_properties_t *pProperties,
    /// [out] pointer to handle of context object created
    ur_context_handle_t *phContext) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urContextCreate");

  UR_CALL(getContext()->urDdiTable.Context.pfnCreate(numDevices, phDevices,
                                                     pProperties, phContext));

  UR_CALL(setupContext(*phContext, numDevices, phDevices));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    /// [in][nocheck] the native handle of the getContext()->
    ur_native_handle_t hNativeContext, ur_adapter_handle_t hAdapter,
    /// [in] number of devices associated with the context
    uint32_t numDevices,
    /// [in][range(0, numDevices)] list of devices associated with the
    /// context
    const ur_device_handle_t *phDevices,
    /// [in][optional] pointer to native context properties struct
    const ur_context_native_properties_t *pProperties,
    /// [out] pointer to the handle of the context object created.
    ur_context_handle_t *phContext) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urContextCreateWithNativeHandle");

  UR_CALL(getContext()->urDdiTable.Context.pfnCreateWithNativeHandle(
      hNativeContext, hAdapter, numDevices, phDevices, pProperties, phContext));

  UR_CALL(setupContext(*phContext, numDevices, phDevices));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
ur_result_t urContextRetain(

    /// [in] handle of the context to get a reference of.
    ur_context_handle_t hContext) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urContextRetain");

  UR_CALL(getContext()->urDdiTable.Context.pfnRetain(hContext));

  auto ContextInfo = getTsanInterceptor()->getContextInfo(hContext);
  if (!ContextInfo) {
    UR_LOG_L(getContext()->logger, ERR, "Invalid context");
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }
  ContextInfo->RefCount++;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
ur_result_t urContextRelease(
    /// [in] handle of the context to release.
    ur_context_handle_t hContext) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urContextRelease");

  UR_CALL(getContext()->urDdiTable.Context.pfnRelease(hContext));

  auto ContextInfo = getTsanInterceptor()->getContextInfo(hContext);
  if (!ContextInfo) {
    UR_LOG_L(getContext()->logger, ERR, "Invalid context");
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  if (--ContextInfo->RefCount == 0) {
    UR_CALL(getTsanInterceptor()->eraseContext(hContext));
  }

  return UR_RESULT_SUCCESS;
}

/// @brief Intercept function for urProgramBuild
ur_result_t urProgramBuild(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] handle of the program object
    ur_program_handle_t hProgram,
    /// [in] string of build options
    const char *pOptions) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urProgramBuild");

  auto UrRes =
      getContext()->urDdiTable.Program.pfnBuild(hContext, hProgram, pOptions);
  if (UrRes != UR_RESULT_SUCCESS) {
    auto Devices = GetDevices(hContext);
    PrintUrBuildLogIfError(UrRes, hProgram, Devices.data(), Devices.size());
    return UrRes;
  }

  UR_CALL(getTsanInterceptor()->registerProgram(hProgram));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLink
ur_result_t urProgramLink(
    /// [in] handle of the context instance.
    ur_context_handle_t hContext,
    /// [in] number of program handles in `phPrograms`.
    uint32_t count,
    /// [in][range(0, count)] pointer to array of program handles.
    const ur_program_handle_t *phPrograms,
    /// [in][optional] pointer to linker options null-terminated string.
    const char *pOptions,
    /// [out] pointer to handle of program object created.
    ur_program_handle_t *phProgram) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urProgramLink");

  auto UrRes = getContext()->urDdiTable.Program.pfnLink(
      hContext, count, phPrograms, pOptions, phProgram);
  if (UrRes != UR_RESULT_SUCCESS) {
    auto Devices = GetDevices(hContext);
    PrintUrBuildLogIfError(UrRes, *phProgram, Devices.data(), Devices.size());
    return UrRes;
  }
  UR_CALL(getTsanInterceptor()->registerProgram(*phProgram));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuildExp
ur_result_t urProgramBuildExp(
    /// [in] Handle of the program to build.
    ur_program_handle_t hProgram,
    /// [in] number of devices
    uint32_t numDevices,
    /// [in][range(0, numDevices)] pointer to array of device handles
    ur_device_handle_t *phDevices,
    /// [in][optional] pointer to build options null-terminated string.
    const char *pOptions) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urProgramBuildExp");

  auto UrRes = getContext()->urDdiTable.ProgramExp.pfnBuildExp(
      hProgram, numDevices, phDevices, pOptions);
  if (UrRes != UR_RESULT_SUCCESS) {
    PrintUrBuildLogIfError(UrRes, hProgram, phDevices, numDevices);
    return UrRes;
  }
  UR_CALL(getTsanInterceptor()->registerProgram(hProgram));

  return UR_RESULT_SUCCESS;
}

/// @brief Intercept function for urProgramLinkExp
ur_result_t urProgramLinkExp(
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
    /// [out] pointer to handle of program object created.
    ur_program_handle_t *phProgram) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urProgramLinkExp");

  auto UrRes = getContext()->urDdiTable.ProgramExp.pfnLinkExp(
      hContext, numDevices, phDevices, count, phPrograms, pOptions, phProgram);
  if (UrRes != UR_RESULT_SUCCESS) {
    PrintUrBuildLogIfError(UrRes, *phProgram, phDevices, numDevices);
    return UrRes;
  }

  UR_CALL(getTsanInterceptor()->registerProgram(*phProgram));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreate
ur_result_t urMemBufferCreate(
    /// [in] handle of the context object
    ur_context_handle_t hContext,
    /// [in] allocation and usage information flags
    ur_mem_flags_t flags,
    /// [in] size in bytes of the memory object to be allocated
    size_t size,
    /// [in][optional] pointer to buffer creation properties
    const ur_buffer_properties_t *pProperties,
    /// [out] pointer to handle of the memory buffer created
    ur_mem_handle_t *phBuffer) {
  if (nullptr == phBuffer) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  UR_LOG_L(getContext()->logger, DEBUG, "==== urMemBufferCreate");

  void *Host = nullptr;
  if (pProperties) {
    Host = pProperties->pHost;
  }

  char *hostPtrOrNull =
      (flags & UR_MEM_FLAG_USE_HOST_POINTER) ? ur_cast<char *>(Host) : nullptr;

  std::shared_ptr<MemBuffer> pMemBuffer =
      std::make_shared<MemBuffer>(hContext, size, hostPtrOrNull);

  if (Host && (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) {
    std::shared_ptr<ContextInfo> CtxInfo =
        getTsanInterceptor()->getContextInfo(hContext);
    for (const auto &hDevice : CtxInfo->DeviceList) {
      ManagedQueue InternalQueue(hContext, hDevice);
      char *Handle = nullptr;
      UR_CALL(pMemBuffer->getHandle(hDevice, Handle));
      UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
          InternalQueue, true, Handle, Host, size, 0, nullptr, nullptr));
    }
  }

  ur_result_t result = getTsanInterceptor()->insertMemBuffer(pMemBuffer);
  *phBuffer = ur_cast<ur_mem_handle_t>(pMemBuffer.get());

  return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
ur_result_t urMemRetain(
    /// [in] handle of the memory object to get access
    ur_mem_handle_t hMem) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urMemRetain");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hMem)) {
    MemBuffer->RefCount++;
  } else {
    UR_CALL(getContext()->urDdiTable.Mem.pfnRetain(hMem));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
ur_result_t urMemRelease(
    /// [in] handle of the memory object to release
    ur_mem_handle_t hMem) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urMemRelease");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hMem)) {
    if (--MemBuffer->RefCount != 0) {
      return UR_RESULT_SUCCESS;
    }
    UR_CALL(MemBuffer->free());
    UR_CALL(getTsanInterceptor()->eraseMemBuffer(hMem));
  } else {
    UR_CALL(getContext()->urDdiTable.Mem.pfnRelease(hMem));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferPartition
ur_result_t urMemBufferPartition(

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
  UR_LOG_L(getContext()->logger, DEBUG, "==== urMemBufferPartition");

  if (auto ParentBuffer = getTsanInterceptor()->getMemBuffer(hBuffer)) {
    if (ParentBuffer->Size < (pRegion->origin + pRegion->size)) {
      return UR_RESULT_ERROR_INVALID_BUFFER_SIZE;
    }
    std::shared_ptr<MemBuffer> SubBuffer = std::make_shared<MemBuffer>(
        ParentBuffer, pRegion->origin, pRegion->size);
    UR_CALL(getTsanInterceptor()->insertMemBuffer(SubBuffer));
    *phMem = reinterpret_cast<ur_mem_handle_t>(SubBuffer.get());
  } else {
    UR_CALL(getContext()->urDdiTable.Mem.pfnBufferPartition(
        hBuffer, flags, bufferCreateType, pRegion, phMem));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
ur_result_t urMemGetNativeHandle(
    /// [in] handle of the mem.
    ur_mem_handle_t hMem, ur_device_handle_t hDevice,
    /// [out] a pointer to the native handle of the mem.
    ur_native_handle_t *phNativeMem) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urMemGetNativeHandle");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hMem)) {
    char *Handle = nullptr;
    UR_CALL(MemBuffer->getHandle(hDevice, Handle));
    *phNativeMem = ur_cast<ur_native_handle_t>(Handle);
  } else {
    UR_CALL(getContext()->urDdiTable.Mem.pfnGetNativeHandle(hMem, hDevice,
                                                            phNativeMem));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetInfo
ur_result_t urMemGetInfo(
    /// [in] handle to the memory object being queried.
    ur_mem_handle_t hMemory,
    /// [in] type of the info to retrieve.
    ur_mem_info_t propName,
    /// [in] the number of bytes of memory pointed to by pPropValue.
    size_t propSize,
    /// [out][optional][typename(propName, propSize)] array of bytes holding the
    /// info. If propSize is less than the real number of bytes needed to return
    /// the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pPropValue is not used.
    void *pPropValue,
    /// [out][optional] pointer to the actual size in bytes of the queried
    /// propName.
    size_t *pPropSizeRet) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urMemGetInfo");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hMemory)) {
    UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
    switch (propName) {
    case UR_MEM_INFO_CONTEXT: {
      return ReturnValue(MemBuffer->Context);
    }
    case UR_MEM_INFO_SIZE: {
      return ReturnValue(size_t{MemBuffer->Size});
    }
    default: {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    }
  } else {
    UR_CALL(getContext()->urDdiTable.Mem.pfnGetInfo(hMemory, propName, propSize,
                                                    pPropValue, pPropSizeRet));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferRead
ur_result_t urEnqueueMemBufferRead(
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
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferRead");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hBuffer)) {
    ur_device_handle_t Device = GetDevice(hQueue);
    char *pSrc = nullptr;
    UR_CALL(MemBuffer->getHandle(Device, pSrc));
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
        hQueue, blockingRead, pDst, pSrc + offset, size, numEventsInWaitList,
        phEventWaitList, phEvent));
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferRead(
        hQueue, hBuffer, blockingRead, offset, size, pDst, numEventsInWaitList,
        phEventWaitList, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWrite
ur_result_t urEnqueueMemBufferWrite(
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
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferWrite");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hBuffer)) {
    ur_device_handle_t Device = GetDevice(hQueue);
    char *pDst = nullptr;
    UR_CALL(MemBuffer->getHandle(Device, pDst));
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
        hQueue, blockingWrite, pDst + offset, pSrc, size, numEventsInWaitList,
        phEventWaitList, phEvent));
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferWrite(
        hQueue, hBuffer, blockingWrite, offset, size, pSrc, numEventsInWaitList,
        phEventWaitList, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferReadRect
ur_result_t urEnqueueMemBufferReadRect(
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
    /// [in] length of each 2D slice in bytes in the host memory region pointed
    /// by dst
    size_t hostSlicePitch,
    /// [in] pointer to host memory where data is to be read into
    void *pDst,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferReadRect");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hBuffer)) {
    char *SrcHandle = nullptr;
    ur_device_handle_t Device = GetDevice(hQueue);
    UR_CALL(MemBuffer->getHandle(Device, SrcHandle));

    UR_CALL(EnqueueMemCopyRectHelper(
        hQueue, SrcHandle, ur_cast<char *>(pDst), bufferOrigin, hostOrigin,
        region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
        blockingRead, numEventsInWaitList, phEventWaitList, phEvent));
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferReadRect(
        hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pDst,
        numEventsInWaitList, phEventWaitList, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWriteRect
ur_result_t urEnqueueMemBufferWriteRect(
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
    /// [in] length of each 2D slice in bytes in the buffer object being written
    size_t bufferSlicePitch,
    /// [in] length of each row in bytes in the host memory region pointed by
    /// src
    size_t hostRowPitch,
    /// [in] length of each 2D slice in bytes in the host memory region pointed
    /// by src
    size_t hostSlicePitch,
    /// [in] pointer to host memory where data is to be written from
    void *pSrc,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] points to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferWriteRect");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hBuffer)) {
    char *DstHandle = nullptr;
    ur_device_handle_t Device = GetDevice(hQueue);
    UR_CALL(MemBuffer->getHandle(Device, DstHandle));

    UR_CALL(EnqueueMemCopyRectHelper(
        hQueue, ur_cast<char *>(pSrc), DstHandle, hostOrigin, bufferOrigin,
        region, hostRowPitch, hostSlicePitch, bufferRowPitch, bufferSlicePitch,
        blockingWrite, numEventsInWaitList, phEventWaitList, phEvent));
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferWriteRect(
        hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
        bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch, pSrc,
        numEventsInWaitList, phEventWaitList, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopy
ur_result_t urEnqueueMemBufferCopy(
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
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferCopy");

  auto SrcBuffer = getTsanInterceptor()->getMemBuffer(hBufferSrc);
  auto DstBuffer = getTsanInterceptor()->getMemBuffer(hBufferDst);

  UR_ASSERT((SrcBuffer && DstBuffer) || (!SrcBuffer && !DstBuffer),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  if (SrcBuffer && DstBuffer) {
    ur_device_handle_t Device = GetDevice(hQueue);
    std::shared_ptr<DeviceInfo> DeviceInfo =
        getTsanInterceptor()->getDeviceInfo(Device);
    char *SrcHandle = nullptr;
    UR_CALL(SrcBuffer->getHandle(Device, SrcHandle));

    char *DstHandle = nullptr;
    UR_CALL(DstBuffer->getHandle(Device, DstHandle));

    UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
        hQueue, false, DstHandle + dstOffset, SrcHandle + srcOffset, size,
        numEventsInWaitList, phEventWaitList, phEvent));
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferCopy(
        hQueue, hBufferSrc, hBufferDst, srcOffset, dstOffset, size,
        numEventsInWaitList, phEventWaitList, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopyRect
ur_result_t urEnqueueMemBufferCopyRect(
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
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferCopyRect");

  auto SrcBuffer = getTsanInterceptor()->getMemBuffer(hBufferSrc);
  auto DstBuffer = getTsanInterceptor()->getMemBuffer(hBufferDst);

  UR_ASSERT((SrcBuffer && DstBuffer) || (!SrcBuffer && !DstBuffer),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  if (SrcBuffer && DstBuffer) {
    ur_device_handle_t Device = GetDevice(hQueue);
    char *SrcHandle = nullptr;
    UR_CALL(SrcBuffer->getHandle(Device, SrcHandle));

    char *DstHandle = nullptr;
    UR_CALL(DstBuffer->getHandle(Device, DstHandle));

    UR_CALL(EnqueueMemCopyRectHelper(
        hQueue, SrcHandle, DstHandle, srcOrigin, dstOrigin, region, srcRowPitch,
        srcSlicePitch, dstRowPitch, dstSlicePitch, false, numEventsInWaitList,
        phEventWaitList, phEvent));
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferCopyRect(
        hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region,
        srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
        numEventsInWaitList, phEventWaitList, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferFill
ur_result_t urEnqueueMemBufferFill(
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
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferFill");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hBuffer)) {
    char *Handle = nullptr;
    ur_device_handle_t Device = GetDevice(hQueue);
    UR_CALL(MemBuffer->getHandle(Device, Handle));
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMFill(
        hQueue, Handle + offset, patternSize, pPattern, size,
        numEventsInWaitList, phEventWaitList, phEvent));
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferFill(
        hQueue, hBuffer, pPattern, patternSize, offset, size,
        numEventsInWaitList, phEventWaitList, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferMap
ur_result_t urEnqueueMemBufferMap(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    ur_mem_handle_t
        /// [in][bounds(offset, size)] handle of the buffer object
        hBuffer,
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
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this
    /// particular command instance.
    ur_event_handle_t *phEvent,
    /// [out] return mapped pointer. TODO: move it before numEventsInWaitList?
    void **ppRetMap) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemBufferMap");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hBuffer)) {
    // Translate the host access mode info.
    MemBuffer::AccessMode AccessMode = MemBuffer::UNKNOWN;
    if (mapFlags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION) {
      AccessMode = MemBuffer::WRITE_ONLY;
    } else {
      if (mapFlags & UR_MAP_FLAG_READ) {
        AccessMode = MemBuffer::READ_ONLY;
        if (mapFlags & UR_MAP_FLAG_WRITE) {
          AccessMode = MemBuffer::READ_WRITE;
        }
      } else if (mapFlags & UR_MAP_FLAG_WRITE) {
        AccessMode = MemBuffer::WRITE_ONLY;
      }
    }

    UR_ASSERT(AccessMode != MemBuffer::UNKNOWN,
              UR_RESULT_ERROR_INVALID_ARGUMENT);

    ur_device_handle_t Device = GetDevice(hQueue);
    // If the buffer used host pointer, then we just reuse it. If not, we
    // need to manually allocate a new host USM.
    if (MemBuffer->HostPtr) {
      *ppRetMap = MemBuffer->HostPtr + offset;
    } else {
      ur_context_handle_t Context = GetContext(hQueue);
      ur_usm_desc_t USMDesc{};
      USMDesc.align = MemBuffer->getAlignment();
      ur_usm_pool_handle_t Pool{};
      UR_CALL(getContext()->urDdiTable.USM.pfnHostAlloc(Context, &USMDesc, Pool,
                                                        size, ppRetMap));
    }

    // Actually, if the access mode is write only, we don't need to do this
    // copy. However, in that way, we cannot generate a event to user. So,
    // we'll aways do copy here.
    char *SrcHandle = nullptr;
    UR_CALL(MemBuffer->getHandle(Device, SrcHandle));
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
        hQueue, blockingMap, *ppRetMap, SrcHandle + offset, size,
        numEventsInWaitList, phEventWaitList, phEvent));

    {
      std::scoped_lock<ur_shared_mutex> Guard(MemBuffer->Mutex);
      UR_ASSERT(MemBuffer->Mappings.find(*ppRetMap) ==
                    MemBuffer->Mappings.end(),
                UR_RESULT_ERROR_INVALID_VALUE);
      MemBuffer->Mappings[*ppRetMap] = {offset, size};
    }
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemBufferMap(
        hQueue, hBuffer, blockingMap, mapFlags, offset, size,
        numEventsInWaitList, phEventWaitList, phEvent, ppRetMap));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemUnmap
ur_result_t urEnqueueMemUnmap(
    /// [in] handle of the queue object
    ur_queue_handle_t hQueue,
    /// [in] handle of the memory (buffer or image) object
    ur_mem_handle_t hMem,
    /// [in] mapped host address
    void *pMappedPtr,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before this command can be executed. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that this
    /// command does not wait on any event to complete.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this particular
    /// command instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueMemUnmap");

  if (auto MemBuffer = getTsanInterceptor()->getMemBuffer(hMem)) {
    MemBuffer::Mapping Mapping{};
    {
      std::scoped_lock<ur_shared_mutex> Guard(MemBuffer->Mutex);
      auto It = MemBuffer->Mappings.find(pMappedPtr);
      UR_ASSERT(It != MemBuffer->Mappings.end(), UR_RESULT_ERROR_INVALID_VALUE);
      Mapping = It->second;
      MemBuffer->Mappings.erase(It);
    }

    // Write back mapping memory data to device and release mapping memory
    // if we allocated a host USM. But for now, UR doesn't support event
    // call back, we can only do blocking copy here.
    char *DstHandle = nullptr;
    ur_context_handle_t Context = GetContext(hQueue);
    ur_device_handle_t Device = GetDevice(hQueue);
    UR_CALL(MemBuffer->getHandle(Device, DstHandle));
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
        hQueue, true, DstHandle + Mapping.Offset, pMappedPtr, Mapping.Size,
        numEventsInWaitList, phEventWaitList, phEvent));

    if (!MemBuffer->HostPtr) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context, pMappedPtr));
    }
  } else {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnMemUnmap(
        hQueue, hMem, pMappedPtr, numEventsInWaitList, phEventWaitList,
        phEvent));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
ur_result_t UR_APICALL urKernelCreate(
    /// [in] handle of the program instance
    ur_program_handle_t hProgram,
    /// [in] pointer to null-terminated string.
    const char *pKernelName,
    /// [out][alloc] pointer to handle of kernel object created.
    ur_kernel_handle_t *phKernel) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urKernelCreate");

  UR_CALL(getContext()->urDdiTable.Kernel.pfnCreate(hProgram, pKernelName,
                                                    phKernel));
  UR_CALL(getTsanInterceptor()->insertKernel(*phKernel));

  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
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
  UR_LOG_L(getContext()->logger, DEBUG, "==== urKernelCreate");

  UR_CALL(getContext()->urDdiTable.Kernel.pfnCreateWithNativeHandle(
      hNativeKernel, hContext, hProgram, pProperties, phKernel));

  UR_CALL(getTsanInterceptor()->insertKernel(*phKernel));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
ur_result_t urKernelRetain(
    /// [in] handle for the Kernel to retain
    ur_kernel_handle_t hKernel) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urKernelRetain");

  UR_CALL(getContext()->urDdiTable.Kernel.pfnRetain(hKernel));

  auto &KernelInfo = getTsanInterceptor()->getKernelInfo(hKernel);
  KernelInfo.RefCount++;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
ur_result_t urKernelRelease(
    /// [in] handle for the Kernel to release
    ur_kernel_handle_t hKernel) {
  auto pfnRelease = getContext()->urDdiTable.Kernel.pfnRelease;

  UR_LOG_L(getContext()->logger, DEBUG, "==== urKernelRelease");

  auto &KernelInfo = getTsanInterceptor()->getKernelInfo(hKernel);
  if (--KernelInfo.RefCount == 0) {
    UR_CALL(getTsanInterceptor()->eraseKernel(hKernel));
  }
  UR_CALL(pfnRelease(hKernel));

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgValue
ur_result_t urKernelSetArgValue(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t argIndex,
    /// [in] size of argument type
    size_t argSize,
    /// [in][optional] pointer to value properties.
    const ur_kernel_arg_value_properties_t *pProperties,
    /// [in] argument value represented as matching arg type.
    const void *pArgValue) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urKernelSetArgValue");

  std::shared_ptr<MemBuffer> MemBuffer;
  if (argSize == sizeof(ur_mem_handle_t) &&
      (MemBuffer = getTsanInterceptor()->getMemBuffer(
           *ur_cast<const ur_mem_handle_t *>(pArgValue)))) {
    auto &KernelInfo = getTsanInterceptor()->getKernelInfo(hKernel);
    std::scoped_lock<ur_shared_mutex> Guard(KernelInfo.Mutex);
    KernelInfo.BufferArgs[argIndex] = std::move(MemBuffer);
  } else {
    UR_CALL(getContext()->urDdiTable.Kernel.pfnSetArgValue(
        hKernel, argIndex, argSize, pProperties, pArgValue));
  }

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgMemObj
ur_result_t urKernelSetArgMemObj(
    /// [in] handle of the kernel object
    ur_kernel_handle_t hKernel,
    /// [in] argument index in range [0, num args - 1]
    uint32_t argIndex,
    /// [in][optional] pointer to Memory object properties.
    const ur_kernel_arg_mem_obj_properties_t *pProperties,
    /// [in][optional] handle of Memory object.
    ur_mem_handle_t hArgValue) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urKernelSetArgMemObj");

  if (std::shared_ptr<MemBuffer> MemBuffer =
          getTsanInterceptor()->getMemBuffer(hArgValue)) {
    auto &KernelInfo = getTsanInterceptor()->getKernelInfo(hKernel);
    std::scoped_lock<ur_shared_mutex> Guard(KernelInfo.Mutex);
    KernelInfo.BufferArgs[argIndex] = std::move(MemBuffer);
  } else {
    UR_CALL(getContext()->urDdiTable.Kernel.pfnSetArgMemObj(
        hKernel, argIndex, pProperties, hArgValue));
  }

  return UR_RESULT_SUCCESS;
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
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM device memory object
    void **ppMem) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urUSMDeviceAlloc");

  return getTsanInterceptor()->allocateMemory(
      hContext, hDevice, pUSMDesc, pool, size, AllocType::DEVICE_USM, ppMem);
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
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM host memory object
    void **ppMem) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urUSMHostAlloc");

  return getTsanInterceptor()->allocateMemory(hContext, nullptr, pUSMDesc, pool,
                                              size, AllocType::HOST_USM, ppMem);
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
    /// [in] size in bytes of the USM memory object to be allocated
    size_t size,
    /// [out] pointer to USM shared memory object
    void **ppMem) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urUSMSharedAlloc");

  return getTsanInterceptor()->allocateMemory(
      hContext, hDevice, pUSMDesc, pool, size, AllocType::SHARED_USM, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunch
ur_result_t urEnqueueKernelLaunch(
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
    /// execute the kernel function. If nullptr, the runtime implementation will
    /// choose the work-group size.
    const size_t *pLocalWorkSize,
    /// [in] size of the launch prop list
    uint32_t numPropsInLaunchPropList,
    /// [in][range(0, numPropsInLaunchPropList)] pointer to a list of launch
    /// properties
    const ur_kernel_launch_property_t *launchPropList,
    /// [in] size of the event wait list
    uint32_t numEventsInWaitList,
    /// [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    /// events that must be complete before the kernel execution. If
    /// nullptr, the numEventsInWaitList must be 0, indicating that no wait
    /// event.
    const ur_event_handle_t *phEventWaitList,
    /// [out][optional] return an event object that identifies this
    /// particular kernel execution instance.
    ur_event_handle_t *phEvent) {
  UR_LOG_L(getContext()->logger, DEBUG, "==== urEnqueueKernelLaunch");

  LaunchInfo LaunchInfo(GetContext(hQueue), GetDevice(hQueue));

  UR_CALL(getTsanInterceptor()->preLaunchKernel(hKernel, hQueue, LaunchInfo));

  UR_CALL(getContext()->urDdiTable.Enqueue.pfnKernelLaunch(
      hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
      pLocalWorkSize, numPropsInLaunchPropList, launchPropList,
      numEventsInWaitList, phEventWaitList, phEvent));

  UR_CALL(getTsanInterceptor()->postLaunchKernel(hKernel, hQueue, LaunchInfo));

  return UR_RESULT_SUCCESS;
}

ur_result_t urCheckVersion(ur_api_version_t version) {
  if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
          UR_MAJOR_VERSION(version) ||
      UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
          UR_MINOR_VERSION(version)) {
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }
  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
__urdlllocal ur_result_t UR_APICALL urGetContextProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_context_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pDdiTable->pfnCreate = ur_sanitizer_layer::tsan::urContextCreate;
  pDdiTable->pfnCreateWithNativeHandle =
      ur_sanitizer_layer::tsan::urContextCreateWithNativeHandle;
  pDdiTable->pfnRetain = ur_sanitizer_layer::tsan::urContextRetain;
  pDdiTable->pfnRelease = ur_sanitizer_layer::tsan::urContextRelease;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetProgramProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_program_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pDdiTable->pfnBuild = ur_sanitizer_layer::tsan::urProgramBuild;
  pDdiTable->pfnLink = ur_sanitizer_layer::tsan::urProgramLink;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetProgramExpProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_program_exp_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pDdiTable->pfnBuildExp = ur_sanitizer_layer::tsan::urProgramBuildExp;
  pDdiTable->pfnLinkExp = ur_sanitizer_layer::tsan::urProgramLinkExp;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetKernelProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_kernel_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pDdiTable->pfnCreate = ur_sanitizer_layer::tsan::urKernelCreate;
  pDdiTable->pfnRetain = ur_sanitizer_layer::tsan::urKernelRetain;
  pDdiTable->pfnRelease = ur_sanitizer_layer::tsan::urKernelRelease;
  pDdiTable->pfnSetArgValue = ur_sanitizer_layer::tsan::urKernelSetArgValue;
  pDdiTable->pfnSetArgMemObj = ur_sanitizer_layer::tsan::urKernelSetArgMemObj;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetMemProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_mem_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pDdiTable->pfnBufferCreate = ur_sanitizer_layer::tsan::urMemBufferCreate;
  pDdiTable->pfnRetain = ur_sanitizer_layer::tsan::urMemRetain;
  pDdiTable->pfnRelease = ur_sanitizer_layer::tsan::urMemRelease;
  pDdiTable->pfnBufferPartition =
      ur_sanitizer_layer::tsan::urMemBufferPartition;
  pDdiTable->pfnGetNativeHandle =
      ur_sanitizer_layer::tsan::urMemGetNativeHandle;
  pDdiTable->pfnGetInfo = ur_sanitizer_layer::tsan::urMemGetInfo;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
__urdlllocal ur_result_t UR_APICALL urGetUSMProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_usm_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pDdiTable->pfnDeviceAlloc = ur_sanitizer_layer::tsan::urUSMDeviceAlloc;
  pDdiTable->pfnHostAlloc = ur_sanitizer_layer::tsan::urUSMHostAlloc;
  pDdiTable->pfnSharedAlloc = ur_sanitizer_layer::tsan::urUSMSharedAlloc;

  return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
__urdlllocal ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    /// [in,out] pointer to table of DDI function pointers
    ur_enqueue_dditable_t *pDdiTable) {
  if (nullptr == pDdiTable) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pDdiTable->pfnMemBufferRead =
      ur_sanitizer_layer::tsan::urEnqueueMemBufferRead;
  pDdiTable->pfnMemBufferWrite =
      ur_sanitizer_layer::tsan::urEnqueueMemBufferWrite;
  pDdiTable->pfnMemBufferReadRect =
      ur_sanitizer_layer::tsan::urEnqueueMemBufferReadRect;
  pDdiTable->pfnMemBufferWriteRect =
      ur_sanitizer_layer::tsan::urEnqueueMemBufferWriteRect;
  pDdiTable->pfnMemBufferCopy =
      ur_sanitizer_layer::tsan::urEnqueueMemBufferCopy;
  pDdiTable->pfnMemBufferCopyRect =
      ur_sanitizer_layer::tsan::urEnqueueMemBufferCopyRect;
  pDdiTable->pfnMemBufferFill =
      ur_sanitizer_layer::tsan::urEnqueueMemBufferFill;
  pDdiTable->pfnMemBufferMap = ur_sanitizer_layer::tsan::urEnqueueMemBufferMap;
  pDdiTable->pfnMemUnmap = ur_sanitizer_layer::tsan::urEnqueueMemUnmap;
  pDdiTable->pfnKernelLaunch = ur_sanitizer_layer::tsan::urEnqueueKernelLaunch;

  return UR_RESULT_SUCCESS;
}

} // namespace tsan

ur_result_t initTsanDDITable(ur_dditable_t *dditable) {
  ur_result_t result = UR_RESULT_SUCCESS;

  UR_LOG_L(getContext()->logger, QUIET, "==== DeviceSanitizer: TSAN");

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urCheckVersion(UR_API_VERSION_CURRENT);
  }

  if (UR_RESULT_SUCCESS == result) {
    result =
        ur_sanitizer_layer::tsan::urGetContextProcAddrTable(&dditable->Context);
  }

  if (UR_RESULT_SUCCESS == result) {
    result =
        ur_sanitizer_layer::tsan::urGetProgramProcAddrTable(&dditable->Program);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urGetProgramExpProcAddrTable(
        &dditable->ProgramExp);
  }

  if (UR_RESULT_SUCCESS == result) {
    result =
        ur_sanitizer_layer::tsan::urGetKernelProcAddrTable(&dditable->Kernel);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urGetMemProcAddrTable(&dditable->Mem);
  }

  if (UR_RESULT_SUCCESS == result) {
    result = ur_sanitizer_layer::tsan::urGetUSMProcAddrTable(&dditable->USM);
  }

  if (UR_RESULT_SUCCESS == result) {
    result =
        ur_sanitizer_layer::tsan::urGetEnqueueProcAddrTable(&dditable->Enqueue);
  }

  if (result != UR_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, ERR, "Initialize TSAN DDI table failed: {}",
             result);
  }

  return result;
}

} // namespace ur_sanitizer_layer
