//===--------- enqueue.cpp - OpenCL Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "adapter.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "program.hpp"
#include "queue.hpp"

cl_map_flags convertURMapFlagsToCL(ur_map_flags_t URFlags) {
  cl_map_flags CLFlags = 0;
  if (URFlags & UR_MAP_FLAG_READ) {
    CLFlags |= CL_MAP_READ;
  }
  if (URFlags & UR_MAP_FLAG_WRITE) {
    CLFlags |= CL_MAP_WRITE;
  }
  if (URFlags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION) {
    CLFlags |= CL_MAP_WRITE_INVALIDATE_REGION;
  }

  return CLFlags;
}

void MapUREventsToCL(uint32_t numEvents, const ur_event_handle_t *UREvents,
                     std::vector<cl_event> &CLEvents) {
  for (uint32_t i = 0; i < numEvents; i++) {
    CLEvents[i] = UREvents[i]->CLEvent;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numPropsInLaunchPropList,
    const ur_kernel_launch_property_t *launchPropList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  for (uint32_t propIndex = 0; propIndex < numPropsInLaunchPropList;
       propIndex++) {
    // Adapters that don't support cooperative kernels are currently expected
    // to ignore COOPERATIVE launch properties. Ideally we should avoid passing
    // these at the SYCL RT level instead, see
    // https://github.com/intel/llvm/issues/18421
    if (launchPropList[propIndex].id == UR_KERNEL_LAUNCH_PROPERTY_ID_IGNORE ||
        launchPropList[propIndex].id ==
            UR_KERNEL_LAUNCH_PROPERTY_ID_COOPERATIVE) {
      continue;
    }
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  std::vector<size_t> compiledLocalWorksize;
  if (!pLocalWorkSize) {
    cl_device_id device = nullptr;
    CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
        hQueue->CLQueue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr));
    // This query always returns size_t[3], if nothing was specified it returns
    // all zeroes.
    size_t queriedLocalWorkSize[3] = {0, 0, 0};
    CL_RETURN_ON_FAILURE(clGetKernelWorkGroupInfo(
        hKernel->CLKernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
        sizeof(size_t[3]), queriedLocalWorkSize, nullptr));
    if (queriedLocalWorkSize[0] != 0) {
      for (uint32_t i = 0; i < workDim; i++) {
        compiledLocalWorksize.push_back(queriedLocalWorkSize[i]);
      }
    }
  }

  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  auto Err = clEnqueueNDRangeKernel(
      hQueue->CLQueue, hKernel->CLKernel, workDim, pGlobalWorkOffset,
      pGlobalWorkSize,
      compiledLocalWorksize.empty() ? pLocalWorkSize
                                    : compiledLocalWorksize.data(),
      numEventsInWaitList, CLWaitEvents.data(), ifUrEvent(phEvent, Event));
  if (Err == CL_INVALID_KERNEL_ARGS) {
    UR_LOG_L(ur::cl::getAdapter()->log, ERR,
             "Kernel called with invalid arguments");
  }
  CL_RETURN_ON_FAILURE(Err);

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueMarkerWithWaitList(
      hQueue->CLQueue, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueBarrierWithWaitList(
      hQueue->CLQueue, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t urEnqueueEventsWaitWithBarrierExt(
    ur_queue_handle_t hQueue, const ur_exp_enqueue_ext_properties_t *,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueReadBuffer(
      hQueue->CLQueue, hBuffer->CLMemory, blockingRead, offset, size, pDst,
      numEventsInWaitList, CLWaitEvents.data(), ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueWriteBuffer(
      hQueue->CLQueue, hBuffer->CLMemory, blockingWrite, offset, size, pSrc,
      numEventsInWaitList, CLWaitEvents.data(), ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  const size_t BufferOrigin[3] = {bufferOrigin.x, bufferOrigin.y,
                                  bufferOrigin.z};
  const size_t HostOrigin[3] = {hostOrigin.x, hostOrigin.y, hostOrigin.z};
  const size_t Region[3] = {region.width, region.height, region.depth};
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueReadBufferRect(
      hQueue->CLQueue, hBuffer->CLMemory, blockingRead, BufferOrigin,
      HostOrigin, Region, bufferRowPitch, bufferSlicePitch, hostRowPitch,
      hostSlicePitch, pDst, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  const size_t BufferOrigin[3] = {bufferOrigin.x, bufferOrigin.y,
                                  bufferOrigin.z};
  const size_t HostOrigin[3] = {hostOrigin.x, hostOrigin.y, hostOrigin.z};
  const size_t Region[3] = {region.width, region.height, region.depth};
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueWriteBufferRect(
      hQueue->CLQueue, hBuffer->CLMemory, blockingWrite, BufferOrigin,
      HostOrigin, Region, bufferRowPitch, bufferSlicePitch, hostRowPitch,
      hostSlicePitch, pSrc, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueCopyBuffer(
      hQueue->CLQueue, hBufferSrc->CLMemory, hBufferDst->CLMemory, srcOffset,
      dstOffset, size, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  const size_t SrcOrigin[3] = {srcOrigin.x, srcOrigin.y, srcOrigin.z};
  const size_t DstOrigin[3] = {dstOrigin.x, dstOrigin.y, dstOrigin.z};
  const size_t Region[3] = {region.width, region.height, region.depth};
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueCopyBufferRect(
      hQueue->CLQueue, hBufferSrc->CLMemory, hBufferDst->CLMemory, SrcOrigin,
      DstOrigin, Region, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
      numEventsInWaitList, CLWaitEvents.data(), ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, const void *pPattern,
    size_t patternSize, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // CL FillBuffer only allows pattern sizes up to the largest CL type:
  // long16/double16
  if (patternSize <= 128) {
    cl_event Event;
    std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
    MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
    CL_RETURN_ON_FAILURE(clEnqueueFillBuffer(
        hQueue->CLQueue, hBuffer->CLMemory, pPattern, patternSize, offset, size,
        numEventsInWaitList, CLWaitEvents.data(), ifUrEvent(phEvent, Event)));

    UR_RETURN_ON_FAILURE(
        createUREvent(Event, hQueue->Context, hQueue, phEvent));
    return UR_RESULT_SUCCESS;
  }

  auto NumValues = size / sizeof(uint64_t);
  auto HostBuffer = new uint64_t[NumValues];
  auto NumChunks = patternSize / sizeof(uint64_t);
  for (size_t i = 0; i < NumValues; i++) {
    HostBuffer[i] = static_cast<const uint64_t *>(pPattern)[i % NumChunks];
  }

  cl_event WriteEvent = nullptr;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  auto ClErr = clEnqueueWriteBuffer(
      hQueue->CLQueue, hBuffer->CLMemory, false, offset, size, HostBuffer,
      numEventsInWaitList, CLWaitEvents.data(), &WriteEvent);
  if (ClErr != CL_SUCCESS) {
    delete[] HostBuffer;
    CL_RETURN_ON_FAILURE(ClErr);
  }

  auto DeleteCallback = [](cl_event, cl_int, void *pUserData) {
    delete[] static_cast<uint64_t *>(pUserData);
  };
  ClErr =
      clSetEventCallback(WriteEvent, CL_COMPLETE, DeleteCallback, HostBuffer);
  if (ClErr != CL_SUCCESS) {
    // We can attempt to recover gracefully by attempting to wait for the write
    // to finish and deleting the host buffer.
    clWaitForEvents(1, &WriteEvent);
    delete[] HostBuffer;
    clReleaseEvent(WriteEvent);
    CL_RETURN_ON_FAILURE(ClErr);
  }

  if (phEvent) {
    UR_RETURN_ON_FAILURE(
        createUREvent(WriteEvent, hQueue->Context, hQueue, phEvent));
  } else {
    CL_RETURN_ON_FAILURE(clReleaseEvent(WriteEvent));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingRead,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  const size_t Origin[3] = {origin.x, origin.y, origin.z};
  const size_t Region[3] = {region.width, region.height, region.depth};
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueReadImage(
      hQueue->CLQueue, hImage->CLMemory, blockingRead, Origin, Region, rowPitch,
      slicePitch, pDst, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingWrite,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  const size_t Origin[3] = {origin.x, origin.y, origin.z};
  const size_t Region[3] = {region.width, region.height, region.depth};
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueWriteImage(
      hQueue->CLQueue, hImage->CLMemory, blockingWrite, Origin, Region,
      rowPitch, slicePitch, pSrc, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));
  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImageSrc,
    ur_mem_handle_t hImageDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  const size_t SrcOrigin[3] = {srcOrigin.x, srcOrigin.y, srcOrigin.z};
  const size_t DstOrigin[3] = {dstOrigin.x, dstOrigin.y, dstOrigin.z};
  const size_t Region[3] = {region.width, region.height, region.depth};
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueCopyImage(
      hQueue->CLQueue, hImageSrc->CLMemory, hImageDst->CLMemory, SrcOrigin,
      DstOrigin, Region, numEventsInWaitList, CLWaitEvents.data(),
      ifUrEvent(phEvent, Event)));
  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingMap,
    ur_map_flags_t mapFlags, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, void **ppRetMap) {
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  cl_int Err;
  *ppRetMap = clEnqueueMapBuffer(
      hQueue->CLQueue, hBuffer->CLMemory, blockingMap,
      convertURMapFlagsToCL(mapFlags), offset, size, numEventsInWaitList,
      CLWaitEvents.data(), ifUrEvent(phEvent, Event), &Err);
  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return mapCLErrorToUR(Err);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  CL_RETURN_ON_FAILURE(clEnqueueUnmapMemObject(
      hQueue->CLQueue, hMem->CLMemory, pMappedPtr, numEventsInWaitList,
      CLWaitEvents.data(), ifUrEvent(phEvent, Event)));
  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context Ctx = hQueue->Context->CLContext;
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  cl_ext::clEnqueueWriteGlobalVariable_fn F = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<decltype(F)>(
      Ctx, ur::cl::getAdapter()->fnCache.clEnqueueWriteGlobalVariableCache,
      cl_ext::EnqueueWriteGlobalVariableName, &F));

  cl_int Res = F(hQueue->CLQueue, hProgram->CLProgram, name, blockingWrite,
                 count, offset, pSrc, numEventsInWaitList, CLWaitEvents.data(),
                 ifUrEvent(phEvent, Event));
  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return mapCLErrorToUR(Res);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context Ctx = hQueue->Context->CLContext;
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  cl_ext::clEnqueueReadGlobalVariable_fn F = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<decltype(F)>(
      Ctx, ur::cl::getAdapter()->fnCache.clEnqueueReadGlobalVariableCache,
      cl_ext::EnqueueReadGlobalVariableName, &F));

  cl_int Res = F(hQueue->CLQueue, hProgram->CLProgram, name, blockingRead,
                 count, offset, pDst, numEventsInWaitList, CLWaitEvents.data(),
                 ifUrEvent(phEvent, Event));

  UR_RETURN_ON_FAILURE(createUREvent(Event, hQueue->Context, hQueue, phEvent));
  return mapCLErrorToUR(Res);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pDst, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context CLContext = hQueue->Context->CLContext;
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  cl_ext::clEnqueueReadHostPipeINTEL_fn FuncPtr = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<cl_ext::clEnqueueReadHostPipeINTEL_fn>(
          CLContext,
          ur::cl::getAdapter()->fnCache.clEnqueueReadHostPipeINTELCache,
          cl_ext::EnqueueReadHostPipeName, &FuncPtr));

  if (FuncPtr) {
    CL_RETURN_ON_FAILURE(FuncPtr(
        hQueue->CLQueue, hProgram->CLProgram, pipe_symbol, blocking, pDst, size,
        numEventsInWaitList, CLWaitEvents.data(), ifUrEvent(phEvent, Event)));

    UR_RETURN_ON_FAILURE(
        createUREvent(Event, hQueue->Context, hQueue, phEvent));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context CLContext = hQueue->Context->CLContext;
  cl_event Event;
  std::vector<cl_event> CLWaitEvents(numEventsInWaitList);
  MapUREventsToCL(numEventsInWaitList, phEventWaitList, CLWaitEvents);
  cl_ext::clEnqueueWriteHostPipeINTEL_fn FuncPtr = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<cl_ext::clEnqueueWriteHostPipeINTEL_fn>(
          CLContext,
          ur::cl::getAdapter()->fnCache.clEnqueueWriteHostPipeINTELCache,
          cl_ext::EnqueueWriteHostPipeName, &FuncPtr));

  if (FuncPtr) {
    CL_RETURN_ON_FAILURE(FuncPtr(
        hQueue->CLQueue, hProgram->CLProgram, pipe_symbol, blocking, pSrc, size,
        numEventsInWaitList, CLWaitEvents.data(), ifUrEvent(phEvent, Event)));
    UR_RETURN_ON_FAILURE(
        createUREvent(Event, hQueue->Context, hQueue, phEvent));
  }

  return UR_RESULT_SUCCESS;
}
