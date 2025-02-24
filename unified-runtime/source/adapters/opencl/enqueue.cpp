//===--------- enqueue.cpp - OpenCL Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

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

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::vector<size_t> compiledLocalWorksize;
  if (!pLocalWorkSize) {
    cl_device_id device = nullptr;
    CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
        cl_adapter::cast<cl_command_queue>(hQueue), CL_QUEUE_DEVICE,
        sizeof(device), &device, nullptr));
    // This query always returns size_t[3], if nothing was specified it returns
    // all zeroes.
    size_t queriedLocalWorkSize[3] = {0, 0, 0};
    CL_RETURN_ON_FAILURE(clGetKernelWorkGroupInfo(
        cl_adapter::cast<cl_kernel>(hKernel), device,
        CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t[3]),
        queriedLocalWorkSize, nullptr));
    if (queriedLocalWorkSize[0] != 0) {
      for (uint32_t i = 0; i < workDim; i++) {
        compiledLocalWorksize.push_back(queriedLocalWorkSize[i]);
      }
    }
  }

  CL_RETURN_ON_FAILURE(clEnqueueNDRangeKernel(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_kernel>(hKernel), workDim, pGlobalWorkOffset,
      pGlobalWorkSize,
      compiledLocalWorksize.empty() ? pLocalWorkSize
                                    : compiledLocalWorksize.data(),
      numEventsInWaitList, cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return urEnqueueKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                               pGlobalWorkSize, pLocalWorkSize,
                               numEventsInWaitList, phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  CL_RETURN_ON_FAILURE(clEnqueueMarkerWithWaitList(
      cl_adapter::cast<cl_command_queue>(hQueue), numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  CL_RETURN_ON_FAILURE(clEnqueueBarrierWithWaitList(
      cl_adapter::cast<cl_command_queue>(hQueue), numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

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

  CL_RETURN_ON_FAILURE(clEnqueueReadBuffer(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBuffer), blockingRead, offset, size, pDst,
      numEventsInWaitList, cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  CL_RETURN_ON_FAILURE(clEnqueueWriteBuffer(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBuffer), blockingWrite, offset, size, pSrc,
      numEventsInWaitList, cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

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

  CL_RETURN_ON_FAILURE(clEnqueueReadBufferRect(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBuffer), blockingRead, BufferOrigin, HostOrigin,
      Region, bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
      pDst, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

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

  CL_RETURN_ON_FAILURE(clEnqueueWriteBufferRect(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBuffer), blockingWrite, BufferOrigin,
      HostOrigin, Region, bufferRowPitch, bufferSlicePitch, hostRowPitch,
      hostSlicePitch, pSrc, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  CL_RETURN_ON_FAILURE(clEnqueueCopyBuffer(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBufferSrc),
      cl_adapter::cast<cl_mem>(hBufferDst), srcOffset, dstOffset, size,
      numEventsInWaitList, cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

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

  CL_RETURN_ON_FAILURE(clEnqueueCopyBufferRect(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBufferSrc),
      cl_adapter::cast<cl_mem>(hBufferDst), SrcOrigin, DstOrigin, Region,
      srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
      numEventsInWaitList, cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

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
    CL_RETURN_ON_FAILURE(
        clEnqueueFillBuffer(cl_adapter::cast<cl_command_queue>(hQueue),
                            cl_adapter::cast<cl_mem>(hBuffer), pPattern,
                            patternSize, offset, size, numEventsInWaitList,
                            cl_adapter::cast<const cl_event *>(phEventWaitList),
                            cl_adapter::cast<cl_event *>(phEvent)));
    return UR_RESULT_SUCCESS;
  }

  auto NumValues = size / sizeof(uint64_t);
  auto HostBuffer = new uint64_t[NumValues];
  auto NumChunks = patternSize / sizeof(uint64_t);
  for (size_t i = 0; i < NumValues; i++) {
    HostBuffer[i] = static_cast<const uint64_t *>(pPattern)[i % NumChunks];
  }

  cl_event WriteEvent = nullptr;
  auto ClErr = clEnqueueWriteBuffer(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBuffer), false, offset, size, HostBuffer,
      numEventsInWaitList, cl_adapter::cast<const cl_event *>(phEventWaitList),
      &WriteEvent);
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
    *phEvent = cl_adapter::cast<ur_event_handle_t>(WriteEvent);
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

  CL_RETURN_ON_FAILURE(clEnqueueReadImage(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hImage), blockingRead, Origin, Region, rowPitch,
      slicePitch, pDst, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingWrite,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  const size_t Origin[3] = {origin.x, origin.y, origin.z};
  const size_t Region[3] = {region.width, region.height, region.depth};

  CL_RETURN_ON_FAILURE(clEnqueueWriteImage(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hImage), blockingWrite, Origin, Region, rowPitch,
      slicePitch, pSrc, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

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

  CL_RETURN_ON_FAILURE(clEnqueueCopyImage(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hImageSrc), cl_adapter::cast<cl_mem>(hImageDst),
      SrcOrigin, DstOrigin, Region, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingMap,
    ur_map_flags_t mapFlags, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, void **ppRetMap) {

  cl_int Err;
  *ppRetMap = clEnqueueMapBuffer(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hBuffer), blockingMap,
      convertURMapFlagsToCL(mapFlags), offset, size, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent), &Err);

  return mapCLErrorToUR(Err);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  CL_RETURN_ON_FAILURE(clEnqueueUnmapMemObject(
      cl_adapter::cast<cl_command_queue>(hQueue),
      cl_adapter::cast<cl_mem>(hMem), pMappedPtr, numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context Ctx = nullptr;
  cl_int Res =
      clGetCommandQueueInfo(cl_adapter::cast<cl_command_queue>(hQueue),
                            CL_QUEUE_CONTEXT, sizeof(Ctx), &Ctx, nullptr);

  if (Res != CL_SUCCESS)
    return mapCLErrorToUR(Res);

  cl_ext::clEnqueueWriteGlobalVariable_fn F = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<decltype(F)>(
      Ctx, cl_ext::ExtFuncPtrCache->clEnqueueWriteGlobalVariableCache,
      cl_ext::EnqueueWriteGlobalVariableName, &F));

  Res = F(cl_adapter::cast<cl_command_queue>(hQueue),
          cl_adapter::cast<cl_program>(hProgram), name, blockingWrite, count,
          offset, pSrc, numEventsInWaitList,
          cl_adapter::cast<const cl_event *>(phEventWaitList),
          cl_adapter::cast<cl_event *>(phEvent));

  return mapCLErrorToUR(Res);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context Ctx = nullptr;
  cl_int Res =
      clGetCommandQueueInfo(cl_adapter::cast<cl_command_queue>(hQueue),
                            CL_QUEUE_CONTEXT, sizeof(Ctx), &Ctx, nullptr);

  if (Res != CL_SUCCESS)
    return mapCLErrorToUR(Res);

  cl_ext::clEnqueueReadGlobalVariable_fn F = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<decltype(F)>(
      Ctx, cl_ext::ExtFuncPtrCache->clEnqueueReadGlobalVariableCache,
      cl_ext::EnqueueReadGlobalVariableName, &F));

  Res = F(cl_adapter::cast<cl_command_queue>(hQueue),
          cl_adapter::cast<cl_program>(hProgram), name, blockingRead, count,
          offset, pDst, numEventsInWaitList,
          cl_adapter::cast<const cl_event *>(phEventWaitList),
          cl_adapter::cast<cl_event *>(phEvent));

  return mapCLErrorToUR(Res);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pDst, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context CLContext;
  cl_int CLErr = clGetCommandQueueInfo(
      cl_adapter::cast<cl_command_queue>(hQueue), CL_QUEUE_CONTEXT,
      sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return mapCLErrorToUR(CLErr);
  }

  cl_ext::clEnqueueReadHostPipeINTEL_fn FuncPtr = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<cl_ext::clEnqueueReadHostPipeINTEL_fn>(
          CLContext, cl_ext::ExtFuncPtrCache->clEnqueueReadHostPipeINTELCache,
          cl_ext::EnqueueReadHostPipeName, &FuncPtr));

  if (FuncPtr) {
    CL_RETURN_ON_FAILURE(
        FuncPtr(cl_adapter::cast<cl_command_queue>(hQueue),
                cl_adapter::cast<cl_program>(hProgram), pipe_symbol, blocking,
                pDst, size, numEventsInWaitList,
                cl_adapter::cast<const cl_event *>(phEventWaitList),
                cl_adapter::cast<cl_event *>(phEvent)));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  cl_context CLContext;
  cl_int CLErr = clGetCommandQueueInfo(
      cl_adapter::cast<cl_command_queue>(hQueue), CL_QUEUE_CONTEXT,
      sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return mapCLErrorToUR(CLErr);
  }

  cl_ext::clEnqueueWriteHostPipeINTEL_fn FuncPtr = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<cl_ext::clEnqueueWriteHostPipeINTEL_fn>(
          CLContext, cl_ext::ExtFuncPtrCache->clEnqueueWriteHostPipeINTELCache,
          cl_ext::EnqueueWriteHostPipeName, &FuncPtr));

  if (FuncPtr) {
    CL_RETURN_ON_FAILURE(
        FuncPtr(cl_adapter::cast<cl_command_queue>(hQueue),
                cl_adapter::cast<cl_program>(hProgram), pipe_symbol, blocking,
                pSrc, size, numEventsInWaitList,
                cl_adapter::cast<const cl_event *>(phEventWaitList),
                cl_adapter::cast<cl_event *>(phEvent)));
  }

  return UR_RESULT_SUCCESS;
}
