//===--------- command_buffer.cpp - HIP Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "command_buffer.hpp"
#include "common.hpp"

/// Stub implementations of UR experimental feature command-buffers

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    ur_exp_command_buffer_handle_t *phCommandBuffer) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pCommandBufferDesc;
  std::ignore = phCommandBuffer;
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  std::ignore = hCommandBuffer;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  std::ignore = hCommandBuffer;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  std::ignore = hCommandBuffer;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = hKernel;
  std::ignore = workDim;
  std::ignore = pGlobalWorkOffset;
  std::ignore = pGlobalWorkSize;
  std::ignore = pLocalWorkSize;
  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemcpyUSMExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = pDst;
  std::ignore = pSrc;
  std::ignore = size;
  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMembufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = hSrcMem;
  std::ignore = hDstMem;
  std::ignore = srcOffset;
  std::ignore = dstOffset;
  std::ignore = size;
  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMembufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = hSrcMem;
  std::ignore = hDstMem;
  std::ignore = srcOrigin;
  std::ignore = dstOrigin;
  std::ignore = region;
  std::ignore = srcRowPitch;
  std::ignore = srcSlicePitch;
  std::ignore = dstRowPitch;
  std::ignore = dstSlicePitch;
  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = hBuffer;
  std::ignore = offset;
  std::ignore = size;
  std::ignore = pSrc;
  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = hBuffer;
  std::ignore = offset;
  std::ignore = size;
  std::ignore = pDst;
  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = hBuffer;
  std::ignore = bufferOffset;
  std::ignore = hostOffset;
  std::ignore = region;
  std::ignore = bufferRowPitch;
  std::ignore = bufferSlicePitch;
  std::ignore = hostRowPitch;
  std::ignore = hostSlicePitch;
  std::ignore = pSrc;
  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  std::ignore = hCommandBuffer;
  std::ignore = hBuffer;
  std::ignore = bufferOffset;
  std::ignore = hostOffset;
  std::ignore = region;
  std::ignore = bufferRowPitch;
  std::ignore = bufferSlicePitch;
  std::ignore = hostRowPitch;
  std::ignore = hostSlicePitch;
  std::ignore = pDst;

  std::ignore = numSyncPointsInWaitList;
  std::ignore = pSyncPointWaitList;
  std::ignore = pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hCommandBuffer;
  std::ignore = hQueue;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
