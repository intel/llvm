//===--------- command_buffer.cpp - OpenCL Adapter ---------------------===//
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
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    [[maybe_unused]] ur_exp_command_buffer_handle_t *phCommandBuffer) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferRetainExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferReleaseExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferFinalizeExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_kernel_handle_t hKernel,
    [[maybe_unused]] uint32_t workDim,
    [[maybe_unused]] const size_t *pGlobalWorkOffset,
    [[maybe_unused]] const size_t *pGlobalWorkSize,
    [[maybe_unused]] const size_t *pLocalWorkSize,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] void *pDst, [[maybe_unused]] const void *pSrc,
    [[maybe_unused]] size_t size,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hSrcMem,
    [[maybe_unused]] ur_mem_handle_t hDstMem, [[maybe_unused]] size_t srcOffset,
    [[maybe_unused]] size_t dstOffset, [[maybe_unused]] size_t size,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hSrcMem,
    [[maybe_unused]] ur_mem_handle_t hDstMem,
    [[maybe_unused]] ur_rect_offset_t srcOrigin,
    [[maybe_unused]] ur_rect_offset_t dstOrigin,
    [[maybe_unused]] ur_rect_region_t region,
    [[maybe_unused]] size_t srcRowPitch, [[maybe_unused]] size_t srcSlicePitch,
    [[maybe_unused]] size_t dstRowPitch, [[maybe_unused]] size_t dstSlicePitch,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer, [[maybe_unused]] size_t offset,
    [[maybe_unused]] size_t size, [[maybe_unused]] const void *pSrc,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer, [[maybe_unused]] size_t offset,
    [[maybe_unused]] size_t size, [[maybe_unused]] void *pDst,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer,
    [[maybe_unused]] ur_rect_offset_t bufferOffset,
    [[maybe_unused]] ur_rect_offset_t hostOffset,
    [[maybe_unused]] ur_rect_region_t region,
    [[maybe_unused]] size_t bufferRowPitch,
    [[maybe_unused]] size_t bufferSlicePitch,
    [[maybe_unused]] size_t hostRowPitch,
    [[maybe_unused]] size_t hostSlicePitch, [[maybe_unused]] void *pSrc,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_mem_handle_t hBuffer,
    [[maybe_unused]] ur_rect_offset_t bufferOffset,
    [[maybe_unused]] ur_rect_offset_t hostOffset,
    [[maybe_unused]] ur_rect_region_t region,
    [[maybe_unused]] size_t bufferRowPitch,
    [[maybe_unused]] size_t bufferSlicePitch,
    [[maybe_unused]] size_t hostRowPitch,
    [[maybe_unused]] size_t hostSlicePitch, [[maybe_unused]] void *pDst,
    [[maybe_unused]] uint32_t numSyncPointsInWaitList,
    [[maybe_unused]] const ur_exp_command_buffer_sync_point_t
        *pSyncPointWaitList,
    [[maybe_unused]] ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    [[maybe_unused]] ur_exp_command_buffer_handle_t hCommandBuffer,
    [[maybe_unused]] ur_queue_handle_t hQueue,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {

  cl_adapter::die("Experimental Command-buffer feature is not "
                  "implemented for OpenCL adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
