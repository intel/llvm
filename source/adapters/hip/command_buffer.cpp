//===--------- command_buffer.cpp - HIP Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer.hpp"
#include "common.hpp"

/// Stub implementations of UR experimental feature command-buffers

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t, ur_device_handle_t,
    const ur_exp_command_buffer_desc_t *, ur_exp_command_buffer_handle_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t, ur_kernel_handle_t, uint32_t,
    const size_t *, const size_t *, const size_t *, uint32_t,
    const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_command_handle_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t, void *, const void *, size_t, uint32_t,
    const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_mem_handle_t, size_t,
    size_t, size_t, uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_mem_handle_t,
    ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, size_t, size_t,
    size_t, size_t, uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t, ur_mem_handle_t, size_t, size_t,
    const void *, uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t, ur_mem_handle_t, size_t, size_t, void *,
    uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_rect_offset_t,
    ur_rect_offset_t, ur_rect_region_t, size_t, size_t, size_t, size_t, void *,
    uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_rect_offset_t,
    ur_rect_offset_t, ur_rect_region_t, size_t, size_t, size_t, size_t, void *,
    uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t, const void *, size_t,
    ur_usm_migration_flags_t, uint32_t,
    const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t, const void *, size_t, ur_usm_advice_flags_t,
    uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t, ur_mem_handle_t, const void *, size_t,
    size_t, size_t, uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t, void *, const void *, size_t, size_t,
    uint32_t, const ur_exp_command_buffer_sync_point_t *,
    ur_exp_command_buffer_sync_point_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t, ur_queue_handle_t, uint32_t,
    const ur_event_handle_t *, ur_event_handle_t *) {
  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainCommandExp(ur_exp_command_buffer_command_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseCommandExp(ur_exp_command_buffer_command_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_command_handle_t,
    const ur_exp_command_buffer_update_kernel_launch_desc_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferGetInfoExp(
    ur_exp_command_buffer_handle_t, ur_exp_command_buffer_info_t, size_t,
    void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCommandGetInfoExp(
    ur_exp_command_buffer_command_handle_t,
    ur_exp_command_buffer_command_info_t, size_t, void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
