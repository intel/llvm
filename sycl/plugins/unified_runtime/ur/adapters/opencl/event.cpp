//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

cl_event_info convertUREventInfoToCL(const ur_event_info_t PropName) {
  switch (PropName) {
  case UR_EVENT_INFO_COMMAND_QUEUE:
    return CL_EVENT_COMMAND_QUEUE;
    break;
  case UR_EVENT_INFO_CONTEXT:
    return CL_EVENT_CONTEXT;
    break;
  case UR_EVENT_INFO_COMMAND_TYPE:
    return CL_EVENT_COMMAND_TYPE;
    break;
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS:
    return CL_EVENT_COMMAND_EXECUTION_STATUS;
    break;
  case UR_EVENT_INFO_REFERENCE_COUNT:
    return CL_EVENT_REFERENCE_COUNT;
    break;
  default:
    return -1;
    break;
  }
}

cl_profiling_info
convertURProfilingInfoToCL(const ur_profiling_info_t PropName) {
  switch (PropName) {
  case UR_PROFILING_INFO_COMMAND_QUEUED:
    return CL_PROFILING_COMMAND_QUEUED;
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
    return CL_PROFILING_COMMAND_SUBMIT;
  case UR_PROFILING_INFO_COMMAND_START:
    return CL_PROFILING_COMMAND_START;
  // TODO(ur) add UR_PROFILING_INFO_COMMAND_COMPLETE once spec has been updated
  case UR_PROFILING_INFO_COMMAND_END:
    return CL_PROFILING_COMMAND_END;
  default:
    return -1;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent,
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const ur_event_native_properties_t *pProperties,
    ur_event_handle_t *phEvent) {
  *phEvent = reinterpret_cast<ur_event_handle_t>(hNativeEvent);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ur_native_handle_t *phNativeEvent) {
  return getNativeHandle(hEvent, phNativeEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(ur_event_handle_t hEvent) {
  cl_int RetErr = clReleaseEvent(cl_adapter::cast<cl_event>(hEvent));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(ur_event_handle_t hEvent) {
  cl_int RetErr = clRetainEvent(cl_adapter::cast<cl_event>(hEvent));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  cl_int RetErr = clWaitForEvents(
      numEvents, cl_adapter::cast<const cl_event *>(phEventWaitList));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  cl_event_info CLEventInfo = convertUREventInfoToCL(propName);
  cl_int RetErr =
      clGetEventInfo(cl_adapter::cast<cl_event>(hEvent), CLEventInfo, propSize,
                     pPropValue, pPropSizeRet);
  CL_RETURN_ON_FAILURE(RetErr);

  if (RetErr == CL_SUCCESS &&
      propName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    /* If the CL_EVENT_COMMAND_EXECUTION_STATUS info value is CL_QUEUED, change
     * it to CL_SUBMITTED. sycl::info::event::event_command_status has no
     * equivalent to CL_QUEUED.
     *
     * FIXME UR Port: This should not be part of the UR adapter. Since PI_QUEUED
     * exists, SYCL RT should be changed to handle this situation. In addition,
     * SYCL RT is relying on PI_QUEUED status to make sure that the queues are
     * flushed. */
    const auto param_value_int = static_cast<ur_event_status_t *>(pPropValue);
    if (*param_value_int == UR_EVENT_STATUS_QUEUED) {
      *param_value_int = UR_EVENT_STATUS_SUBMITTED;
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ur_profiling_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  cl_profiling_info CLProfilingInfo = convertURProfilingInfoToCL(propName);
  cl_int RetErr = clGetEventProfilingInfo(cl_adapter::cast<cl_event>(hEvent),
                                          CLProfilingInfo, propSize, pPropValue,
                                          pPropSizeRet);
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventSetCallback(ur_event_handle_t hEvent, ur_execution_info_t execStatus,
                   ur_event_callback_t pfnNotify, void *pUserData) {
  std::ignore = hEvent;
  std::ignore = execStatus;
  std::ignore = pfnNotify;
  std::ignore = pUserData;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
