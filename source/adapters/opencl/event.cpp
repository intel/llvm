//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include <mutex>
#include <set>
#include <unordered_map>

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

ur_command_t convertCLCommandTypeToUR(const cl_command_type &CommandType) {
  /* Note: the following enums don't have a CL equivalent:
    UR_COMMAND_USM_FILL_2D
    UR_COMMAND_USM_MEMCPY_2D
    UR_COMMAND_DEVICE_GLOBAL_VARIABLE_WRITE
    UR_COMMAND_DEVICE_GLOBAL_VARIABLE_READ
    UR_COMMAND_READ_HOST_PIPE
    UR_COMMAND_WRITE_HOST_PIPE
    UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP
    UR_COMMAND_INTEROP_SEMAPHORE_WAIT_EXP
    UR_COMMAND_INTEROP_SEMAPHORE_SIGNAL_EXP */
  switch (CommandType) {
  case CL_COMMAND_NDRANGE_KERNEL:
    return UR_COMMAND_KERNEL_LAUNCH;
  case CL_COMMAND_MARKER:
    // CL can't distinguish between UR_COMMAND_EVENTS_WAIT_WITH_BARRIER and
    // UR_COMMAND_EVENTS_WAIT.
    return UR_COMMAND_EVENTS_WAIT;
  case CL_COMMAND_READ_BUFFER:
    return UR_COMMAND_MEM_BUFFER_READ;
  case CL_COMMAND_WRITE_BUFFER:
    return UR_COMMAND_MEM_BUFFER_WRITE;
  case CL_COMMAND_READ_BUFFER_RECT:
    return UR_COMMAND_MEM_BUFFER_READ_RECT;
  case CL_COMMAND_WRITE_BUFFER_RECT:
    return UR_COMMAND_MEM_BUFFER_WRITE_RECT;
  case CL_COMMAND_COPY_BUFFER:
    return UR_COMMAND_MEM_BUFFER_COPY;
  case CL_COMMAND_COPY_BUFFER_RECT:
    return UR_COMMAND_MEM_BUFFER_COPY_RECT;
  case CL_COMMAND_FILL_BUFFER:
    return UR_COMMAND_MEM_BUFFER_FILL;
  case CL_COMMAND_READ_IMAGE:
    return UR_COMMAND_MEM_IMAGE_READ;
  case CL_COMMAND_WRITE_IMAGE:
    return UR_COMMAND_MEM_IMAGE_WRITE;
  case CL_COMMAND_COPY_IMAGE:
    return UR_COMMAND_MEM_IMAGE_COPY;
  case CL_COMMAND_MAP_BUFFER:
    return UR_COMMAND_MEM_BUFFER_MAP;
  case CL_COMMAND_UNMAP_MEM_OBJECT:
    return UR_COMMAND_MEM_UNMAP;
  case CL_COMMAND_MEMFILL_INTEL:
    return UR_COMMAND_USM_FILL;
  case CL_COMMAND_MEMCPY_INTEL:
    return UR_COMMAND_USM_MEMCPY;
  case CL_COMMAND_MIGRATEMEM_INTEL:
    return UR_COMMAND_USM_PREFETCH;
  case CL_COMMAND_MEMADVISE_INTEL:
    return UR_COMMAND_USM_ADVISE;
  default:
    return UR_COMMAND_FORCE_UINT32;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventCreateWithNativeHandle(ur_native_handle_t hNativeEvent,
                              [[maybe_unused]] ur_context_handle_t hContext,
                              const ur_event_native_properties_t *pProperties,
                              ur_event_handle_t *phEvent) {
  *phEvent = reinterpret_cast<ur_event_handle_t>(hNativeEvent);
  if (!pProperties || !pProperties->isNativeHandleOwned) {
    return urEventRetain(*phEvent);
  }
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

  size_t CheckPropSize = 0;
  cl_int RetErr =
      clGetEventInfo(cl_adapter::cast<cl_event>(hEvent), CLEventInfo, propSize,
                     pPropValue, &CheckPropSize);
  if (pPropValue && CheckPropSize != propSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }
  CL_RETURN_ON_FAILURE(RetErr);
  if (pPropSizeRet) {
    *pPropSizeRet = CheckPropSize;
  }

  if (pPropValue) {
    if (propName == UR_EVENT_INFO_COMMAND_TYPE) {
      *reinterpret_cast<ur_command_t *>(pPropValue) = convertCLCommandTypeToUR(
          *reinterpret_cast<cl_command_type *>(pPropValue));
    } else if (propName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
      /* If the CL_EVENT_COMMAND_EXECUTION_STATUS info value is CL_QUEUED,
       * change it to CL_SUBMITTED. sycl::info::event::event_command_status has
       * no equivalent to CL_QUEUED.
       *
       * FIXME UR Port: This should not be part of the UR adapter. Since
       * PI_QUEUED exists, SYCL RT should be changed to handle this situation.
       * In addition, SYCL RT is relying on PI_QUEUED status to make sure that
       * the queues are flushed. */
      const auto param_value_int = static_cast<ur_event_status_t *>(pPropValue);
      if (*param_value_int == UR_EVENT_STATUS_QUEUED) {
        *param_value_int = UR_EVENT_STATUS_SUBMITTED;
      }
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
  static std::unordered_map<ur_event_handle_t, std::set<ur_event_callback_t>>
      EventCallbackMap;
  static std::mutex EventCallbackMutex;

  {
    std::lock_guard<std::mutex> Lock(EventCallbackMutex);
    // Callbacks can only be registered once and we need to avoid double
    // allocating.
    if (EventCallbackMap.count(hEvent) &&
        EventCallbackMap[hEvent].count(pfnNotify)) {
      return UR_RESULT_SUCCESS;
    }

    EventCallbackMap[hEvent].insert(pfnNotify);
  }

  cl_int CallbackType = 0;
  switch (execStatus) {
  case UR_EXECUTION_INFO_EXECUTION_INFO_SUBMITTED:
    CallbackType = CL_SUBMITTED;
    break;
  case UR_EXECUTION_INFO_EXECUTION_INFO_RUNNING:
    CallbackType = CL_RUNNING;
    break;
  case UR_EXECUTION_INFO_EXECUTION_INFO_COMPLETE:
    CallbackType = CL_COMPLETE;
    break;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  struct EventCallback {
    void execute() {
      pfnNotify(hEvent, execStatus, pUserData);
      {
        std::lock_guard<std::mutex> Lock(*CallbackMutex);
        (*CallbackMap)[hEvent].erase(pfnNotify);
        if ((*CallbackMap)[hEvent].empty()) {
          CallbackMap->erase(hEvent);
        }
      }
      delete this;
    }
    ur_event_handle_t hEvent;
    ur_execution_info_t execStatus;
    ur_event_callback_t pfnNotify;
    void *pUserData;
    std::unordered_map<ur_event_handle_t, std::set<ur_event_callback_t>>
        *CallbackMap;
    std::mutex *CallbackMutex;
  };
  auto Callback = new EventCallback({hEvent, execStatus, pfnNotify, pUserData,
                                     &EventCallbackMap, &EventCallbackMutex});
  auto ClCallback = [](cl_event, cl_int, void *pUserData) {
    auto *C = static_cast<EventCallback *>(pUserData);
    C->execute();
  };
  CL_RETURN_ON_FAILURE(clSetEventCallback(cl_adapter::cast<cl_event>(hEvent),
                                          CallbackType, ClCallback, Callback));
  return UR_RESULT_SUCCESS;
}
