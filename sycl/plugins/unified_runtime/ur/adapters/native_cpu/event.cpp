//===--------- event.cpp - NATIVE CPU Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  std::ignore = hEvent;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ur_profiling_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hEvent;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  std::ignore = numEvents;
  std::ignore = phEventWaitList;
  // TODO: currently we do everything synchronously so this is a no-op
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(ur_event_handle_t hEvent) {
  std::ignore = hEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(ur_event_handle_t hEvent) {
  std::ignore = hEvent;
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ur_native_handle_t *phNativeEvent) {
  std::ignore = hEvent;
  std::ignore = phNativeEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ur_context_handle_t hContext,
    const ur_event_native_properties_t *pProperties,
    ur_event_handle_t *phEvent) {
  std::ignore = hNativeEvent;
  std::ignore = hContext;
  std::ignore = pProperties;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventSetCallback(ur_event_handle_t hEvent, ur_execution_info_t execStatus,
                   ur_event_callback_t pfnNotify, void *pUserData) {
  std::ignore = hEvent;
  std::ignore = execStatus;
  std::ignore = pfnNotify;
  std::ignore = pUserData;

  DIE_NO_IMPLEMENTATION;
}
