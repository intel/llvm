//===--------- enqueue_native.cpp - LevelZero Adapter ---------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <tuple>
#include <ur_api.h>
#include <utility>

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueNativeCommandExp(
    ur_queue_handle_t hQueue,
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
    uint32_t numMemsInMemList, const ur_mem_handle_t *phMemList,
    const ur_exp_enqueue_native_command_properties_t *pProperties,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = pfnNativeEnqueue;
  std::ignore = data;
  std::ignore = numMemsInMemList;
  std::ignore = phMemList;
  std::ignore = pProperties;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
