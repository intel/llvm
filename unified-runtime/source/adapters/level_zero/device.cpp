//===--------- device.cpp - Level Zero Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unified-runtime/ur_api.h>
#include <ur/ur.hpp>

#include "common/device.hpp"
#include "common/platform.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero::v1 {

ur_result_t urDeviceGetInfo(::ur_device_handle_t hDeviceOpque,
                            ::ur_device_info_t propName, size_t propSize,
                            void *propValue, size_t *propSizeRet) {
  UrReturnHelper ReturnValue(propSize, propValue, propSizeRet);

  switch (propName) {
  case UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP:
    return ReturnValue(static_cast<ur_bool_t>(false));
  case UR_DEVICE_INFO_PER_EVENT_PROFILING_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_USM_HOST_ALLOC_REGISTER_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_IPC_PHYSICAL_MEMORY_SUPPORT_EXP:
    return ReturnValue(false);
  case UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP:
    return ReturnValue(false);
  default:
    return ur::level_zero::urDeviceGetInfo(hDeviceOpque, propName, propSize,
                                           propValue, propSizeRet);
  }
}

} // namespace ur::level_zero::v1
