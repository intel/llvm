//===--------- device.cpp - Level Zero Adapter v2 ------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur/ur.hpp>
#include <ur_api.h>

#include "../device.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero_v2 {

ur_result_t urDeviceGetInfo(ur_device_handle_t hDevice,
                            ur_device_info_t propName, size_t propSize,
                            void *propValue, size_t *propSizeRet) {
  if (propName == UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP) {
    UrReturnHelper ReturnValue(propSize, propValue, propSizeRet);
    return ReturnValue(hDevice->Platform->ZeGraphExt.Supported);
  }
  if (propName == UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP) {
    UrReturnHelper ReturnValue(propSize, propValue, propSizeRet);
    return ReturnValue(hDevice->Platform->ZeHostTaskExt.Supported);
  }
  return ur::level_zero::urDeviceGetInfo(hDevice, propName, propSize, propValue,
                                         propSizeRet);
}

} // namespace ur::level_zero_v2
