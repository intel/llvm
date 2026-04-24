//===--------- platform.cpp - Level Zero Adapter v2 ----------------------===//
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

#include "../common/platform.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero_v2 {

ur_result_t urPlatformGetInfo(ur_platform_handle_t hPlatform,
                              ur_platform_info_t paramName, size_t size,
                              void *paramValue, size_t *sizeRet) {
  if (paramName == UR_PLATFORM_INFO_NAME) {
    UrReturnHelper ReturnValue(size, paramValue, sizeRet);
    return ReturnValue(
        "Intel(R) oneAPI Unified Runtime over Level-Zero V2");
  }
  return ur::level_zero::common::urPlatformGetInfo(hPlatform, paramName, size,
                                                   paramValue, sizeRet);
}

} // namespace ur::level_zero_v2
