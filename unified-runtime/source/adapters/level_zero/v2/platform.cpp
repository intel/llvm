//===--------- platform.cpp - Level Zero Adapter v2 ----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unified-runtime/ur_api.h>
#include <ur/ur.hpp>

#include "../common/platform.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero::v2 {

ur_result_t urPlatformGetInfo(::ur_platform_handle_t hPlatformOpque,
                              ::ur_platform_info_t paramName, size_t size,
                              void *paramValue, size_t *sizeRet) {
  if (paramName == UR_PLATFORM_INFO_NAME) {
    UrReturnHelper ReturnValue(size, paramValue, sizeRet);
    return ReturnValue("Intel(R) oneAPI Unified Runtime over Level-Zero V2");
  }
  return ur::level_zero::urPlatformGetInfo(hPlatformOpque, paramName, size,
                                           paramValue, sizeRet);
}

} // namespace ur::level_zero::v2
