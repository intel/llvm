//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD size_t __spirv_BuiltInGlobalInvocationId(int dim) {
  switch (dim) {
  case 0:
    return __spirv_BuiltInWorkgroupId(0) * __spirv_BuiltInWorkgroupSize(0) +
           __spirv_BuiltInLocalInvocationId(0) + __spirv_BuiltInGlobalOffset(0);
  case 1:
    return __spirv_BuiltInWorkgroupId(1) * __spirv_BuiltInWorkgroupSize(1) +
           __spirv_BuiltInLocalInvocationId(1) + __spirv_BuiltInGlobalOffset(1);
  case 2:
    return __spirv_BuiltInWorkgroupId(2) * __spirv_BuiltInWorkgroupSize(2) +
           __spirv_BuiltInLocalInvocationId(2) + __spirv_BuiltInGlobalOffset(2);
  default:
    return 0;
  }
}
