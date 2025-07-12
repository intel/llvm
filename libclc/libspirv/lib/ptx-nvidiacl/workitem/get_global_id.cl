//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

extern int __nvvm_reflect_ocl(constant char *);

_CLC_DEF _CLC_OVERLOAD size_t __spirv_BuiltInGlobalInvocationId(int dim) {
  switch (dim) {
  case 0: {
    if (__nvvm_reflect_ocl("__CUDA_ID_QUERIES_FIT_IN_INT")) {
      return (uint)__spirv_BuiltInWorkgroupId(0) *
                 (uint)__spirv_BuiltInWorkgroupSize(0) +
             (uint)__spirv_BuiltInLocalInvocationId(0) +
             (uint)__spirv_BuiltInGlobalOffset(0);
    }
    return __spirv_BuiltInWorkgroupId(0) * __spirv_BuiltInWorkgroupSize(0) +
           __spirv_BuiltInLocalInvocationId(0) + __spirv_BuiltInGlobalOffset(0);
  }
  case 1: {
    if (__nvvm_reflect_ocl("__CUDA_ID_QUERIES_FIT_IN_INT")) {
      return (uint)__spirv_BuiltInWorkgroupId(1) *
                 (uint)__spirv_BuiltInWorkgroupSize(1) +
             (uint)__spirv_BuiltInLocalInvocationId(1) +
             (uint)__spirv_BuiltInGlobalOffset(1);
    }
    return __spirv_BuiltInWorkgroupId(1) * __spirv_BuiltInWorkgroupSize(1) +
           __spirv_BuiltInLocalInvocationId(1) + __spirv_BuiltInGlobalOffset(1);
  }
  case 2: {
    if (__nvvm_reflect_ocl("__CUDA_ID_QUERIES_FIT_IN_INT")) {
      return (uint)__spirv_BuiltInWorkgroupId(2) *
                 (uint)__spirv_BuiltInWorkgroupSize(2) +
             (uint)__spirv_BuiltInLocalInvocationId(2) +
             (uint)__spirv_BuiltInGlobalOffset(2);
    }
    return __spirv_BuiltInWorkgroupId(2) * __spirv_BuiltInWorkgroupSize(2) +
           __spirv_BuiltInLocalInvocationId(2) + __spirv_BuiltInGlobalOffset(2);
  }
  default:
    return 0;
  }
}
