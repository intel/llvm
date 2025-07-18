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
  if (__nvvm_reflect_ocl("__CUDA_ID_QUERIES_FIT_IN_INT")) {
    return (uint)__spirv_BuiltInWorkgroupId(dim) *
               (uint)__spirv_BuiltInWorkgroupSize(dim) +
           (uint)__spirv_BuiltInLocalInvocationId(dim) +
           (uint)__spirv_BuiltInGlobalOffset(dim);
  }
  return __spirv_BuiltInWorkgroupId(dim) * __spirv_BuiltInWorkgroupSize(dim) +
         __spirv_BuiltInLocalInvocationId(dim) +
         __spirv_BuiltInGlobalOffset(dim);
}
