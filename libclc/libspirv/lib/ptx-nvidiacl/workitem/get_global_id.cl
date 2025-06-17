//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

extern int __nvvm_reflect_ocl(constant char *);

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalInvocationId_x() {
  if (__nvvm_reflect_ocl("__CUDA_ID_QUERIES_FIT_IN_INT")) {
    return (uint)__spirv_WorkgroupId_x() * (uint)__spirv_WorkgroupSize_x() +
           (uint)__spirv_LocalInvocationId_x() + (uint)__spirv_GlobalOffset_x();
  }
  return __spirv_WorkgroupId_x() * __spirv_WorkgroupSize_x() +
         __spirv_LocalInvocationId_x() + __spirv_GlobalOffset_x();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalInvocationId_y() {
  if (__nvvm_reflect_ocl("__CUDA_ID_QUERIES_FIT_IN_INT")) {
    return (uint)__spirv_WorkgroupId_y() * (uint)__spirv_WorkgroupSize_y() +
           (uint)__spirv_LocalInvocationId_y() + (uint)__spirv_GlobalOffset_y();
  }
  return __spirv_WorkgroupId_y() * __spirv_WorkgroupSize_y() +
         __spirv_LocalInvocationId_y() + __spirv_GlobalOffset_y();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_GlobalInvocationId_z() {
  if (__nvvm_reflect_ocl("__CUDA_ID_QUERIES_FIT_IN_INT")) {
    return (uint)__spirv_WorkgroupId_z() * (uint)__spirv_WorkgroupSize_z() +
           (uint)__spirv_LocalInvocationId_z() + (uint)__spirv_GlobalOffset_z();
  }
  return __spirv_WorkgroupId_z() * __spirv_WorkgroupSize_z() +
         __spirv_LocalInvocationId_z() + __spirv_GlobalOffset_z();
}
