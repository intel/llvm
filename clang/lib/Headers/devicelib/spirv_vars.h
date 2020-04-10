//==---------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#ifndef __SPIRV_VARS_H__
#define __SPIRV_VARS_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define __EXTERN_C extern "C"
#else
#define __EXTERN_C
#endif

#ifdef __SYCL_DEVICE_ONLY__

#ifdef __SYCL_NVPTX__

SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_x();
SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_y();
SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_z();

SYCL_EXTERNAL size_t __spirv_GlobalSize_x();
SYCL_EXTERNAL size_t __spirv_GlobalSize_y();
SYCL_EXTERNAL size_t __spirv_GlobalSize_z();

SYCL_EXTERNAL size_t __spirv_GlobalOffset_x();
SYCL_EXTERNAL size_t __spirv_GlobalOffset_y();
SYCL_EXTERNAL size_t __spirv_GlobalOffset_z();

SYCL_EXTERNAL size_t __spirv_NumWorkgroups_x();
SYCL_EXTERNAL size_t __spirv_NumWorkgroups_y();
SYCL_EXTERNAL size_t __spirv_NumWorkgroups_z();

SYCL_EXTERNAL size_t __spirv_WorkgroupSize_x();
SYCL_EXTERNAL size_t __spirv_WorkgroupSize_y();
SYCL_EXTERNAL size_t __spirv_WorkgroupSize_z();

SYCL_EXTERNAL size_t __spirv_WorkgroupId_x();
SYCL_EXTERNAL size_t __spirv_WorkgroupId_y();
SYCL_EXTERNAL size_t __spirv_WorkgroupId_z();

SYCL_EXTERNAL size_t __spirv_LocalInvocationId_x();
SYCL_EXTERNAL size_t __spirv_LocalInvocationId_y();
SYCL_EXTERNAL size_t __spirv_LocalInvocationId_z();

#else // __SYCL_NVPTX__

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
__EXTERN_C const __attribute__((opencl_constant))
size_t_vec __spirv_BuiltInGlobalInvocationId;
__EXTERN_C const __attribute__((opencl_constant))
size_t_vec __spirv_BuiltInGlobalSize;
__EXTERN_C const __attribute__((opencl_constant))
size_t_vec __spirv_BuiltInGlobalOffset;
__EXTERN_C const __attribute__((opencl_constant))
size_t_vec __spirv_BuiltInNumWorkgroups;
__EXTERN_C const __attribute__((opencl_constant))
size_t_vec __spirv_BuiltInWorkgroupSize;
__EXTERN_C const __attribute__((opencl_constant))
size_t_vec __spirv_BuiltInWorkgroupId;
__EXTERN_C const __attribute__((opencl_constant))
size_t_vec __spirv_BuiltInLocalInvocationId;

SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_x() {
  return __spirv_BuiltInGlobalInvocationId.x;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_y() {
  return __spirv_BuiltInGlobalInvocationId.y;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_z() {
  return __spirv_BuiltInGlobalInvocationId.z;
}

SYCL_EXTERNAL inline size_t __spirv_GlobalSize_x() {
  return __spirv_BuiltInGlobalSize.x;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalSize_y() {
  return __spirv_BuiltInGlobalSize.y;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalSize_z() {
  return __spirv_BuiltInGlobalSize.z;
}

SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_x() {
  return __spirv_BuiltInGlobalOffset.x;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_y() {
  return __spirv_BuiltInGlobalOffset.y;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_z() {
  return __spirv_BuiltInGlobalOffset.z;
}

SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_x() {
  return __spirv_BuiltInNumWorkgroups.x;
}
SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_y() {
  return __spirv_BuiltInNumWorkgroups.y;
}
SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_z() {
  return __spirv_BuiltInNumWorkgroups.z;
}

SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_x() {
  return __spirv_BuiltInWorkgroupSize.x;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_y() {
  return __spirv_BuiltInWorkgroupSize.y;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_z() {
  return __spirv_BuiltInWorkgroupSize.z;
}

SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_x() {
  return __spirv_BuiltInWorkgroupId.x;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_y() {
  return __spirv_BuiltInWorkgroupId.y;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_z() {
  return __spirv_BuiltInWorkgroupId.z;
}

SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_x() {
  return __spirv_BuiltInLocalInvocationId.x;
}
SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_y() {
  return __spirv_BuiltInLocalInvocationId.y;
}
SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_z() {
  return __spirv_BuiltInLocalInvocationId.z;
}

#endif // __SYCL_NVPTX__

#endif // __SYCL_DEVICE_ONLY__

#undef __EXTERN_C

#endif // __SPIRV_VARS_H__
