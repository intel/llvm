//==---------- spirv_vars.h --- SPIRV variables --------------*- C++ -*-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBDEVICE_SPIRV_VARS_H
#define __LIBDEVICE_SPIRV_VARS_H

#include "device.h"
#include <cstddef>

#if defined(__NVPTX__) || defined(__AMDGCN__)
// For AMD/Cuda those symbols will be provided by libclc.
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_z();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_z();
DEVICE_EXTERNAL size_t __spirv_GlobalSize_x();
DEVICE_EXTERNAL size_t __spirv_GlobalSize_y();
DEVICE_EXTERNAL size_t __spirv_GlobalSize_z();
#endif // __NVPTX__ || __AMDGCN__

#if defined(__SPIR__) || defined(__SPIRV__)

#include <cstdint>

#define __SPIRV_VAR_QUALIFIERS EXTERN_C const
typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t __spirv_BuiltInGlobalLinearId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalSize;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInLocalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInNumWorkgroups;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupSize;

__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInNumSubgroups;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupId;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupSize;

__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

// FIXME: change DEVICE_EXTERNAL to static and rename the functions,
//        when #3311 is fixed.
//        These are just internal functions used within libdevice.
//        We must not intrude the __spirv "namespace", so we'd better
//        use names like getGlobalInvocationIdX.
//        Libdevice must not export these APIs either, but it currently
//        exports them due to DEVICE_EXTERNAL.
DEVICE_EXTERNAL inline size_t __spirv_GlobalInvocationId_x() {
  return __spirv_BuiltInGlobalInvocationId.x;
}
DEVICE_EXTERNAL inline size_t __spirv_GlobalInvocationId_y() {
  return __spirv_BuiltInGlobalInvocationId.y;
}
DEVICE_EXTERNAL inline size_t __spirv_GlobalInvocationId_z() {
  return __spirv_BuiltInGlobalInvocationId.z;
}

DEVICE_EXTERNAL inline size_t __spirv_LocalInvocationId_x() {
  return __spirv_BuiltInLocalInvocationId.x;
}
DEVICE_EXTERNAL inline size_t __spirv_LocalInvocationId_y() {
  return __spirv_BuiltInLocalInvocationId.y;
}
DEVICE_EXTERNAL inline size_t __spirv_LocalInvocationId_z() {
  return __spirv_BuiltInLocalInvocationId.z;
}

#if !defined(__SPIR__) && !defined(__SPIRV__)
const size_t_vec __spirv_BuiltInGlobalInvocationId{};
const size_t_vec __spirv_BuiltInLocalInvocationId{};
#endif // !__SPIR__ && !__SPIRV__

#endif // __SPIR__ || __SPIRV__
#endif // __LIBDEVICE_SPIRV_VARS_H
