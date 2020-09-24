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

#ifdef __SPIR__

#include <cstddef>
#include <cstdint>

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
extern "C" const size_t_vec __spirv_BuiltInGlobalInvocationId;
extern "C" const size_t_vec __spirv_BuiltInLocalInvocationId;

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

#endif // __SPIR__
#endif // __LIBDEVICE_SPIRV_VARS_H
