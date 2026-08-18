//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#pragma OPENCL EXTENSION __cl_clang_function_scope_local_variables : enable

__attribute__((always_inline)) local char* __clc__get_group_scratch_char() {
  local char a[32];
  return a;
}

__attribute__((always_inline)) local bool* __clc__get_group_scratch_bool() {
  return (local bool*)__clc__get_group_scratch_char();
}

__attribute__((always_inline)) local uchar* __clc__get_group_scratch_uchar() {
  return (local uchar*)__clc__get_group_scratch_char();
}

__attribute__((always_inline)) local short* __clc__get_group_scratch_short() {
  local short a[32];
  return a;
}

__attribute__((always_inline)) local ushort* __clc__get_group_scratch_ushort() {
  return (local ushort*)__clc__get_group_scratch_short();
}

__attribute__((always_inline)) local int* __clc__get_group_scratch_int() {
  local int a[32];
  return a;
}

__attribute__((always_inline)) local uint* __clc__get_group_scratch_uint() {
  return (local uint*)__clc__get_group_scratch_int();
}

__attribute__((always_inline)) local long* __clc__get_group_scratch_long() {
  local long a[32];
  return a;
}

__attribute__((always_inline)) local ulong* __clc__get_group_scratch_ulong() {
  return (local ulong*)__clc__get_group_scratch_long();
}

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((always_inline)) local half* __clc__get_group_scratch_half() {
  return (local half*)__clc__get_group_scratch_short();
}

#endif // cl_khr_fp16

__attribute__((always_inline)) local float* __clc__get_group_scratch_float() {
  return (local float*)__clc__get_group_scratch_int();
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__attribute__((always_inline)) local double* __clc__get_group_scratch_double() {
  return (local double*)__clc__get_group_scratch_long();
}

#endif // cl_khr_fp64
