//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEFINE_RELATIONAL_BINARY_SCALAR(bool, _CLC_SPIRV_BUILTIN,
                                     _CLC_BUILTIN_IMPL, float, float)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(schar, _CLC_SPIRV_BUILTIN, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_RELATIONAL_BINARY_SCALAR(bool, _CLC_SPIRV_BUILTIN,
                                     _CLC_BUILTIN_IMPL, double, double)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(schar, _CLC_SPIRV_BUILTIN, double, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_RELATIONAL_BINARY_SCALAR(bool, _CLC_SPIRV_BUILTIN,
                                     _CLC_BUILTIN_IMPL, half, half)
_CLC_DEFINE_RELATIONAL_BINARY_VEC_ALL(schar, _CLC_SPIRV_BUILTIN, half, half)

#endif
