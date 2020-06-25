//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEFINE_RELATIONAL_UNARY_SCALAR(bool, _CLC_SPIRV_BUILTIN, _CLC_BUILTIN_IMPL,
                                    float)
_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, _CLC_SPIRV_BUILTIN, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_RELATIONAL_UNARY_SCALAR(bool, _CLC_SPIRV_BUILTIN, _CLC_BUILTIN_IMPL,
                                    double)
_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, _CLC_SPIRV_BUILTIN, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_RELATIONAL_UNARY_SCALAR(bool, _CLC_SPIRV_BUILTIN, _CLC_BUILTIN_IMPL,
                                    half)
_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(schar, _CLC_SPIRV_BUILTIN, half)

#endif
