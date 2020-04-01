//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_FLOAT float
#define __SPIRV_FPSIZE 32

#define __SPIRV_FLOATN float
#define __SPIRV_SCALAR
#include __SPIRV_BODY
#undef __SPIRV_FLOATN
#undef __SPIRV_SCALAR

#define __SPIRV_FLOATN float2
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN float3
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN float4
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#undef __SPIRV_FLOAT
#undef __SPIRV_FPSIZE

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __SPIRV_FLOAT double
#define __SPIRV_FPSIZE 64

#define __SPIRV_FLOATN double
#define __SPIRV_SCALAR
#include __SPIRV_BODY
#undef __SPIRV_FLOATN
#undef __SPIRV_SCALAR

#define __SPIRV_FLOATN double2
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN double3
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN double4
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#undef __SPIRV_FLOAT
#undef __SPIRV_FPSIZE

#endif
#endif

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __SPIRV_FLOAT half
#define __SPIRV_FPSIZE 16

#define __SPIRV_FLOATN half
#define __SPIRV_SCALAR
#include __SPIRV_BODY
#undef __SPIRV_FLOATN
#undef __SPIRV_SCALAR

#define __SPIRV_FLOATN half2
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN half3
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN half4
#include __SPIRV_BODY
#undef __SPIRV_FLOATN

#undef __SPIRV_FLOAT
#undef __SPIRV_FPSIZE

#endif
#endif

#undef __SPIRV_BODY
