//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_FLOATN float
#define __SPIRV_INTN int
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN float2
#define __SPIRV_INTN int2
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN float3
#define __SPIRV_INTN int3
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN float4
#define __SPIRV_INTN int4
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN float8
#define __SPIRV_INTN int8
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN float16
#define __SPIRV_INTN int16
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#undef __SPIRV_FLOAT
#undef __SPIRV_INT

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __SPIRV_FLOATN double
#define __SPIRV_INTN int
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN double2
#define __SPIRV_INTN long2
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN double3
#define __SPIRV_INTN long3
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN double4
#define __SPIRV_INTN long4
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN double8
#define __SPIRV_INTN long8
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN double16
#define __SPIRV_INTN long16
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __SPIRV_FLOATN half
#define __SPIRV_INTN int
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN half2
#define __SPIRV_INTN short2
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN half3
#define __SPIRV_INTN short3
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN half4
#define __SPIRV_INTN short4
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN half8
#define __SPIRV_INTN short8
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#define __SPIRV_FLOATN half16
#define __SPIRV_INTN short16
#include __SPIRV_BODY
#undef __SPIRV_INTN
#undef __SPIRV_FLOATN

#endif

#undef __SPIRV_BODY
