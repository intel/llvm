//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_SCALAR_GENTYPE float
#define __SPIRV_FPSIZE 32

#define __SPIRV_GENTYPE float
#define __SPIRV_INTN int
#define __SPIRV_SCALAR
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN
#undef __SPIRV_SCALAR

#define __SPIRV_GENTYPE float2
#define __SPIRV_INTN int2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE float3
#define __SPIRV_INTN int3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE float4
#define __SPIRV_INTN int4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE float8
#define __SPIRV_INTN int8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE float16
#define __SPIRV_INTN int16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#undef __SPIRV_FPSIZE
#undef __SPIRV_SCALAR_GENTYPE

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __SPIRV_SCALAR_GENTYPE double
#define __SPIRV_FPSIZE 64

#define __SPIRV_SCALAR
#define __SPIRV_GENTYPE double
#define __SPIRV_INTN int
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN
#undef __SPIRV_SCALAR

#define __SPIRV_GENTYPE double2
#define __SPIRV_INTN int2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE double3
#define __SPIRV_INTN int3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE double4
#define __SPIRV_INTN int4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE double8
#define __SPIRV_INTN int8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE double16
#define __SPIRV_INTN int16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#undef __SPIRV_FPSIZE
#undef __SPIRV_SCALAR_GENTYPE
#endif
#endif

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __SPIRV_SCALAR_GENTYPE half
#define __SPIRV_FPSIZE 16

#define __SPIRV_SCALAR
#define __SPIRV_GENTYPE half
#define __SPIRV_INTN int
#include __SPIRV_BODY
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN
#undef __SPIRV_SCALAR

#define __SPIRV_GENTYPE half2
#define __SPIRV_INTN int2
#define __SPIRV_VECSIZE 2
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE half3
#define __SPIRV_INTN int3
#define __SPIRV_VECSIZE 3
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE half4
#define __SPIRV_INTN int4
#define __SPIRV_VECSIZE 4
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE half8
#define __SPIRV_INTN int8
#define __SPIRV_VECSIZE 8
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#define __SPIRV_GENTYPE half16
#define __SPIRV_INTN int16
#define __SPIRV_VECSIZE 16
#include __SPIRV_BODY
#undef __SPIRV_VECSIZE
#undef __SPIRV_GENTYPE
#undef __SPIRV_INTN

#undef __SPIRV_FPSIZE
#undef __SPIRV_SCALAR_GENTYPE
#endif
#endif

#undef __SPIRV_BODY
