//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _SPIRV_VLOAD_DECL(SUFFIX, MEM_TYPE, VEC_TYPE, WIDTH, ADDR_SPACE) \
  _CLC_OVERLOAD _CLC_DECL VEC_TYPE __spirv_ocl_vload##SUFFIXn__R##VEC_TYPE##WIDTH( \
      size_t offset, const ADDR_SPACE MEM_TYPE *x);

#define _SPIRV_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, ADDR_SPACE) \
  _SPIRV_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##2, 2, ADDR_SPACE) \
  _SPIRV_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##3, 3, ADDR_SPACE) \
  _SPIRV_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##4, 4, ADDR_SPACE) \
  _SPIRV_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##8, 8, ADDR_SPACE) \
  _SPIRV_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE##16, 16, ADDR_SPACE)

#define _SPIRV_VECTOR_VLOAD_PRIM3(SUFFIX, MEM_TYPE, PRIM_TYPE) \
  _SPIRV_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __private) \
  _SPIRV_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __local) \
  _SPIRV_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __constant) \
  _SPIRV_VECTOR_VLOAD_DECL(SUFFIX, MEM_TYPE, PRIM_TYPE, __global)

#define _SPIRV_VECTOR_VLOAD_PRIM1(PRIM_TYPE) \
  _SPIRV_VECTOR_VLOAD_PRIM3(, PRIM_TYPE, PRIM_TYPE)

// Declare vector load prototypes
_SPIRV_VECTOR_VLOAD_PRIM1(char)
_SPIRV_VECTOR_VLOAD_PRIM1(uchar)
_SPIRV_VECTOR_VLOAD_PRIM1(short)
_SPIRV_VECTOR_VLOAD_PRIM1(ushort)
_SPIRV_VECTOR_VLOAD_PRIM1(int)
_SPIRV_VECTOR_VLOAD_PRIM1(uint)
_SPIRV_VECTOR_VLOAD_PRIM1(long)
_SPIRV_VECTOR_VLOAD_PRIM1(ulong)
_SPIRV_VECTOR_VLOAD_PRIM1(float)
_SPIRV_VECTOR_VLOAD_PRIM3(_half, half, float)
// Use suffix to declare aligned vloada_halfN
_SPIRV_VECTOR_VLOAD_PRIM3(a_half, half, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
  _SPIRV_VECTOR_VLOAD_PRIM1(double)
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16: enable
  _SPIRV_VECTOR_VLOAD_PRIM1(half)
#endif

// Scalar __spirv_ocl_vload_half__Rfloat also needs to be declared
_SPIRV_VLOAD_DECL(_half, half, float, , __constant)
_SPIRV_VLOAD_DECL(_half, half, float, , __global)
_SPIRV_VLOAD_DECL(_half, half, float, , __local)
_SPIRV_VLOAD_DECL(_half, half, float, , __private)

// Scalar __spirv_ocl_vloada_half__Rfloat is not part of the specs but CTS expects it
_SPIRV_VLOAD_DECL(a_half, half, float, , __constant)
_SPIRV_VLOAD_DECL(a_half, half, float, , __global)
_SPIRV_VLOAD_DECL(a_half, half, float, , __local)
_SPIRV_VLOAD_DECL(a_half, half, float, , __private)

#undef _SPIRV_VLOAD_DECL
#undef _SPIRV_VECTOR_VLOAD_DECL
#undef _SPIRV_VECTOR_VLOAD_PRIM3
#undef _SPIRV_VECTOR_VLOAD_PRIM1
