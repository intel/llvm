//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#define VLOAD_VECTORIZE(RTYPE, PRIM_TYPE, ADDR_SPACE)                          \
  typedef PRIM_TYPE less_aligned_##ADDR_SPACE##PRIM_TYPE                       \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE __spirv_ocl_vload_R##RTYPE(                 \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *((                                                                 \
        const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE *)(&x[offset])); \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##2 less_aligned_##ADDR_SPACE##PRIM_TYPE##2                 \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##2 __spirv_ocl_vloadn_R##RTYPE##2(          \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2         \
                  *)(&x[2 * offset]));                                         \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##3 less_aligned_##ADDR_SPACE##PRIM_TYPE##3                 \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##3 __spirv_ocl_vloadn_R##RTYPE##3(          \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    PRIM_TYPE##2 vec =                                                         \
        *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2            \
               *)(&x[3 * offset]));                                            \
    return (PRIM_TYPE##3)(vec.s0, vec.s1, x[offset * 3 + 2]);                  \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##4 less_aligned_##ADDR_SPACE##PRIM_TYPE##4                 \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##4 __spirv_ocl_vloadn_R##RTYPE##4(          \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##4         \
                  *)(&x[4 * offset]));                                         \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##8 less_aligned_##ADDR_SPACE##PRIM_TYPE##8                 \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##8 __spirv_ocl_vloadn_R##RTYPE##8(          \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##8         \
                  *)(&x[8 * offset]));                                         \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##16 less_aligned_##ADDR_SPACE##PRIM_TYPE##16               \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##16 __spirv_ocl_vloadn_R##RTYPE##16(        \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##16        \
                  *)(&x[16 * offset]));                                        \
  }

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define VLOAD_VECTORIZE_GENERIC VLOAD_VECTORIZE
#else
// The generic address space isn't available, so make the macro do nothing
#define VLOAD_VECTORIZE_GENERIC(X,Y,Z)
#endif

#define VLOAD_ADDR_SPACES_IMPL(__CLC_RET_GENTYPE, __CLC_SCALAR_GENTYPE)        \
  VLOAD_VECTORIZE(__CLC_RET_GENTYPE, __CLC_SCALAR_GENTYPE, __private)          \
  VLOAD_VECTORIZE(__CLC_RET_GENTYPE, __CLC_SCALAR_GENTYPE, __local)            \
  VLOAD_VECTORIZE(__CLC_RET_GENTYPE, __CLC_SCALAR_GENTYPE, __constant)         \
  VLOAD_VECTORIZE(__CLC_RET_GENTYPE, __CLC_SCALAR_GENTYPE, __global)           \
  VLOAD_VECTORIZE_GENERIC(__CLC_RET_GENTYPE, __CLC_SCALAR_GENTYPE, __generic)

#define VLOAD_ADDR_SPACES(__CLC_SCALAR_GENTYPE)                                \
  VLOAD_ADDR_SPACES_IMPL(__CLC_SCALAR_GENTYPE, __CLC_SCALAR_GENTYPE)

VLOAD_ADDR_SPACES_IMPL(char, schar)

#define VLOAD_TYPES()                                                          \
  VLOAD_ADDR_SPACES(uchar)                                                     \
  VLOAD_ADDR_SPACES(short)                                                     \
  VLOAD_ADDR_SPACES(ushort)                                                    \
  VLOAD_ADDR_SPACES(int)                                                       \
  VLOAD_ADDR_SPACES(uint)                                                      \
  VLOAD_ADDR_SPACES(long)                                                      \
  VLOAD_ADDR_SPACES(ulong)                                                     \
  VLOAD_ADDR_SPACES(float)

VLOAD_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
VLOAD_ADDR_SPACES(double)
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
VLOAD_ADDR_SPACES(half)
#endif

/* vload_half are legal even without cl_khr_fp16 */
/* no vload_half for double */
#if __clang_major__ < 6
float __clc_vload_half_float_helper__constant(const __constant half *);
float __clc_vload_half_float_helper__global(const __global half *);
float __clc_vload_half_float_helper__local(const __local half *);
float __clc_vload_half_float_helper__private(const __private half *);

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
float __clc_vload_half_float_helper__generic(const __generic half *);
#endif

#define VEC_LOAD1(val, AS)                                                     \
  val = __clc_vload_half_float_helper##AS(&mem[offset++]);
#else
#define VEC_LOAD1(val, AS) val = __builtin_load_halff(&mem[offset++]);
#endif

#define VEC_LOAD2(val, AS)                                                     \
  VEC_LOAD1(val.lo, AS)                                                        \
  VEC_LOAD1(val.hi, AS)
#define VEC_LOAD3(val, AS)                                                     \
  VEC_LOAD1(val.s0, AS)                                                        \
  VEC_LOAD1(val.s1, AS)                                                        \
  VEC_LOAD1(val.s2, AS)
#define VEC_LOAD4(val, AS)                                                     \
  VEC_LOAD2(val.lo, AS)                                                        \
  VEC_LOAD2(val.hi, AS)
#define VEC_LOAD8(val, AS)                                                     \
  VEC_LOAD4(val.lo, AS)                                                        \
  VEC_LOAD4(val.hi, AS)
#define VEC_LOAD16(val, AS)                                                    \
  VEC_LOAD8(val.lo, AS)                                                        \
  VEC_LOAD8(val.hi, AS)

#define VLOAD_HALF_VEC_IMPL(VEC_SIZE, OFFSET_SIZE, AS)                         \
  _CLC_OVERLOAD _CLC_DEF float##VEC_SIZE                                       \
      __spirv_ocl_vload_halfn_Rfloat##VEC_SIZE(size_t offset,                  \
                                               const AS half *mem) {           \
    offset *= VEC_SIZE;                                                        \
    float##VEC_SIZE __tmp;                                                     \
    VEC_LOAD##VEC_SIZE(__tmp, AS) return __tmp;                                \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF float##VEC_SIZE                                       \
      __spirv_ocl_vloada_halfn_Rfloat##VEC_SIZE(size_t offset,                 \
                                                const AS half *mem) {          \
    offset *= OFFSET_SIZE;                                                     \
    float##VEC_SIZE __tmp;                                                     \
    VEC_LOAD##VEC_SIZE(__tmp, AS) return __tmp;                                \
  }

#define VLOAD_HALF_IMPL(AS)                                                    \
  _CLC_OVERLOAD _CLC_DEF float __spirv_ocl_vload_half(size_t offset,           \
                                                      const AS half *mem) {    \
    float __tmp;                                                               \
    VEC_LOAD1(__tmp, AS) return __tmp;                                         \
  }

#define GEN_VLOAD_HALF(AS)                                                     \
  VLOAD_HALF_IMPL(AS)                                                          \
  VLOAD_HALF_VEC_IMPL(2, 2, AS)                                                \
  VLOAD_HALF_VEC_IMPL(3, 4, AS)                                                \
  VLOAD_HALF_VEC_IMPL(4, 4, AS)                                                \
  VLOAD_HALF_VEC_IMPL(8, 8, AS)                                                \
  VLOAD_HALF_VEC_IMPL(16, 16, AS)

GEN_VLOAD_HALF(__private)
GEN_VLOAD_HALF(__global)
GEN_VLOAD_HALF(__local)
GEN_VLOAD_HALF(__constant)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
GEN_VLOAD_HALF(__generic)
#endif

#undef VLOAD_HALF_IMPL
#undef VLOAD_HALF_VEC_IMPL
#undef GEN_VLOAD_HALF
#undef VEC_LOAD16
#undef VEC_LOAD8
#undef VEC_LOAD4
#undef VEC_LOAD3
#undef VEC_LOAD2
#undef VEC_LOAD1
#undef VLOAD_TYPES
#undef VLOAD_ADDR_SPACES
#undef VLOAD_VECTORIZE
#undef VLOAD_VECTORIZE_GENERIC
#undef VLOAD_VECTORIZE
