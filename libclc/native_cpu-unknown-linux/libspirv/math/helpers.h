#include "func.h"
#include "types.h"

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef IS_NATIVE
#define GETNAME(ID) __spirv_ocl_##ID
#else
#define GETNAME(ID) __spirv_ocl_native_##ID
#endif

// Todo: fabs is the only builtin whose vector version is not named
// __builtin_elementwise_##NAME
#ifndef IS_FABS
#define GEN_UNARY_VECTOR_BUILTIN(NAME, TYPE, NUM)                              \
  _CLC_OVERLOAD TYPE##NUM GETNAME(NAME)(TYPE##NUM n) {                         \
    return __builtin_elementwise_##NAME(n);                                    \
  }
#else
#define GEN_UNARY_VECTOR_BUILTIN(NAME, TYPE, NUM)                              \
  _CLC_OVERLOAD TYPE##NUM GETNAME(NAME)(TYPE##NUM n) {                         \
    return __builtin_elementwise_abs(n);                                       \
  }
#endif

#define GEN_UNARY_VECTOR_BUILTIN_T(NAME, TYPE)                                 \
  GEN_UNARY_VECTOR_BUILTIN(NAME, TYPE, 2)                                      \
  GEN_UNARY_VECTOR_BUILTIN(NAME, TYPE, 3)                                      \
  GEN_UNARY_VECTOR_BUILTIN(NAME, TYPE, 4)                                      \
  GEN_UNARY_VECTOR_BUILTIN(NAME, TYPE, 8)                                      \
  GEN_UNARY_VECTOR_BUILTIN(NAME, TYPE, 16)

#define GEN_UNARY_BUILTIN_T(NAME, TYPE)                                        \
  _CLC_OVERLOAD TYPE GETNAME(NAME)(TYPE n) { return __builtin_##NAME(n); }

#if defined(cl_khr_fp16)
#define GEN_UNARY_FP16(NAME)                                                   \
  GEN_UNARY_BUILTIN_T(NAME, half)                                              \
  GEN_UNARY_VECTOR_BUILTIN_T(NAME, half)
#else
#define GEN_UNARY_FP16(NAME)
#endif

#if defined(cl_khr_fp64)
#define GEN_UNARY_FP64(NAME)                                                   \
  GEN_UNARY_BUILTIN_T(NAME, double)                                            \
  GEN_UNARY_VECTOR_BUILTIN_T(NAME, double)
#else
#define GEN_UNARY_FP64(NAME)
#endif

#define GEN_UNARY_BUILTIN(NAME)                                                \
  GEN_UNARY_BUILTIN_T(NAME, float)                                             \
  GEN_UNARY_VECTOR_BUILTIN_T(NAME, float)                                      \
  GEN_UNARY_FP16(NAME)                                                         \
  GEN_UNARY_FP64(NAME)

#define GEN_TERNARY_VECTOR_BUILTIN(NAME, TYPE, NUM)                            \
  _CLC_OVERLOAD TYPE##NUM GETNAME(NAME)(TYPE##NUM n1, TYPE##NUM n2,            \
                                        TYPE##NUM n3) {                        \
    return __builtin_elementwise_##NAME(n1, n2, n3);                           \
  }

#define GEN_TERNARY_VECTOR_BUILTIN_T(NAME, TYPE)                               \
  GEN_TERNARY_VECTOR_BUILTIN(NAME, TYPE, 2)                                    \
  GEN_TERNARY_VECTOR_BUILTIN(NAME, TYPE, 3)                                    \
  GEN_TERNARY_VECTOR_BUILTIN(NAME, TYPE, 4)                                    \
  GEN_TERNARY_VECTOR_BUILTIN(NAME, TYPE, 8)                                    \
  GEN_TERNARY_VECTOR_BUILTIN(NAME, TYPE, 16)

#define GEN_TERNARY_BUILTIN_T(NAME, TYPE)                                      \
  _CLC_OVERLOAD TYPE GETNAME(NAME)(TYPE n1, TYPE n2, TYPE n3) {                \
    return __builtin_##NAME(n1, n2, n3);                                       \
  }

#if defined(cl_khr_fp16)
#define GEN_TERNARY_FP16(NAME)                                                 \
  GEN_TERNARY_BUILTIN_T(NAME, half)                                            \
  GEN_TERNARY_VECTOR_BUILTIN_T(NAME, half)
#else
#define GEN_TERNARY_FP16(NAME)
#endif

#if defined(cl_khr_fp64)
#define GEN_TERNARY_FP64(NAME)                                                 \
  GEN_TERNARY_BUILTIN_T(NAME, double)                                          \
  GEN_TERNARY_VECTOR_BUILTIN_T(NAME, double)
#else
#define GEN_TERNARY_FP64(NAME)
#endif

#define GEN_TERNARY_BUILTIN(NAME)                                              \
  GEN_TERNARY_BUILTIN_T(NAME, float)                                           \
  GEN_TERNARY_VECTOR_BUILTIN_T(NAME, float)                                    \
  GEN_TERNARY_FP16(NAME)                                                       \
  GEN_TERNARY_FP64(NAME)
