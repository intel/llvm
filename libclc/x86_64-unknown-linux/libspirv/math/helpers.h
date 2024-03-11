#include "func.h"
#include "types.h"

#ifdef NO_CLANG_BUILTINS

#define GEN_UNARY_BUILTIN_T(NAME, TYPE)                                        \
  _CLC_OVERLOAD TYPE __##NAME##_helper(TYPE);                                  \
  _CLC_OVERLOAD TYPE __spirv_ocl_##NAME(TYPE n) { return __##NAME##_helper(n); }

#define GEN_TERNARY_BUILTIN_T(NAME, TYPE)                                      \
  _CLC_OVERLOAD TYPE __##NAME##_helper(TYPE, TYPE, TYPE);                      \
  _CLC_OVERLOAD TYPE __spirv_ocl_##NAME(TYPE a, TYPE b, TYPE c) {              \
    return __##NAME##_helper(a, b, c);                                         \
  }
#define GEN_UNARY_BUILTIN(NAME)                                                \
  GEN_UNARY_BUILTIN_T(NAME, float)                                             \
  GEN_UNARY_BUILTIN_T(NAME, double)

#define GEN_TERNARY_BUILTIN(NAME)                                              \
  GEN_TERNARY_BUILTIN_T(NAME, float)                                           \
  GEN_TERNARY_BUILTIN_T(NAME, double)

#else

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

#define GEN_UNARY_BUILTIN(NAME)                                                \
  _CLC_OVERLOAD float GETNAME(NAME)(float n) {                                 \
    return __builtin_##NAME##f(n);                                             \
  }                                                                            \
  _CLC_OVERLOAD double GETNAME(NAME)(double n) { return __builtin_##NAME(n); } \
  GEN_UNARY_VECTOR_BUILTIN_T(NAME, float)                                      \
  GEN_UNARY_VECTOR_BUILTIN_T(NAME, double)

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

#define GEN_TERNARY_BUILTIN(NAME)                                              \
  _CLC_OVERLOAD float GETNAME(NAME)(float n1, float n2, float n3) {            \
    return __builtin_##NAME##f(n1, n2, n3);                                    \
  }                                                                            \
  _CLC_OVERLOAD double GETNAME(NAME)(double n1, double n2, double n3) {        \
    return __builtin_##NAME(n1, n2, n3);                                       \
  }                                                                            \
  GEN_TERNARY_VECTOR_BUILTIN_T(NAME, float)                                    \
  GEN_TERNARY_VECTOR_BUILTIN_T(NAME, double)
#endif
