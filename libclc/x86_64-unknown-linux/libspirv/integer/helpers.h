#include "func.h"

#define GEN_UNARY_BUILTIN_T(NAME, TYPE)                                        \
  _CLC_OVERLOAD TYPE __##NAME##_helper(TYPE);                                  \
  _CLC_OVERLOAD TYPE __spirv_ocl_##NAME(TYPE n) { return __##NAME##_helper(n); }

#define GEN_UNARY_BUILTIN(NAME)                                                \
  GEN_UNARY_BUILTIN_T(NAME, int)                                               \
  GEN_UNARY_BUILTIN_T(NAME, signed char)
