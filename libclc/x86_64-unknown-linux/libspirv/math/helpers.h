#include "func.h"

#define GEN_UNARY_BUILTIN_T(NAME, TYPE) \
_CLC_OVERLOAD TYPE __##NAME##_helper(TYPE); \
_CLC_OVERLOAD TYPE __spirv_ocl_##NAME(TYPE n) { \
  return __##NAME##_helper(n); \
} 

#define GEN_TERNARY_BUILTIN_T(NAME, TYPE) \
_CLC_OVERLOAD TYPE __##NAME##_helper(TYPE, TYPE, TYPE); \
_CLC_OVERLOAD TYPE __spirv_ocl_##NAME(TYPE a, TYPE b, TYPE c) { \
  return __##NAME##_helper(a, b, c); \
} 


#define GEN_UNARY_BUILTIN(NAME) \
  GEN_UNARY_BUILTIN_T(NAME, float) \
  GEN_UNARY_BUILTIN_T(NAME, double) 


#define GEN_TERNARY_BUILTIN(NAME) \
GEN_TERNARY_BUILTIN_T(NAME, float) \
GEN_TERNARY_BUILTIN_T(NAME, double) \
