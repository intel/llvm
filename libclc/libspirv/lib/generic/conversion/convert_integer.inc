//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_GEN_S

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_S(__CLC_S_GENTYPE_SRC x) {
  return __CLC_IMPL_FUNCTION(x);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__CLC_FUNCTION_S_SAT(__CLC_S_GENTYPE_SRC x) {
  return __CLC_IMPL_FUNCTION_SAT(x);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__CLC_FUNCTION_UToS(__CLC_U_GENTYPE_SRC x) {
  return __CLC_IMPL_FUNCTION_SAT(x);
}

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE)                              \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION_S, MODE)(  \
      __CLC_S_GENTYPE_SRC x) {                                                 \
    return __CLC_FUNCTION_S(x);                                                \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(                          \
      __CLC_FUNCTION_S_SAT, MODE)(__CLC_S_GENTYPE_SRC x) {                     \
    return __CLC_FUNCTION_S_SAT(x);                                            \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(                          \
      __CLC_FUNCTION_UToS, MODE)(__CLC_U_GENTYPE_SRC x) {                      \
    return __CLC_FUNCTION_UToS(x);                                             \
  }

#else // __CLC_GEN_S

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_U(__CLC_U_GENTYPE_SRC x) {
  return __CLC_IMPL_FUNCTION(x);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__CLC_FUNCTION_U_SAT(__CLC_U_GENTYPE_SRC x) {
  return __CLC_IMPL_FUNCTION_SAT(x);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__CLC_FUNCTION_SToU(__CLC_S_GENTYPE_SRC x) {
  return __CLC_IMPL_FUNCTION_SAT(x);
}

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE)                              \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION_U, MODE)(  \
      __CLC_U_GENTYPE_SRC x) {                                                 \
    return __CLC_FUNCTION_U(x);                                                \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(                          \
      __CLC_FUNCTION_U_SAT, MODE)(__CLC_U_GENTYPE_SRC x) {                     \
    return __CLC_FUNCTION_U_SAT(x);                                            \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(                          \
      __CLC_FUNCTION_SToU, MODE)(__CLC_S_GENTYPE_SRC x) {                      \
    return __CLC_FUNCTION_SToU(x);                                             \
  }

#endif // __CLC_GEN_S

__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rte)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtz)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtp)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtn)

#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE
