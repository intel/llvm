//==-- helper_macros.hpp -- Utility macros to implement sycl builtins ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Usage:
//   #define HANDLE_TYPE(INVARIANT_ARG1, INVARIANT_ARG2, TYPE) ...
//   FOR_EACH2(HANDLE_TYPE, A1, A2, TYPE1, TYPE2, ...)
// it will expand into
//   HANDLE_TYPE(A1, A2, TYPE1)
//   HANDLE_TYPE(A1, A2, TYPE2)
//   ...
// Number of "invariant" arguments determines the numeric suffix for the
// FOR_EACHN. Only 0-4 are currently supported, and up to 15 types at most.
#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, \
                  _15, NAME, ...)                                              \
  NAME
#define FOR_EACH4_A1(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1)          \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG1)
#define FOR_EACH4_A2(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2)    \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG1)                              \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG2)
#define FOR_EACH4_A3(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2,    \
                     ARG3)                                                     \
  FOR_EACH4_A2(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2)          \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG3)
#define FOR_EACH4_A4(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2,    \
                     ARG3, ARG4)                                               \
  FOR_EACH4_A3(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2, ARG3)    \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG4)
#define FOR_EACH4_A5(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2,    \
                     ARG3, ARG4, ARG5)                                         \
  FOR_EACH4_A4(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2, ARG3,    \
               ARG4)                                                           \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG5)
#define FOR_EACH4_A6(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2,    \
                     ARG3, ARG4, ARG5, ARG6)                                   \
  FOR_EACH4_A5(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2, ARG3,    \
               ARG4, ARG5)                                                     \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG6)
#define FOR_EACH4_A7(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2,    \
                     ARG3, ARG4, ARG5, ARG6, ARG7)                             \
  FOR_EACH4_A6(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2, ARG3,    \
               ARG4, ARG5, ARG6)                                               \
  BASE_CASE(FIXED1, FIXED2, FIXED3, FIXED4, ARG7)
#define FOR_EACH4_A11(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2,   \
                      ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11)  \
  FOR_EACH4_A7(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2, ARG3,    \
               ARG4, ARG5, ARG6, ARG7)                                         \
  FOR_EACH4_A4(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG8, ARG9, ARG10,   \
               ARG11)
#define FOR_EACH4_A14(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2,   \
                      ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11,  \
                      ARG12, ARG13, ARG14)                                     \
  FOR_EACH4_A11(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG1, ARG2, ARG3,   \
                ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11)              \
  FOR_EACH4_A3(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ARG12, ARG13, ARG14)

#define FOR_EACH4(BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, ...)              \
  GET_MACRO(__VA_ARGS__, FOR_EACH4##_A15, FOR_EACH4##_A14, FOR_EACH4##_A13,    \
            FOR_EACH4##_A12, FOR_EACH4##_A11, FOR_EACH4##_A10, FOR_EACH4##_A9, \
            FOR_EACH4##_A8, FOR_EACH4##_A7, FOR_EACH4##_A6, FOR_EACH4##_A5,    \
            FOR_EACH4##_A4, FOR_EACH4##_A3, FOR_EACH4##_A2, FOR_EACH4##_A1,    \
            _0, )                                                              \
  (BASE_CASE, FIXED1, FIXED2, FIXED3, FIXED4, __VA_ARGS__)

#define FOR_EACH3_BASE(BASE_CASE, FIXED1, FIXED2, FIXED3, ARG1)                \
  BASE_CASE(FIXED1, FIXED2, FIXED3, ARG1)
#define FOR_EACH3(BASE_CASE, FIXED1, FIXED2, FIXED3, ...)                      \
  FOR_EACH4(FOR_EACH3_BASE, BASE_CASE, FIXED1, FIXED2, FIXED3, __VA_ARGS__)

#define FOR_EACH2_BASE(BASE_CASE, FIXED1, FIXED2, ARG1)                        \
  BASE_CASE(FIXED1, FIXED2, ARG1)
#define FOR_EACH2(BASE_CASE, FIXED1, FIXED2, ...)                              \
  FOR_EACH3(FOR_EACH2_BASE, BASE_CASE, FIXED1, FIXED2, __VA_ARGS__)

#define FOR_EACH1_BASE(BASE_CASE, FIXED1, ARG1) BASE_CASE(FIXED1, ARG1)
#define FOR_EACH1(BASE_CASE, FIXED1, ...)                                      \
  FOR_EACH2(FOR_EACH1_BASE, BASE_CASE, FIXED1, __VA_ARGS__)

#define FOR_EACH0_BASE(BASE_CASE, ARG1) BASE_CASE(ARG1)
#define FOR_EACH0(BASE_CASE, ...)                                              \
  FOR_EACH1(FOR_EACH0_BASE, BASE_CASE, __VA_ARGS__)

// Some helpers to unify implementation between different numbers of template
// types.

#define ONE_ARG_TYPENAME_TYPE typename T0
#define TWO_ARGS_TYPENAME_TYPE typename T0, typename T1
#define THREE_ARGS_TYPENAME_TYPE typename T0, typename T1, typename T2

#define ONE_ARG_TEMPLATE_TYPE T0
#define TWO_ARGS_TEMPLATE_TYPE T0, T1
#define THREE_ARGS_TEMPLATE_TYPE T0, T1, T2

#define ONE_ARG_TEMPLATE_TYPE_ARG T0 x
#define TWO_ARGS_TEMPLATE_TYPE_ARG T0 x, T1 y
#define THREE_ARGS_TEMPLATE_TYPE_ARG T0 x, T1 y, T2 z

#define ONE_ARG_TEMPLATE_TYPE_ARG_REF T0 &x
#define TWO_ARGS_TEMPLATE_TYPE_ARG_REF T0 &x, T1 &y
#define THREE_ARGS_TEMPLATE_TYPE_ARG_REF T0 &x, T1 &y, T2 &z

#define ONE_ARG_ARG x
#define TWO_ARGS_ARG x, y
#define THREE_ARGS_ARG x, y, z

#define ONE_ARG_SIMPLIFIED_ARG                                                 \
  simplify_if_swizzle_t<T0> { x }
#define TWO_ARGS_SIMPLIFIED_ARG                                                \
  simplify_if_swizzle_t<T0>{x}, simplify_if_swizzle_t<T1> { y }
#define THREE_ARGS_SIMPLIFIED_ARG                                              \
  simplify_if_swizzle_t<T0>{x}, simplify_if_swizzle_t<T1>{y},                  \
      simplify_if_swizzle_t<T2> {                                              \
    z                                                                          \
  }

#define TWO_ARGS_ARG_ROTATED y, x
#define THREE_ARGS_ARG_ROTATED z, x, y

#define ONE_ARG_CONVERTED_ARG detail::builtins::convert_arg(x)
#define TWO_ARGS_CONVERTED_ARG                                                 \
  detail::builtins::convert_arg(x), detail::builtins::convert_arg(y)
#define THREE_ARGS_CONVERTED_ARG                                               \
  detail::builtins::convert_arg(x), detail::builtins::convert_arg(y),          \
      detail::builtins::convert_arg(z)

#define ONE_ARG_AUTO_ARG auto x
#define TWO_ARGS_AUTO_ARG auto x, auto y
#define THREE_ARGS_AUTO_ARG auto x, auto y, auto z

#define ONE_ARG_TYPE_ARG(TYPE) TYPE x
#define TWO_ARGS_TYPE_ARG(TYPE) TYPE x, TYPE y
#define THREE_ARGS_TYPE_ARG(TYPE) TYPE x, TYPE y, TYPE z

#define ONE_ARG_TYPE(TYPE) TYPE
#define TWO_ARGS_TYPE(TYPE) TYPE, TYPE
#define THREE_ARGS_TYPE(TYPE) TYPE, TYPE, TYPE

#define ONE_ARG_VEC_TYPE(TYPE, VL) vec<TYPE, VL>
#define TWO_ARGS_VEC_TYPE(TYPE, VL) vec<TYPE, VL>, vec<TYPE, VL>
#define THREE_ARGS_VEC_TYPE(TYPE, VL)                                          \
  vec<TYPE, VL>, vec<TYPE, VL>, vec<TYPE, VL>

#define ONE_ARG_VEC_TYPE_ARG(TYPE, VL) vec<TYPE, VL> x
#define TWO_ARGS_VEC_TYPE_ARG(TYPE, VL) vec<TYPE, VL> x, vec<TYPE, VL> y
#define THREE_ARGS_VEC_TYPE_ARG(TYPE, VL)                                      \
  vec<TYPE, VL> x, vec<TYPE, VL> y, vec<TYPE, VL> z

#define TWO_ARGS_LESS_ONE ONE_ARG
#define THREE_ARGS_LESS_ONE TWO_ARGS

#define SYCL_CONCAT_IMPL(A, B) A##B
#define SYCL_CONCAT(A, B) SYCL_CONCAT_IMPL(A, B)

#define LESS_ONE(NUM_ARGS) SYCL_CONCAT(NUM_ARGS, _LESS_ONE)

// 3 types.
#define FP_TYPES float, double, half
// 6 types.
#define SIGNED_TYPES char, signed char, short, int, long, long long
// 5 types
#define UNSIGNED_TYPES                                                         \
  unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long
// 11 types
#define INTEGER_TYPES SIGNED_TYPES, UNSIGNED_TYPES

#define DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(                                  \
    NUM_ARGS, NAME, ENABLER, DELEGATOR, NS, /*SCALAR_VEC_IMPL*/...)            \
  template <NUM_ARGS##_TYPENAME_TYPE>                                          \
  detail::ENABLER<NUM_ARGS##_TEMPLATE_TYPE> NAME(                              \
      NUM_ARGS##_TEMPLATE_TYPE_ARG) {                                          \
    if constexpr (detail::is_marray_v<T0>) {                                   \
      return detail::DELEGATOR(                                                \
          [](NUM_ARGS##_AUTO_ARG) { return NS::NAME(NUM_ARGS##_ARG); },        \
          NUM_ARGS##_ARG);                                                     \
    } else {                                                                   \
      return __VA_ARGS__(NUM_ARGS##_CONVERTED_ARG);                            \
    }                                                                          \
  }

#define DEVICE_IMPL_TEMPLATE(NUM_ARGS, NAME, ENABLER, /*SCALAR_VEC_IMPL*/...)  \
  DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(NUM_ARGS, NAME, ENABLER,                \
                                       builtin_marray_impl, sycl, __VA_ARGS__)

// Use extern function declaration in function scope to save compile time.
// Otherwise the FE has to parse multiple types/VLs/functions costing us around
// 0.3s in compile-time. It also allows us to skip providing all the explicit
// declarations through even more macro magic.
#define HOST_IMPL_TEMPLATE_CUSTOM_DELEGATOR(                                   \
    NUM_ARGS, NAME, ENABLER, FUNC_CLASS, RET_TYPE_TRAITS, DELEGATOR)           \
  template <typename... Ts> auto __##FUNC_CLASS##_##NAME##_lambda(Ts... xs) {  \
    /* Can't inline into the real lambda due to                                \
     * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=112867. Can't emulate a    \
     * lambda through a local struct because templates are not allowed in      \
     * local structs. Have to specify FUNC_CLASS to avoid                      \
     * ambiguity between, e.g. sycl::__cos_lambda/sycl::native::__cos_lambda   \
     * or between max in common functions and max in integer functions.        \
     */                                                                        \
    using ret_ty = typename detail::RET_TYPE_TRAITS<                           \
        typename detail::first_type<Ts...>::type>::type;                       \
    extern ret_ty __##NAME##_impl(Ts...);                                      \
    return __##NAME##_impl(xs...);                                             \
  }                                                                            \
  template <NUM_ARGS##_TYPENAME_TYPE>                                          \
  detail::ENABLER<NUM_ARGS##_TEMPLATE_TYPE> NAME(                              \
      NUM_ARGS##_TEMPLATE_TYPE_ARG) {                                          \
    return detail::DELEGATOR(                                                  \
        [](auto... xs) { return __##FUNC_CLASS##_##NAME##_lambda(xs...); },    \
        NUM_ARGS##_ARG);                                                       \
  }

#define HOST_IMPL_TEMPLATE(NUM_ARGS, NAME, ENABLER, FUNC_CLASS,                \
                           RET_TYPE_TRAITS)                                    \
  HOST_IMPL_TEMPLATE_CUSTOM_DELEGATOR(NUM_ARGS, NAME, ENABLER, FUNC_CLASS,     \
                                      RET_TYPE_TRAITS,                         \
                                      builtin_default_host_impl)

#define HOST_IMPL_SCALAR_RET_TYPE(NUM_ARGS, NAME, RET_TYPE, TYPE)              \
  inline RET_TYPE NAME(NUM_ARGS##_TYPE_ARG(TYPE)) {                            \
    extern RET_TYPE __##NAME##_impl(NUM_ARGS##_TYPE(TYPE));                    \
    return __##NAME##_impl(NUM_ARGS##_ARG);                                    \
  }

#define HOST_IMPL_SCALAR(NUM_ARGS, NAME, TYPE)                                 \
  HOST_IMPL_SCALAR_RET_TYPE(NUM_ARGS, NAME, TYPE, TYPE)
