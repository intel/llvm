//==-- host_helper_macros.hpp -- Utility macros to implement sycl builtins -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#define FOR_VEC_1_16(MACRO, ...)                                               \
  MACRO(__VA_ARGS__, 1)                                                        \
  MACRO(__VA_ARGS__, 2)                                                        \
  MACRO(__VA_ARGS__, 3)                                                        \
  MACRO(__VA_ARGS__, 4)                                                        \
  MACRO(__VA_ARGS__, 8)                                                        \
  MACRO(__VA_ARGS__, 16)

#define FOR_VEC_2_4(MACRO, ...)                                                \
  MACRO(__VA_ARGS__, 2)                                                        \
  MACRO(__VA_ARGS__, 3)                                                        \
  MACRO(__VA_ARGS__, 4)

#define FOR_VEC_3_4(MACRO, ...)                                                \
  MACRO(__VA_ARGS__, 3)                                                        \
  MACRO(__VA_ARGS__, 4)

// For both macros below NS is the namespace of the original sycl builtin
// function (e.g., sycl::cos vs native::cos). Headers implementation uses
// something like
//
//   float sin(float x) {
//     extern __sin_impl(float);
//     return __sin_impl(x);
//   }
//
// and that "extern" declaration isn't automatically matched against the symbol
// we export. As such, verify the return type consistency using static_assert.
#define EXPORT_SCALAR_NS(NUM_ARGS, NAME, NS, TYPE)                             \
  __SYCL_EXPORT auto __##NAME##_impl(NUM_ARGS##_TYPE_ARG(TYPE))                \
      -> decltype(NAME##_host_impl(NUM_ARGS##_ARG)) {                          \
    static_assert(std::is_same_v<decltype(NAME##_host_impl(NUM_ARGS##_ARG)),   \
                                 decltype(NS::NAME(NUM_ARGS##_ARG))>);         \
    return NAME##_host_impl(NUM_ARGS##_ARG);                                   \
  }
#define EXPORT_VEC_NS(NUM_ARGS, NAME, NS, TYPE, VL)                            \
  __SYCL_EXPORT auto __##NAME##_impl(NUM_ARGS##_VEC_TYPE_ARG(TYPE, VL))        \
      -> decltype(NAME##_host_impl(NUM_ARGS##_ARG)) {                          \
    static_assert(std::is_same_v<decltype(NAME##_host_impl(NUM_ARGS##_ARG)),   \
                                 decltype(NS::NAME(NUM_ARGS##_ARG))>);         \
    return NAME##_host_impl(NUM_ARGS##_ARG);                                   \
  }

#define EXPORT_SCALAR(NUM_ARGS, NAME, TYPE)                                    \
  EXPORT_SCALAR_NS(NUM_ARGS, NAME, sycl, TYPE)
#define EXPORT_VEC(NUM_ARGS, NAME, TYPE, VL)                                   \
  EXPORT_VEC_NS(NUM_ARGS, NAME, sycl, TYPE, VL)

#define EXPORT_VEC_1_16_IMPL(NUM_ARGS, NAME, NS, TYPE)                         \
  FOR_VEC_1_16(EXPORT_VEC_NS, NUM_ARGS, NAME, NS, TYPE)

#define EXPORT_SCALAR_AND_VEC_1_16_IMPL(NUM_ARGS, NAME, NS, TYPE)              \
  EXPORT_SCALAR_NS(NUM_ARGS, NAME, NS, TYPE)                                   \
  FOR_VEC_1_16(EXPORT_VEC_NS, NUM_ARGS, NAME, NS, TYPE)

#define EXPORT_SCALAR_AND_VEC_2_4_IMPL(NUM_ARGS, NAME, TYPE)                   \
  EXPORT_SCALAR(NUM_ARGS, NAME, TYPE)                                          \
  FOR_VEC_2_4(EXPORT_VEC, NUM_ARGS, NAME, TYPE)

#define EXPORT_VEC_3_4_IMPL(NUM_ARGS, NAME, TYPE)                              \
  FOR_VEC_3_4(EXPORT_VEC, NUM_ARGS, NAME, TYPE)

#define EXPORT_SCALAR_AND_VEC_1_16_NS(NUM_ARGS, NAME, NS, ...)                 \
  FOR_EACH3(EXPORT_SCALAR_AND_VEC_1_16_IMPL, NUM_ARGS, NAME, NS, __VA_ARGS__)
#define EXPORT_VEC_1_16_NS(NUM_ARGS, NAME, NS, ...)                            \
  FOR_EACH3(EXPORT_VEC_1_16_IMPL, NUM_ARGS, NAME, NS, __VA_ARGS__)
#define EXPORT_SCALAR_AND_VEC_1_16(NUM_ARGS, NAME, ...)                        \
  EXPORT_SCALAR_AND_VEC_1_16_NS(NUM_ARGS, NAME, sycl, __VA_ARGS__)
#define EXPORT_VEC_1_16(NUM_ARGS, NAME, ...)                                   \
  EXPORT_VEC_1_16_NS(NUM_ARGS, NAME, sycl, __VA_ARGS__)

#define EXPORT_SCALAR_AND_VEC_2_4(NUM_ARGS, NAME, ...)                         \
  FOR_EACH2(EXPORT_SCALAR_AND_VEC_2_4_IMPL, NUM_ARGS, NAME, __VA_ARGS__)
#define EXPORT_VEC_3_4(NUM_ARGS, NAME, ...)                                    \
  FOR_EACH2(EXPORT_VEC_3_4_IMPL, NUM_ARGS, NAME, __VA_ARGS__)

#define HOST_IMPL(NAME, ...)                                                   \
  template <typename... Ts> static auto NAME##_host_impl(Ts... xs) {           \
    using namespace detail;                                                    \
    if constexpr ((... || is_vec_v<Ts>)) {                                     \
      using ret_elem_type = decltype(NAME##_host_impl(xs[0]...));              \
      using T = typename first_type<Ts...>::type;                              \
      vec<ret_elem_type, T::size()> r{};                                       \
      loop<T::size()>(                                                         \
          [&](auto idx) { r[idx] = NAME##_host_impl(xs[idx]...); });           \
      return r;                                                                \
    } else {                                                                   \
      return __VA_ARGS__(xs...);                                               \
    }                                                                          \
  }
