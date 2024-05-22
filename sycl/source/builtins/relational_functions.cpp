//==------------------- relational_functions.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/builtins/builtins.hpp>

#include "host_helper_macros.hpp"

#include <bitset>
#include <cmath>

namespace sycl {
inline namespace _V1 {

template <typename T> static auto process_arg_for_macos(T x) {
  // Workaround for MacOS that doesn't provide some std::is* functions as
  // overloads over FP types (e.g., isfinite)
  if constexpr (std::is_same_v<T, half>)
    return static_cast<float>(x);
  else
    return x;
}

#if defined(__GNUC__) && !defined(__clang__)
// sycl::vec has UB in operator[] (aliasing violation) that causes the following
// warning here. Note that the way this #pragma works is that we have to put it
// around the macro definition, not where the macro is instantiated.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#define REL_BUILTIN_CUSTOM(NUM_ARGS, NAME, ...)                                \
  template <typename... Ts> static auto NAME##_host_impl(Ts... xs) {           \
    using namespace detail;                                                    \
    if constexpr ((... || is_vec_v<Ts>)) {                                     \
      return builtin_delegate_rel_impl(                                        \
          [](auto... xs) { return NAME##_host_impl(xs...); }, xs...);          \
    } else {                                                                   \
      return __VA_ARGS__(xs...);                                               \
    }                                                                          \
  }                                                                            \
  EXPORT_SCALAR_AND_VEC_1_16(NUM_ARGS, NAME, FP_TYPES)
#define REL_BUILTIN(NUM_ARGS, NAME)                                            \
  REL_BUILTIN_CUSTOM(NUM_ARGS, NAME, [](auto... xs) {                          \
    return std::NAME(process_arg_for_macos(xs)...);                            \
  })

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

REL_BUILTIN_CUSTOM(TWO_ARGS, isequal, ([](auto x, auto y) { return x == y; }))
REL_BUILTIN_CUSTOM(TWO_ARGS, isnotequal,
                   ([](auto x, auto y) { return x != y; }))
REL_BUILTIN_CUSTOM(TWO_ARGS, isgreater, ([](auto x, auto y) { return x > y; }))
REL_BUILTIN_CUSTOM(TWO_ARGS, isgreaterequal,
                   ([](auto x, auto y) { return x >= y; }))
REL_BUILTIN_CUSTOM(TWO_ARGS, isless, ([](auto x, auto y) { return x < y; }))
REL_BUILTIN_CUSTOM(TWO_ARGS, islessequal,
                   ([](auto x, auto y) { return x <= y; }))
REL_BUILTIN_CUSTOM(TWO_ARGS, islessgreater,
                   ([](auto x, auto y) { return x < y || x > y; }))
REL_BUILTIN(ONE_ARG, isfinite)
REL_BUILTIN(ONE_ARG, isinf)
REL_BUILTIN(ONE_ARG, isnan)
REL_BUILTIN(ONE_ARG, isnormal)
REL_BUILTIN(TWO_ARGS, isunordered)
REL_BUILTIN_CUSTOM(TWO_ARGS, isordered,
                   ([](auto x, auto y) { return !sycl::isunordered(x, y); }))

#define _SYCL_BUILTINS_GCC_VER                                                 \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if defined(__GNUC__) && !defined(__clang__) &&                                \
    ((_SYCL_BUILTINS_GCC_VER >= 100000 && _SYCL_BUILTINS_GCC_VER < 110000) ||  \
     (_SYCL_BUILTINS_GCC_VER >= 110000 && _SYCL_BUILTINS_GCC_VER < 110500) ||  \
     (_SYCL_BUILTINS_GCC_VER >= 120000 && _SYCL_BUILTINS_GCC_VER < 120400) ||  \
     (_SYCL_BUILTINS_GCC_VER >= 130000 && _SYCL_BUILTINS_GCC_VER < 130300))
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=112816
// The reproducers in that ticket only affect GCCs 10 to 13. Release
// branches 11.5, 12.4, and 13.3 have been updated with the fix; release branch
// 10 hasn't, so all GCCs 10.x are considered affected.
#define GCC_PR112816_DISABLE_OPT                                               \
  _Pragma("GCC push_options") _Pragma("GCC optimize(\"-O1\")")
#define GCC_PR112816_RESTORE_OPT _Pragma("GCC pop_options")
#else
#define GCC_PR112816_DISABLE_OPT
#define GCC_PR112816_RESTORE_OPT
#endif

GCC_PR112816_DISABLE_OPT

REL_BUILTIN(ONE_ARG, signbit)

GCC_PR112816_RESTORE_OPT

#undef GCC_PR112816_RESTORE_OPT
#undef GCC_PR112816_DISABLE_OPT
#undef _SYCL_BUILTINS_GCC_VER

HOST_IMPL(bitselect, [](auto x, auto y, auto z) {
  using T0 = decltype(x);
  using T1 = decltype(y);
  using T2 = decltype(z);
  constexpr size_t N = sizeof(T0) * 8;
  using bitset = std::bitset<N>;

  static_assert(std::is_same_v<T0, T1> && std::is_same_v<T1, T2> &&
                detail::is_scalar_arithmetic_v<T0>);

  using utype = detail::make_type_t<
      T0, detail::type_list<unsigned char, unsigned short, unsigned int,
                            unsigned long, unsigned long long>>;
  static_assert(sizeof(utype) == sizeof(T0));
  bitset bx(bit_cast<utype>(x)), by(bit_cast<utype>(y)), bz(bit_cast<utype>(z));
  bitset res = (bz & by) | (~bz & bx);
  unsigned long long ures = res.to_ullong();
  assert((ures & std::numeric_limits<utype>::max()) == ures);
  return bit_cast<T0>(static_cast<utype>(ures));
})
FOR_EACH2(EXPORT_SCALAR, THREE_ARGS, bitselect, INTEGER_TYPES, FP_TYPES)
EXPORT_VEC_1_16(THREE_ARGS, bitselect, FIXED_WIDTH_INTEGER_TYPES, FP_TYPES)
} // namespace _V1
} // namespace sycl
