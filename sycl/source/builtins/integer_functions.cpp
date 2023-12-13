//==------------------- integer_functions.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/builtins_preview.hpp>

#include "host_helper_macros.hpp"

namespace {
// A helper function for mul_hi built-in for long
template <typename T> inline T __get_high_half(T a0b0, T a0b1, T a1b0, T a1b1) {
  constexpr int halfsize = (sizeof(T) * 8) / 2;
  // To get the upper 64 bits:
  // 64 bits from a1b1, upper 32 bits from [a1b0 + (a0b1 + a0b0>>32 (carry bit
  // in 33rd bit))] with carry bit on 64th bit - use of hadd. Add the a1b1 to
  // the above 32 bit result.
  return a1b1 +
         (sycl::hadd(a1b0, (a0b1 + (a0b0 >> halfsize))) >> (halfsize - 1));
}

// A helper function for mul_hi built-in for long
template <typename T>
inline void __get_half_products(T a, T b, T &a0b0, T &a0b1, T &a1b0, T &a1b1) {
  constexpr sycl::cl_int halfsize = (sizeof(T) * 8) / 2;
  T a1 = a >> halfsize;
  T a0 = (a << halfsize) >> halfsize;
  T b1 = b >> halfsize;
  T b0 = (b << halfsize) >> halfsize;

  // a1b1 - for bits - [64-128)
  // a1b0 a0b1 for bits - [32-96)
  // a0b0 for bits - [0-64)
  a1b1 = a1 * b1;
  a0b1 = a0 * b1;
  a1b0 = a1 * b0;
  a0b0 = a0 * b0;
}

// T is minimum of 64 bits- long or longlong
template <typename T> inline T __u_long_mul_hi(T a, T b) {
  T a0b0, a0b1, a1b0, a1b1;
  __get_half_products(a, b, a0b0, a0b1, a1b0, a1b1);
  T result = __get_high_half(a0b0, a0b1, a1b0, a1b1);
  return result;
}

template <typename T> inline T __s_long_mul_hi(T a, T b) {
  using UT = std::make_unsigned_t<T>;
  UT absA = std::abs(a);
  UT absB = std::abs(b);

  UT a0b0, a0b1, a1b0, a1b1;
  __get_half_products(absA, absB, a0b0, a0b1, a1b0, a1b1);
  T result = __get_high_half(a0b0, a0b1, a1b0, a1b1);

  bool isResultNegative = (a < 0) != (b < 0);
  if (isResultNegative) {
    result = ~result;

    // Find the low half to see if we need to carry
    constexpr int halfsize = (sizeof(T) * 8) / 2;
    UT low = a0b0 + ((a0b1 + a1b0) << halfsize);
    if (low == 0)
      ++result;
  }

  return result;
}
} // namespace

namespace sycl {
inline namespace _V1 {
#define BUILTIN_GENINT(NUM_ARGS, NAME, IMPL)                                   \
  HOST_IMPL(NAME, IMPL)                                                        \
  EXPORT_SCALAR_AND_VEC_1_16(NUM_ARGS, NAME, INTEGER_TYPES)
#define BUILTIN_GENINT_SU(NUM_ARGS, NAME, IMPL)                                \
  BUILTIN_GENINT(NUM_ARGS, NAME, IMPL)

BUILTIN_GENINT(ONE_ARG, abs, [](auto x) -> decltype(x) {
  if constexpr (std::is_signed_v<decltype(x)>) {
    return std::abs(x);
  } else {
    return x;
  }
})

BUILTIN_GENINT_SU(TWO_ARGS, abs_diff, [](auto x, auto y) -> decltype(x) {
  // From SYCL 2020 revision 8:
  //
  // > The subtraction is done without modulo overflow. The behavior is
  // > undefined if the result cannot be represented by the return type.
  return sycl::abs(x - y);
})

BUILTIN_GENINT_SU(TWO_ARGS, add_sat, [](auto x, auto y) -> decltype(x) {
  using T = decltype(x);
  if constexpr (std::is_signed_v<T>) {
    if (x > 0 && y > 0)
      return (x < (std::numeric_limits<T>::max() - y)
                  ? (x + y)
                  : std::numeric_limits<T>::max());
    if (x < 0 && y < 0)
      return (x > (std::numeric_limits<T>::min() - y)
                  ? (x + y)
                  : std::numeric_limits<T>::min());
    return x + y;
  } else {
    return (x < (std::numeric_limits<T>::max() - y)
                ? x + y
                : std::numeric_limits<T>::max());
  }
})

BUILTIN_GENINT_SU(TWO_ARGS, hadd, [](auto x, auto y) -> decltype(x) {
  const decltype(x) one = 1;
  return (x >> one) + (y >> one) + ((y & x) & one);
})

BUILTIN_GENINT_SU(TWO_ARGS, rhadd, [](auto x, auto y) -> decltype(x) {
  const decltype(x) one = 1;
  return (x >> one) + (y >> one) + ((y | x) & one);
})

BUILTIN_GENINT_SU(THREE_ARGS, mad_hi,
                  [](auto x, auto y, auto z) -> decltype(x) {
                    return sycl::mul_hi(x, y) + z;
                  })

BUILTIN_GENINT_SU(
    THREE_ARGS, mad_sat, [](auto a, auto b, auto c) -> decltype(a) {
      using T = decltype(a);
      if constexpr (std::is_signed_v<T>) {
        if constexpr (sizeof(T) == 8) {
          bool neg_prod = (a < 0) ^ (b < 0);
          T mulhi = __s_long_mul_hi(a, b);

          // check mul_hi. If it is any value != 0.
          // if prod is +ve, any value in mulhi means we need to saturate.
          // if prod is -ve, any value in mulhi besides -1 means we need to
          // saturate.
          if (!neg_prod && mulhi != 0)
            return std::numeric_limits<T>::max();
          if (neg_prod && mulhi != -1)
            return std::numeric_limits<T>::min(); // essentially some other
                                                  // negative value.
          return sycl::add_sat(T(a * b), c);
        } else {
          using UPT = sycl::detail::make_larger_t<T>;
          UPT mul = UPT(a) * UPT(b);
          UPT res = mul + UPT(c);
          const UPT max = std::numeric_limits<T>::max();
          const UPT min = std::numeric_limits<T>::min();
          res = std::min(std::max(res, min), max);
          return T(res);
        }
      } else {
        if constexpr (sizeof(T) == 8) {
          T mulhi = __u_long_mul_hi(a, b);
          // check mul_hi. If it is any value != 0.
          if (mulhi != 0)
            return std::numeric_limits<T>::max();
          return sycl::add_sat(T(a * b), c);
        } else {
          using UPT = sycl::detail::make_larger_t<T>;
          UPT mul = UPT(a) * UPT(b);
          const UPT min = std::numeric_limits<T>::min();
          const UPT max = std::numeric_limits<T>::max();
          mul = std::min(std::max(mul, min), max);
          return sycl::add_sat(T(mul), c);
        }
      }
    })

BUILTIN_GENINT_SU(TWO_ARGS, mul_hi, [](auto a, auto b) -> decltype(a) {
  using T = decltype(a);
  if constexpr (sizeof(T) == 8) {
    if constexpr (std::is_signed_v<T>)
      return __s_long_mul_hi(a, b);
    else
      return __u_long_mul_hi(a, b);
  } else {
    using UPT = sycl::detail::make_larger_t<T>;
    UPT a_s = a;
    UPT b_s = b;
    UPT mul = a_s * b_s;
    return (mul >> (sizeof(T) * 8));
  }
})

BUILTIN_GENINT_SU(TWO_ARGS, sub_sat, [](auto x, auto y) -> decltype(x) {
  using T = decltype(x);
  if constexpr (std::is_signed_v<T>) {
    using UT = std::make_unsigned_t<T>;
    T result = UT(x) - UT(y);
    // Saturate result if (+) - (-) = (-) or (-) - (+) = (+).
    if (((x < 0) ^ (y < 0)) && ((x < 0) ^ (result < 0)))
      result = result < 0 ? std::numeric_limits<T>::max()
                          : std::numeric_limits<T>::min();
    return result;
  } else {
    return (y < (x - std::numeric_limits<T>::min()))
               ? (x - y)
               : std::numeric_limits<T>::min();
  }
})

BUILTIN_GENINT_SU(TWO_ARGS, max,
                  [](auto x, auto y) -> decltype(x) { return x < y ? y : x; })

BUILTIN_GENINT_SU(TWO_ARGS, min,
                  [](auto x, auto y) -> decltype(x) { return y < x ? y : x; })

template <typename T> static inline constexpr T __clz_impl(T x, T m, T n = 0) {
  return (x & m) ? n : __clz_impl(x, T(m >> 1), ++n);
}
template <typename T> static inline constexpr T __clz(T x) {
  using UT = std::make_unsigned_t<T>;
  return (x == T(0)) ? sizeof(T) * 8
                     : __clz_impl<UT>(x, sycl::detail::msbMask<UT>(x));
}
BUILTIN_GENINT(ONE_ARG, clz, __clz)

template <typename T> static inline constexpr T __ctz_impl(T x, T m, T n = 0) {
  return (x & m) ? n : __ctz_impl(x, T(m << 1), ++n);
}

template <typename T> static inline constexpr T __ctz(T x) {
  using UT = std::make_unsigned_t<T>;
  return (x == T(0)) ? sizeof(T) * 8 : __ctz_impl<UT>(x, 1);
}
BUILTIN_GENINT(ONE_ARG, ctz, __ctz)

BUILTIN_GENINT(TWO_ARGS, rotate, [](auto x, auto n) -> decltype(x) {
  using T = decltype(x);
  using UT = std::make_unsigned_t<T>;
  // Shrink the shift width so that it's in the range [0, num_bits(T)). Cast
  // everything to unsigned to avoid type conversion issues.
  constexpr UT size = sizeof(x) * 8;
  UT xu = UT(x);
  UT nu = UT(n) & (size - 1);
  return (xu << nu) | (xu >> (size - nu));
})

template <typename T>
static inline constexpr T __popcount_impl(T x, size_t n = 0) {
  return (x == T(0)) ? n : __popcount_impl(x >> 1, ((x & T(1)) ? ++n : n));
}
template <typename T> static inline constexpr T __popcount(T x) {
  using UT = sycl::detail::make_unsigned_t<T>;
  return __popcount_impl(UT(x));
}
BUILTIN_GENINT(ONE_ARG, popcount, __popcount)
} // namespace _V1
} // namespace sycl
