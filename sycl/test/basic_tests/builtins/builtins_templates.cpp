// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only %s -o %t
// REQUIRES: preview-breaking-changes-supported

// Checks that builtin functions defined as templates allow explicit template
// instantiation.

#include <sycl/sycl.hpp>

#define ONE_ARG_DECLVAL(...) std::declval<__VA_ARGS__>()

#define TWO_ARGS_DECLVAL(...)                                                  \
  ONE_ARG_DECLVAL(__VA_ARGS__), ONE_ARG_DECLVAL(__VA_ARGS__)

#define THREE_ARGS_DECLVAL(...)                                                \
  TWO_ARGS_DECLVAL(__VA_ARGS__), ONE_ARG_DECLVAL(__VA_ARGS__)

#define CHECK_INNER(NUM_ARGS, FUNC_NAME, ...)                                  \
  static_assert(std::is_same_v<decltype(sycl::FUNC_NAME<__VA_ARGS__>(          \
                                   NUM_ARGS##_DECLVAL(__VA_ARGS__))),          \
                               decltype(sycl::FUNC_NAME(                       \
                                   NUM_ARGS##_DECLVAL(__VA_ARGS__)))>);

#define NONSCALAR_FLOAT_CHECK(NUM_ARGS, FUNC_NAME)                             \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<float, 4>)                        \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<float, 4>)

#define FLOAT_CHECK(NUM_ARGS, FUNC_NAME)                                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, float)                                      \
  NONSCALAR_FLOAT_CHECK(NUM_ARGS, FUNC_NAME)

#define NONSCALAR_GENFLOAT_CHECK(NUM_ARGS, FUNC_NAME)                          \
  NONSCALAR_FLOAT_CHECK(NUM_ARGS, FUNC_NAME)                                   \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<sycl::half, 4>)                   \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<sycl::half, 4>)                \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<double, 4>)                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<double, 4>)

#define GENFLOAT_CHECK(NUM_ARGS, FUNC_NAME)                                    \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, float)                                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, double)                                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::half)                                 \
  NONSCALAR_GENFLOAT_CHECK(NUM_ARGS, FUNC_NAME)

#define NONSCALAR_UGENINT_NAN_CHECK(NUM_ARGS, FUNC_NAME)                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint32_t, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint16_t, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint64_t, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<uint32_t, 4>)                  \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<uint16_t, 4>)                  \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<uint64_t, 4>)

#define SGENINT_CHECK(NUM_ARGS, FUNC_NAME)                                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, signed char)                                \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, short)                                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, int)                                        \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, long long)                                  \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, long)                                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int8_t, 4>)                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int16_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int32_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int64_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<signed char, 4>)               \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<short, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<int, 4>)                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<long long, 4>)                 \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<long, 4>)

#define UGENINT_CHECK(NUM_ARGS, FUNC_NAME)                                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, unsigned char)                              \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, unsigned short)                             \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, unsigned int)                               \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, unsigned long long)                         \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, unsigned long)                              \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint8_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<unsigned int, 4>)              \
  NONSCALAR_UGENINT_NAN_CHECK(NUM_ARGS, FUNC_NAME)

#define GENINT_CHECK(NUM_ARGS, FUNC_NAME)                                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, char)                                       \
  SGENINT_CHECK(NUM_ARGS, FUNC_NAME)                                           \
  UGENINT_CHECK(NUM_ARGS, FUNC_NAME)

#define GENTYPE_CHECK(NUM_ARGS, FUNC_NAME)                                     \
  GENFLOAT_CHECK(NUM_ARGS, FUNC_NAME)                                          \
  GENINT_CHECK(NUM_ARGS, FUNC_NAME)

#define NONSCALAR_SGENINT_CHECK(NUM_ARGS, FUNC_NAME)                           \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int8_t, 4>)                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int16_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int32_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<int64_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<signed char, 4>)               \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<short, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<int, 4>)                       \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<long long, 4>)                 \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<long, 4>)

#define NONSCALAR_UGENINT_CHECK(NUM_ARGS, FUNC_NAME)                           \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint8_t, 4>)                      \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint16_t, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint32_t, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::vec<uint64_t, 4>)                     \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<unsigned char, 4>)             \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<unsigned short, 4>)            \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<unsigned int, 4>)              \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<unsigned long long, 4>)        \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::marray<unsigned long, 4>)

void check() {
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, acos)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, acosh)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, acospi)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, asin)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, asinh)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, asinpi)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, atan)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, atanh)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, atanpi)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, atan2)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, atan2pi)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, cbrt)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, ceil)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, copysign)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, cos)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, cosh)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, cospi)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, erfc)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, erf)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, exp)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, exp2)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, exp10)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, expm1)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, fabs)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, fdim)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, floor)
  NONSCALAR_GENFLOAT_CHECK(THREE_ARGS, fma)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, fmax)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, fmin)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, fmod)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, hypot)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, ilogb)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, lgamma)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, log)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, log2)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, log10)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, log1p)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, logb)
  NONSCALAR_GENFLOAT_CHECK(THREE_ARGS, mad)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, maxmag)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, minmag)
  NONSCALAR_UGENINT_NAN_CHECK(ONE_ARG, nan)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, nextafter)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, pow)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, powr)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, remainder)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, rint)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, round)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, rsqrt)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, sin)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, sinh)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, sinpi)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, sqrt)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, tan)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, tanh)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, tanpi)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, tgamma)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, trunc)

  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::cos)
  NONSCALAR_FLOAT_CHECK(TWO_ARGS, native::divide)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::exp)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::exp2)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::exp10)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::log)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::log2)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::log10)
  NONSCALAR_FLOAT_CHECK(TWO_ARGS, native::powr)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::recip)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::rsqrt)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::sin)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::sqrt)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, native::tan)

  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::cos)
  NONSCALAR_FLOAT_CHECK(TWO_ARGS, half_precision::divide)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::exp)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::exp2)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::exp10)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::log)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::log2)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::log10)
  NONSCALAR_FLOAT_CHECK(TWO_ARGS, half_precision::powr)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::recip)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::rsqrt)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::sin)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::sqrt)
  NONSCALAR_FLOAT_CHECK(ONE_ARG, half_precision::tan)

  GENINT_CHECK(ONE_ARG, abs)
  GENINT_CHECK(TWO_ARGS, abs_diff)
  GENINT_CHECK(TWO_ARGS, add_sat)
  GENINT_CHECK(TWO_ARGS, hadd)
  GENINT_CHECK(TWO_ARGS, rhadd)
  GENINT_CHECK(THREE_ARGS, clamp)
  GENINT_CHECK(ONE_ARG, clz)
  GENINT_CHECK(ONE_ARG, ctz)
  GENINT_CHECK(THREE_ARGS, mad_hi)
  GENINT_CHECK(THREE_ARGS, mad_sat)
  GENINT_CHECK(TWO_ARGS, max)
  GENINT_CHECK(TWO_ARGS, min)
  GENINT_CHECK(TWO_ARGS, mul_hi)
  GENINT_CHECK(TWO_ARGS, rotate)
  GENINT_CHECK(TWO_ARGS, sub_sat)
  GENINT_CHECK(ONE_ARG, popcount)

  GENFLOAT_CHECK(THREE_ARGS, clamp)
  GENFLOAT_CHECK(ONE_ARG, degrees)
  GENFLOAT_CHECK(TWO_ARGS, max)
  GENFLOAT_CHECK(TWO_ARGS, min)
  GENFLOAT_CHECK(THREE_ARGS, mix)
  GENFLOAT_CHECK(ONE_ARG, radians)
  GENFLOAT_CHECK(TWO_ARGS, step)
  GENFLOAT_CHECK(THREE_ARGS, smoothstep)
  GENFLOAT_CHECK(ONE_ARG, sign)

  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, cross)
  GENFLOAT_CHECK(TWO_ARGS, dot)
  GENFLOAT_CHECK(TWO_ARGS, distance)
  GENFLOAT_CHECK(ONE_ARG, length)
  GENFLOAT_CHECK(ONE_ARG, normalize)
  FLOAT_CHECK(TWO_ARGS, fast_distance)
  FLOAT_CHECK(ONE_ARG, fast_length)
  FLOAT_CHECK(ONE_ARG, fast_normalize)

  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, isequal)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, isnotequal)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, isgreater)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, isgreaterequal)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, isless)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, islessequal)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, islessgreater)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, isfinite)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, isinf)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, isnan)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, isnormal)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, isordered)
  NONSCALAR_GENFLOAT_CHECK(TWO_ARGS, isunordered)
  NONSCALAR_GENFLOAT_CHECK(ONE_ARG, signbit)
  SGENINT_CHECK(ONE_ARG, any)
  SGENINT_CHECK(ONE_ARG, all)
  GENTYPE_CHECK(THREE_ARGS, bitselect)
}

int main() {
  check();

  sycl::queue Q;
  Q.single_task([=]() { check(); });
}
