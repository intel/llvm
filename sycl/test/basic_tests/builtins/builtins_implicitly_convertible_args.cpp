// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only %s -o %t
// REQUIRES: preview-breaking-changes-supported

// Checks that builtins without template arguments allow for implicit
// conversions of arguments.

#include <sycl/sycl.hpp>

template <typename T> struct ImplicitlyConvertibleType {
  operator T() const { return {}; }
};

#define ONE_ARG_DECLVAL_IMPLICITLY_CONVERTIBLE(...)                            \
  std::declval<ImplicitlyConvertibleType<__VA_ARGS__>>()

#define TWO_ARGS_DECLVAL_IMPLICITLY_CONVERTIBLE(...)                           \
  ONE_ARG_DECLVAL_IMPLICITLY_CONVERTIBLE(__VA_ARGS__),                         \
      ONE_ARG_DECLVAL_IMPLICITLY_CONVERTIBLE(__VA_ARGS__)

#define THREE_ARGS_DECLVAL_IMPLICITLY_CONVERTIBLE(...)                         \
  TWO_ARGS_DECLVAL_IMPLICITLY_CONVERTIBLE(__VA_ARGS__),                        \
      ONE_ARG_DECLVAL_IMPLICITLY_CONVERTIBLE(__VA_ARGS__)

#define ONE_ARG_DECLVAL(...) std::declval<__VA_ARGS__>()

#define TWO_ARGS_DECLVAL(...)                                                  \
  ONE_ARG_DECLVAL(__VA_ARGS__), ONE_ARG_DECLVAL(__VA_ARGS__)

#define THREE_ARGS_DECLVAL(...)                                                \
  TWO_ARGS_DECLVAL(__VA_ARGS__), ONE_ARG_DECLVAL(__VA_ARGS__)

#define CHECK_INNER(NUM_ARGS, FUNC_NAME, ...)                                  \
  static_assert(std::is_same_v<                                                \
                decltype(sycl::FUNC_NAME(                                      \
                    NUM_ARGS##_DECLVAL_IMPLICITLY_CONVERTIBLE(__VA_ARGS__))),  \
                decltype(sycl::FUNC_NAME(NUM_ARGS##_DECLVAL(__VA_ARGS__)))>);

#define FLOAT_CHECK(NUM_ARGS, FUNC_NAME) CHECK_INNER(NUM_ARGS, FUNC_NAME, float)

#define GENFLOAT_CHECK(NUM_ARGS, FUNC_NAME)                                    \
  FLOAT_CHECK(NUM_ARGS, FUNC_NAME)                                             \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, sycl::half)                                 \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, double)

#define UGENINT_NAN_CHECK(NUM_ARGS, FUNC_NAME)                                 \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, uint32_t)                                   \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, uint16_t)                                   \
  CHECK_INNER(NUM_ARGS, FUNC_NAME, uint64_t)

void check() {
  GENFLOAT_CHECK(ONE_ARG, acos)
  GENFLOAT_CHECK(ONE_ARG, acosh)
  GENFLOAT_CHECK(ONE_ARG, acospi)
  GENFLOAT_CHECK(ONE_ARG, asin)
  GENFLOAT_CHECK(ONE_ARG, asinh)
  GENFLOAT_CHECK(ONE_ARG, asinpi)
  GENFLOAT_CHECK(ONE_ARG, atan)
  GENFLOAT_CHECK(ONE_ARG, atanh)
  GENFLOAT_CHECK(ONE_ARG, atanpi)
  GENFLOAT_CHECK(TWO_ARGS, atan2)
  GENFLOAT_CHECK(TWO_ARGS, atan2pi)
  GENFLOAT_CHECK(ONE_ARG, cbrt)
  GENFLOAT_CHECK(ONE_ARG, ceil)
  GENFLOAT_CHECK(TWO_ARGS, copysign)
  GENFLOAT_CHECK(ONE_ARG, cos)
  GENFLOAT_CHECK(ONE_ARG, cosh)
  GENFLOAT_CHECK(ONE_ARG, cospi)
  GENFLOAT_CHECK(ONE_ARG, erfc)
  GENFLOAT_CHECK(ONE_ARG, erf)
  GENFLOAT_CHECK(ONE_ARG, exp)
  GENFLOAT_CHECK(ONE_ARG, exp2)
  GENFLOAT_CHECK(ONE_ARG, exp10)
  GENFLOAT_CHECK(ONE_ARG, expm1)
  GENFLOAT_CHECK(ONE_ARG, fabs)
  GENFLOAT_CHECK(TWO_ARGS, fdim)
  GENFLOAT_CHECK(ONE_ARG, floor)
  GENFLOAT_CHECK(THREE_ARGS, fma)
  GENFLOAT_CHECK(TWO_ARGS, fmax)
  GENFLOAT_CHECK(TWO_ARGS, fmin)
  GENFLOAT_CHECK(TWO_ARGS, fmod)
  GENFLOAT_CHECK(TWO_ARGS, hypot)
  GENFLOAT_CHECK(ONE_ARG, ilogb)
  GENFLOAT_CHECK(ONE_ARG, lgamma)
  GENFLOAT_CHECK(ONE_ARG, log)
  GENFLOAT_CHECK(ONE_ARG, log2)
  GENFLOAT_CHECK(ONE_ARG, log10)
  GENFLOAT_CHECK(ONE_ARG, log1p)
  GENFLOAT_CHECK(ONE_ARG, logb)
  GENFLOAT_CHECK(THREE_ARGS, mad)
  GENFLOAT_CHECK(TWO_ARGS, maxmag)
  GENFLOAT_CHECK(TWO_ARGS, minmag)
  UGENINT_NAN_CHECK(ONE_ARG, nan)
  GENFLOAT_CHECK(TWO_ARGS, nextafter)
  GENFLOAT_CHECK(TWO_ARGS, pow)
  GENFLOAT_CHECK(TWO_ARGS, powr)
  GENFLOAT_CHECK(TWO_ARGS, remainder)
  GENFLOAT_CHECK(ONE_ARG, rint)
  GENFLOAT_CHECK(ONE_ARG, round)
  GENFLOAT_CHECK(ONE_ARG, rsqrt)
  GENFLOAT_CHECK(ONE_ARG, sin)
  GENFLOAT_CHECK(ONE_ARG, sinh)
  GENFLOAT_CHECK(ONE_ARG, sinpi)
  GENFLOAT_CHECK(ONE_ARG, sqrt)
  GENFLOAT_CHECK(ONE_ARG, tan)
  GENFLOAT_CHECK(ONE_ARG, tanh)
  GENFLOAT_CHECK(ONE_ARG, tanpi)
  GENFLOAT_CHECK(ONE_ARG, tgamma)
  GENFLOAT_CHECK(ONE_ARG, trunc)

  FLOAT_CHECK(ONE_ARG, native::cos)
  FLOAT_CHECK(TWO_ARGS, native::divide)
  FLOAT_CHECK(ONE_ARG, native::exp)
  FLOAT_CHECK(ONE_ARG, native::exp2)
  FLOAT_CHECK(ONE_ARG, native::exp10)
  FLOAT_CHECK(ONE_ARG, native::log)
  FLOAT_CHECK(ONE_ARG, native::log2)
  FLOAT_CHECK(ONE_ARG, native::log10)
  FLOAT_CHECK(TWO_ARGS, native::powr)
  FLOAT_CHECK(ONE_ARG, native::recip)
  FLOAT_CHECK(ONE_ARG, native::rsqrt)
  FLOAT_CHECK(ONE_ARG, native::sin)
  FLOAT_CHECK(ONE_ARG, native::sqrt)
  FLOAT_CHECK(ONE_ARG, native::tan)

  FLOAT_CHECK(ONE_ARG, half_precision::cos)
  FLOAT_CHECK(TWO_ARGS, half_precision::divide)
  FLOAT_CHECK(ONE_ARG, half_precision::exp)
  FLOAT_CHECK(ONE_ARG, half_precision::exp2)
  FLOAT_CHECK(ONE_ARG, half_precision::exp10)
  FLOAT_CHECK(ONE_ARG, half_precision::log)
  FLOAT_CHECK(ONE_ARG, half_precision::log2)
  FLOAT_CHECK(ONE_ARG, half_precision::log10)
  FLOAT_CHECK(TWO_ARGS, half_precision::powr)
  FLOAT_CHECK(ONE_ARG, half_precision::recip)
  FLOAT_CHECK(ONE_ARG, half_precision::rsqrt)
  FLOAT_CHECK(ONE_ARG, half_precision::sin)
  FLOAT_CHECK(ONE_ARG, half_precision::sqrt)
  FLOAT_CHECK(ONE_ARG, half_precision::tan)

  GENFLOAT_CHECK(TWO_ARGS, isequal)
  GENFLOAT_CHECK(TWO_ARGS, isnotequal)
  GENFLOAT_CHECK(TWO_ARGS, isgreater)
  GENFLOAT_CHECK(TWO_ARGS, isgreaterequal)
  GENFLOAT_CHECK(TWO_ARGS, isless)
  GENFLOAT_CHECK(TWO_ARGS, islessequal)
  GENFLOAT_CHECK(TWO_ARGS, islessgreater)
  GENFLOAT_CHECK(ONE_ARG, isfinite)
  GENFLOAT_CHECK(ONE_ARG, isinf)
  GENFLOAT_CHECK(ONE_ARG, isnan)
  GENFLOAT_CHECK(ONE_ARG, isnormal)
  GENFLOAT_CHECK(TWO_ARGS, isordered)
  GENFLOAT_CHECK(TWO_ARGS, isunordered)
  GENFLOAT_CHECK(ONE_ARG, signbit)
}

int main() {
  check();

  sycl::queue Q;
  Q.single_task([=]() { check(); });
}
