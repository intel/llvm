// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only %s -o %t
// REQUIRES: preview-breaking-changes-supported

// Checks that builtins without template arguments allow for implicit
// conversions of arguments.

#include <sycl/sycl.hpp>

template <typename T> struct ImplicitlyConvertibleType {
  operator T() const { return {}; }
};

#define UNARY_CHECK_INNER(FUNC_NAME, ...)                                      \
  static_assert(std::is_same_v<                                                \
                decltype(sycl::FUNC_NAME(                                      \
                    std::declval<ImplicitlyConvertibleType<__VA_ARGS__>>())),  \
                decltype(sycl::FUNC_NAME(std::declval<__VA_ARGS__>()))>);

#define FLOAT_UNARY_CHECK(FUNC_NAME) UNARY_CHECK_INNER(FUNC_NAME, float)

#define GENFLOAT_UNARY_CHECK(FUNC_NAME)                                        \
  FLOAT_UNARY_CHECK(FUNC_NAME)                                                 \
  UNARY_CHECK_INNER(FUNC_NAME, sycl::half)                                     \
  UNARY_CHECK_INNER(FUNC_NAME, double)

#define UGENINT_NAN_UNARY_CHECK(FUNC_NAME)                                     \
  UNARY_CHECK_INNER(FUNC_NAME, unsigned int)                                   \
  UNARY_CHECK_INNER(FUNC_NAME, unsigned short)                                 \
  UNARY_CHECK_INNER(FUNC_NAME, unsigned long)

#define BINARY_CHECK_INNER(FUNC_NAME, ...)                                     \
  static_assert(std::is_same_v<                                                \
                decltype(sycl::FUNC_NAME(                                      \
                    std::declval<ImplicitlyConvertibleType<__VA_ARGS__>>(),    \
                    std::declval<ImplicitlyConvertibleType<__VA_ARGS__>>())),  \
                decltype(sycl::FUNC_NAME(std::declval<__VA_ARGS__>(),          \
                                         std::declval<__VA_ARGS__>()))>);

#define FLOAT_BINARY_CHECK(FUNC_NAME) BINARY_CHECK_INNER(FUNC_NAME, float)

#define GENFLOAT_BINARY_CHECK(FUNC_NAME)                                       \
  FLOAT_BINARY_CHECK(FUNC_NAME)                                                \
  BINARY_CHECK_INNER(FUNC_NAME, sycl::half)                                    \
  BINARY_CHECK_INNER(FUNC_NAME, double)

#define TRINARY_CHECK_INNER(FUNC_NAME, ...)                                    \
  static_assert(std::is_same_v<                                                \
                decltype(sycl::FUNC_NAME(                                      \
                    std::declval<ImplicitlyConvertibleType<__VA_ARGS__>>(),    \
                    std::declval<ImplicitlyConvertibleType<__VA_ARGS__>>(),    \
                    std::declval<ImplicitlyConvertibleType<__VA_ARGS__>>())),  \
                decltype(sycl::FUNC_NAME(std::declval<__VA_ARGS__>(),          \
                                         std::declval<__VA_ARGS__>(),          \
                                         std::declval<__VA_ARGS__>()))>);

#define GENFLOAT_TRINARY_CHECK(FUNC_NAME)                                      \
  TRINARY_CHECK_INNER(FUNC_NAME, float)                                        \
  TRINARY_CHECK_INNER(FUNC_NAME, double)                                       \
  TRINARY_CHECK_INNER(FUNC_NAME, sycl::half)

void check() {
  GENFLOAT_UNARY_CHECK(acos)
  GENFLOAT_UNARY_CHECK(acosh)
  GENFLOAT_UNARY_CHECK(acospi)
  GENFLOAT_UNARY_CHECK(asin)
  GENFLOAT_UNARY_CHECK(asinh)
  GENFLOAT_UNARY_CHECK(asinpi)
  GENFLOAT_UNARY_CHECK(atan)
  GENFLOAT_UNARY_CHECK(atanh)
  GENFLOAT_UNARY_CHECK(atanpi)
  GENFLOAT_BINARY_CHECK(atan2)
  GENFLOAT_BINARY_CHECK(atan2pi)
  GENFLOAT_UNARY_CHECK(cbrt)
  GENFLOAT_UNARY_CHECK(ceil)
  GENFLOAT_BINARY_CHECK(copysign)
  GENFLOAT_UNARY_CHECK(cos)
  GENFLOAT_UNARY_CHECK(cosh)
  GENFLOAT_UNARY_CHECK(cospi)
  GENFLOAT_UNARY_CHECK(erfc)
  GENFLOAT_UNARY_CHECK(erf)
  GENFLOAT_UNARY_CHECK(exp)
  GENFLOAT_UNARY_CHECK(exp2)
  GENFLOAT_UNARY_CHECK(exp10)
  GENFLOAT_UNARY_CHECK(expm1)
  GENFLOAT_UNARY_CHECK(fabs)
  GENFLOAT_BINARY_CHECK(fdim)
  GENFLOAT_UNARY_CHECK(floor)
  GENFLOAT_TRINARY_CHECK(fma)
  GENFLOAT_BINARY_CHECK(fmax)
  GENFLOAT_BINARY_CHECK(fmin)
  GENFLOAT_BINARY_CHECK(fmod)
  GENFLOAT_BINARY_CHECK(hypot)
  GENFLOAT_UNARY_CHECK(ilogb)
  GENFLOAT_UNARY_CHECK(lgamma)
  GENFLOAT_UNARY_CHECK(log)
  GENFLOAT_UNARY_CHECK(log2)
  GENFLOAT_UNARY_CHECK(log10)
  GENFLOAT_UNARY_CHECK(log1p)
  GENFLOAT_UNARY_CHECK(logb)
  GENFLOAT_TRINARY_CHECK(mad)
  GENFLOAT_BINARY_CHECK(maxmag)
  GENFLOAT_BINARY_CHECK(minmag)
  UGENINT_NAN_UNARY_CHECK(nan)
  GENFLOAT_BINARY_CHECK(nextafter)
  GENFLOAT_BINARY_CHECK(pow)
  GENFLOAT_BINARY_CHECK(powr)
  GENFLOAT_BINARY_CHECK(remainder)
  GENFLOAT_UNARY_CHECK(rint)
  GENFLOAT_UNARY_CHECK(round)
  GENFLOAT_UNARY_CHECK(rsqrt)
  GENFLOAT_UNARY_CHECK(sin)
  GENFLOAT_UNARY_CHECK(sinh)
  GENFLOAT_UNARY_CHECK(sinpi)
  GENFLOAT_UNARY_CHECK(sqrt)
  GENFLOAT_UNARY_CHECK(tan)
  GENFLOAT_UNARY_CHECK(tanh)
  GENFLOAT_UNARY_CHECK(tanpi)
  GENFLOAT_UNARY_CHECK(tgamma)
  GENFLOAT_UNARY_CHECK(trunc)

  FLOAT_UNARY_CHECK(native::cos)
  FLOAT_BINARY_CHECK(native::divide)
  FLOAT_UNARY_CHECK(native::exp)
  FLOAT_UNARY_CHECK(native::exp2)
  FLOAT_UNARY_CHECK(native::exp10)
  FLOAT_UNARY_CHECK(native::log)
  FLOAT_UNARY_CHECK(native::log2)
  FLOAT_UNARY_CHECK(native::log10)
  FLOAT_BINARY_CHECK(native::powr)
  FLOAT_UNARY_CHECK(native::recip)
  FLOAT_UNARY_CHECK(native::rsqrt)
  FLOAT_UNARY_CHECK(native::sin)
  FLOAT_UNARY_CHECK(native::sqrt)
  FLOAT_UNARY_CHECK(native::tan)

  FLOAT_UNARY_CHECK(half_precision::cos)
  FLOAT_BINARY_CHECK(half_precision::divide)
  FLOAT_UNARY_CHECK(half_precision::exp)
  FLOAT_UNARY_CHECK(half_precision::exp2)
  FLOAT_UNARY_CHECK(half_precision::exp10)
  FLOAT_UNARY_CHECK(half_precision::log)
  FLOAT_UNARY_CHECK(half_precision::log2)
  FLOAT_UNARY_CHECK(half_precision::log10)
  FLOAT_BINARY_CHECK(half_precision::powr)
  FLOAT_UNARY_CHECK(half_precision::recip)
  FLOAT_UNARY_CHECK(half_precision::rsqrt)
  FLOAT_UNARY_CHECK(half_precision::sin)
  FLOAT_UNARY_CHECK(half_precision::sqrt)
  FLOAT_UNARY_CHECK(half_precision::tan)

  GENFLOAT_BINARY_CHECK(isequal)
  GENFLOAT_BINARY_CHECK(isnotequal)
  GENFLOAT_BINARY_CHECK(isgreater)
  GENFLOAT_BINARY_CHECK(isgreaterequal)
  GENFLOAT_BINARY_CHECK(isless)
  GENFLOAT_BINARY_CHECK(islessequal)
  GENFLOAT_BINARY_CHECK(islessgreater)
  GENFLOAT_UNARY_CHECK(isfinite)
  GENFLOAT_UNARY_CHECK(isinf)
  GENFLOAT_UNARY_CHECK(isnan)
  GENFLOAT_UNARY_CHECK(isnormal)
  GENFLOAT_BINARY_CHECK(isordered)
  GENFLOAT_BINARY_CHECK(isunordered)
  GENFLOAT_UNARY_CHECK(signbit)
}

int main() {
  check();

  sycl::queue Q;
  Q.single_task([=]() { check(); });
}
