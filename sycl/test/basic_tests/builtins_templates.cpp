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

#define NONSCALAR_FLOAT_UNARY_CHECK(FUNC_NAME)                                 \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<float, 4>)                         \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<float, 4>)

#define NONSCALAR_GENFLOAT_UNARY_CHECK(FUNC_NAME)                              \
  NONSCALAR_FLOAT_UNARY_CHECK(FUNC_NAME)                                       \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<sycl::half, 4>)                    \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<sycl::half, 4>)                 \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<double, 4>)                        \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<double, 4>)

#define GENFLOAT_UNARY_CHECK(FUNC_NAME)                                        \
  CHECK_INNER(ONE_ARG, FUNC_NAME, float)                                       \
  CHECK_INNER(ONE_ARG, FUNC_NAME, double)                                      \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::half)                                  \
  NONSCALAR_GENFLOAT_UNARY_CHECK(FUNC_NAME)

#define NONSCALAR_UGENINT_NAN_UNARY_CHECK(FUNC_NAME)                           \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<uint32_t, 4>)                      \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<uint16_t, 4>)                      \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<uint64_t, 4>)                      \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<unsigned int, 4>)               \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<unsigned short, 4>)             \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<unsigned long, 4>)

#define SGENINT_UNARY_CHECK(FUNC_NAME)                                         \
  CHECK_INNER(ONE_ARG, FUNC_NAME, signed char)                                 \
  CHECK_INNER(ONE_ARG, FUNC_NAME, short)                                       \
  CHECK_INNER(ONE_ARG, FUNC_NAME, int)                                         \
  CHECK_INNER(ONE_ARG, FUNC_NAME, long long)                                   \
  CHECK_INNER(ONE_ARG, FUNC_NAME, long)                                        \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<int8_t, 4>)                        \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<int16_t, 4>)                       \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<int32_t, 4>)                       \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<int64_t, 4>)                       \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<signed char, 4>)                \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<short, 4>)                      \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<int, 4>)                        \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<long long, 4>)                  \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<long, 4>)

#define UGENINT_UNARY_CHECK(FUNC_NAME)                                         \
  CHECK_INNER(ONE_ARG, FUNC_NAME, unsigned char)                               \
  CHECK_INNER(ONE_ARG, FUNC_NAME, unsigned short)                              \
  CHECK_INNER(ONE_ARG, FUNC_NAME, unsigned int)                                \
  CHECK_INNER(ONE_ARG, FUNC_NAME, unsigned long long)                          \
  CHECK_INNER(ONE_ARG, FUNC_NAME, unsigned long)                               \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::vec<uint8_t, 4>)                       \
  CHECK_INNER(ONE_ARG, FUNC_NAME, sycl::marray<unsigned int, 4>)               \
  NONSCALAR_UGENINT_NAN_UNARY_CHECK(FUNC_NAME)

#define GENINT_UNARY_CHECK(FUNC_NAME)                                          \
  CHECK_INNER(ONE_ARG, FUNC_NAME, char)                                        \
  SGENINT_UNARY_CHECK(FUNC_NAME)                                               \
  UGENINT_UNARY_CHECK(FUNC_NAME)

#define NONSCALAR_FLOAT_BINARY_CHECK(FUNC_NAME)                                \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<float, 4>)                        \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<float, 4>)

#define NONSCALAR_GENFLOAT_BINARY_CHECK(FUNC_NAME)                             \
  NONSCALAR_FLOAT_BINARY_CHECK(FUNC_NAME)                                      \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<sycl::half, 4>)                   \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<sycl::half, 4>)                \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<double, 4>)                       \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<double, 4>)

#define GENFLOAT_BINARY_CHECK(FUNC_NAME)                                       \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, float)                                      \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, double)                                     \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::half)                                 \
  NONSCALAR_GENFLOAT_BINARY_CHECK(FUNC_NAME)

#define SGENINT_BINARY_CHECK(FUNC_NAME)                                        \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, signed char)                                \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, short)                                      \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, int)                                        \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, long long)                                  \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, long)                                       \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<int8_t, 4>)                       \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<int16_t, 4>)                      \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<int32_t, 4>)                      \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<int64_t, 4>)                      \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<signed char, 4>)               \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<short, 4>)                     \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<int, 4>)                       \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<long long, 4>)                 \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<long, 4>)

#define UGENINT_BINARY_CHECK(FUNC_NAME)                                        \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, unsigned char)                              \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, unsigned short)                             \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, unsigned int)                               \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, unsigned long long)                         \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, unsigned long)                              \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<uint8_t, 4>)                      \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<uint16_t, 4>)                     \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<uint32_t, 4>)                     \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::vec<uint64_t, 4>)                     \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<unsigned char, 4>)             \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<unsigned short, 4>)            \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<unsigned int, 4>)              \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<unsigned long long, 4>)        \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, sycl::marray<unsigned long, 4>)

#define GENINT_BINARY_CHECK(FUNC_NAME)                                         \
  CHECK_INNER(TWO_ARGS, FUNC_NAME, char)                                       \
  SGENINT_BINARY_CHECK(FUNC_NAME)                                              \
  UGENINT_BINARY_CHECK(FUNC_NAME)

#define GENTYPE_TRINARY_CHECK(FUNC_NAME)                                       \
  GENFLOAT_TRINARY_CHECK(FUNC_NAME)                                            \
  GENINT_TRINARY_CHECK(FUNC_NAME)

#define NONSCALAR_GENFLOAT_TRINARY_CHECK(FUNC_NAME)                            \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<float, 4>)                      \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<double, 4>)                     \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<sycl::half, 4>)                 \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<float, 4>)                   \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<double, 4>)                  \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<sycl::half, 4>)

#define NONSCALAR_SGENINT_TRINARY_CHECK(FUNC_NAME)                             \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<int8_t, 4>)                     \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<int16_t, 4>)                    \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<int32_t, 4>)                    \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<int64_t, 4>)                    \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<signed char, 4>)             \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<short, 4>)                   \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<int, 4>)                     \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<long long, 4>)               \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<long, 4>)

#define NONSCALAR_UGENINT_TRINARY_CHECK(FUNC_NAME)                             \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<uint8_t, 4>)                    \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<uint16_t, 4>)                   \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<uint32_t, 4>)                   \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::vec<uint64_t, 4>)                   \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<unsigned char, 4>)           \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<unsigned short, 4>)          \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<unsigned int, 4>)            \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<unsigned long long, 4>)      \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::marray<unsigned long, 4>)

#define GENFLOAT_TRINARY_CHECK(FUNC_NAME)                                      \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, float)                                    \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, double)                                   \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, sycl::half)                               \
  NONSCALAR_GENFLOAT_TRINARY_CHECK(FUNC_NAME)

#define SGENINT_TRINARY_CHECK(FUNC_NAME)                                       \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, signed char)                              \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, short)                                    \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, int)                                      \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, long long)                                \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, long)                                     \
  NONSCALAR_SGENINT_TRINARY_CHECK(FUNC_NAME)

#define UGENINT_TRINARY_CHECK(FUNC_NAME)                                       \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, unsigned char)                            \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, unsigned short)                           \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, unsigned int)                             \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, unsigned long long)                       \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, unsigned long)                            \
  NONSCALAR_UGENINT_TRINARY_CHECK(FUNC_NAME)

#define GENINT_TRINARY_CHECK(FUNC_NAME)                                        \
  CHECK_INNER(THREE_ARGS, FUNC_NAME, char)                                     \
  SGENINT_TRINARY_CHECK(FUNC_NAME)                                             \
  UGENINT_TRINARY_CHECK(FUNC_NAME)

#define GENTYPE_TRINARY_CHECK(FUNC_NAME)                                       \
  GENFLOAT_TRINARY_CHECK(FUNC_NAME)                                            \
  GENINT_TRINARY_CHECK(FUNC_NAME)

void check() {
  NONSCALAR_GENFLOAT_UNARY_CHECK(acos)
  NONSCALAR_GENFLOAT_UNARY_CHECK(acosh)
  NONSCALAR_GENFLOAT_UNARY_CHECK(acospi)
  NONSCALAR_GENFLOAT_UNARY_CHECK(asin)
  NONSCALAR_GENFLOAT_UNARY_CHECK(asinh)
  NONSCALAR_GENFLOAT_UNARY_CHECK(asinpi)
  NONSCALAR_GENFLOAT_UNARY_CHECK(atan)
  NONSCALAR_GENFLOAT_UNARY_CHECK(atanh)
  NONSCALAR_GENFLOAT_UNARY_CHECK(atanpi)
  NONSCALAR_GENFLOAT_BINARY_CHECK(atan2)
  NONSCALAR_GENFLOAT_BINARY_CHECK(atan2pi)
  NONSCALAR_GENFLOAT_UNARY_CHECK(cbrt)
  NONSCALAR_GENFLOAT_UNARY_CHECK(ceil)
  NONSCALAR_GENFLOAT_BINARY_CHECK(copysign)
  NONSCALAR_GENFLOAT_UNARY_CHECK(cos)
  NONSCALAR_GENFLOAT_UNARY_CHECK(cosh)
  NONSCALAR_GENFLOAT_UNARY_CHECK(cospi)
  NONSCALAR_GENFLOAT_UNARY_CHECK(erfc)
  NONSCALAR_GENFLOAT_UNARY_CHECK(erf)
  NONSCALAR_GENFLOAT_UNARY_CHECK(exp)
  NONSCALAR_GENFLOAT_UNARY_CHECK(exp2)
  NONSCALAR_GENFLOAT_UNARY_CHECK(exp10)
  NONSCALAR_GENFLOAT_UNARY_CHECK(expm1)
  NONSCALAR_GENFLOAT_UNARY_CHECK(fabs)
  NONSCALAR_GENFLOAT_BINARY_CHECK(fdim)
  NONSCALAR_GENFLOAT_UNARY_CHECK(floor)
  NONSCALAR_GENFLOAT_TRINARY_CHECK(fma)
  NONSCALAR_GENFLOAT_BINARY_CHECK(fmax)
  NONSCALAR_GENFLOAT_BINARY_CHECK(fmin)
  NONSCALAR_GENFLOAT_BINARY_CHECK(fmod)
  NONSCALAR_GENFLOAT_BINARY_CHECK(hypot)
  NONSCALAR_GENFLOAT_UNARY_CHECK(ilogb)
  NONSCALAR_GENFLOAT_UNARY_CHECK(lgamma)
  NONSCALAR_GENFLOAT_UNARY_CHECK(log)
  NONSCALAR_GENFLOAT_UNARY_CHECK(log2)
  NONSCALAR_GENFLOAT_UNARY_CHECK(log10)
  NONSCALAR_GENFLOAT_UNARY_CHECK(log1p)
  NONSCALAR_GENFLOAT_UNARY_CHECK(logb)
  NONSCALAR_GENFLOAT_TRINARY_CHECK(mad)
  NONSCALAR_GENFLOAT_BINARY_CHECK(maxmag)
  NONSCALAR_GENFLOAT_BINARY_CHECK(minmag)
  NONSCALAR_UGENINT_NAN_UNARY_CHECK(nan)
  NONSCALAR_GENFLOAT_BINARY_CHECK(nextafter)
  NONSCALAR_GENFLOAT_BINARY_CHECK(pow)
  NONSCALAR_GENFLOAT_BINARY_CHECK(powr)
  NONSCALAR_GENFLOAT_BINARY_CHECK(remainder)
  NONSCALAR_GENFLOAT_UNARY_CHECK(rint)
  NONSCALAR_GENFLOAT_UNARY_CHECK(round)
  NONSCALAR_GENFLOAT_UNARY_CHECK(rsqrt)
  NONSCALAR_GENFLOAT_UNARY_CHECK(sin)
  NONSCALAR_GENFLOAT_UNARY_CHECK(sinh)
  NONSCALAR_GENFLOAT_UNARY_CHECK(sinpi)
  NONSCALAR_GENFLOAT_UNARY_CHECK(sqrt)
  NONSCALAR_GENFLOAT_UNARY_CHECK(tan)
  NONSCALAR_GENFLOAT_UNARY_CHECK(tanh)
  NONSCALAR_GENFLOAT_UNARY_CHECK(tanpi)
  NONSCALAR_GENFLOAT_UNARY_CHECK(tgamma)
  NONSCALAR_GENFLOAT_UNARY_CHECK(trunc)

  NONSCALAR_FLOAT_UNARY_CHECK(native::cos)
  NONSCALAR_FLOAT_BINARY_CHECK(native::divide)
  NONSCALAR_FLOAT_UNARY_CHECK(native::exp)
  NONSCALAR_FLOAT_UNARY_CHECK(native::exp2)
  NONSCALAR_FLOAT_UNARY_CHECK(native::exp10)
  NONSCALAR_FLOAT_UNARY_CHECK(native::log)
  NONSCALAR_FLOAT_UNARY_CHECK(native::log2)
  NONSCALAR_FLOAT_UNARY_CHECK(native::log10)
  NONSCALAR_FLOAT_BINARY_CHECK(native::powr)
  NONSCALAR_FLOAT_UNARY_CHECK(native::recip)
  NONSCALAR_FLOAT_UNARY_CHECK(native::rsqrt)
  NONSCALAR_FLOAT_UNARY_CHECK(native::sin)
  NONSCALAR_FLOAT_UNARY_CHECK(native::sqrt)
  NONSCALAR_FLOAT_UNARY_CHECK(native::tan)

  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::cos)
  NONSCALAR_FLOAT_BINARY_CHECK(half_precision::divide)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::exp)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::exp2)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::exp10)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::log)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::log2)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::log10)
  NONSCALAR_FLOAT_BINARY_CHECK(half_precision::powr)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::recip)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::rsqrt)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::sin)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::sqrt)
  NONSCALAR_FLOAT_UNARY_CHECK(half_precision::tan)

  GENINT_UNARY_CHECK(abs)
  GENINT_BINARY_CHECK(abs_diff)
  GENINT_BINARY_CHECK(add_sat)
  GENINT_BINARY_CHECK(hadd)
  GENINT_BINARY_CHECK(rhadd)
  GENINT_TRINARY_CHECK(clamp)
  GENINT_UNARY_CHECK(clz)
  GENINT_UNARY_CHECK(ctz)
  GENINT_TRINARY_CHECK(mad_hi)
  GENINT_TRINARY_CHECK(mad_sat)
  GENINT_BINARY_CHECK(max)
  GENINT_BINARY_CHECK(min)
  GENINT_BINARY_CHECK(mul_hi)
  GENINT_BINARY_CHECK(rotate)
  GENINT_BINARY_CHECK(sub_sat)
  GENINT_UNARY_CHECK(popcount)

  GENFLOAT_TRINARY_CHECK(clamp)
  GENFLOAT_UNARY_CHECK(degrees)
  GENFLOAT_BINARY_CHECK(max)
  GENFLOAT_BINARY_CHECK(min)
  GENFLOAT_TRINARY_CHECK(mix)
  GENFLOAT_UNARY_CHECK(radians)
  GENFLOAT_BINARY_CHECK(step)
  GENFLOAT_TRINARY_CHECK(smoothstep)
  GENFLOAT_UNARY_CHECK(sign)

  NONSCALAR_GENFLOAT_BINARY_CHECK(cross)
  GENFLOAT_BINARY_CHECK(dot)
  GENFLOAT_BINARY_CHECK(distance)
  GENFLOAT_UNARY_CHECK(length)
  GENFLOAT_UNARY_CHECK(normalize)
  GENFLOAT_BINARY_CHECK(fast_distance)
  GENFLOAT_UNARY_CHECK(fast_length)
  GENFLOAT_UNARY_CHECK(fast_normalize)

  NONSCALAR_GENFLOAT_BINARY_CHECK(isequal)
  NONSCALAR_GENFLOAT_BINARY_CHECK(isnotequal)
  NONSCALAR_GENFLOAT_BINARY_CHECK(isgreater)
  NONSCALAR_GENFLOAT_BINARY_CHECK(isgreaterequal)
  NONSCALAR_GENFLOAT_BINARY_CHECK(isless)
  NONSCALAR_GENFLOAT_BINARY_CHECK(islessequal)
  NONSCALAR_GENFLOAT_BINARY_CHECK(islessgreater)
  NONSCALAR_GENFLOAT_UNARY_CHECK(isfinite)
  NONSCALAR_GENFLOAT_UNARY_CHECK(isinf)
  NONSCALAR_GENFLOAT_UNARY_CHECK(isnan)
  NONSCALAR_GENFLOAT_UNARY_CHECK(isnormal)
  NONSCALAR_GENFLOAT_BINARY_CHECK(isordered)
  NONSCALAR_GENFLOAT_BINARY_CHECK(isunordered)
  NONSCALAR_GENFLOAT_UNARY_CHECK(signbit)
  SGENINT_UNARY_CHECK(any)
  SGENINT_UNARY_CHECK(all)
  GENTYPE_TRINARY_CHECK(bitselect)
}

int main() {
  check();

  sycl::queue Q;
  Q.single_task([=]() { check(); });
}
