// This test checks that the outputs of std:: math functions and sycl:: math
// built-ins are bitwise equal in device code.

// RUN: %{build} -fno-builtin -o %t.default.out
// RUN: %{run} %t.default.out
// RUN: %{build} -fno-builtin -ffast-math -o %t.fast-math.out
// RUN: %{run} %t.fast-math.out
// RUN: %{build} -fno-builtin -fgpu-flush-denormals-to-zero -o %t.flush-denormals.out
// RUN: %{run} %t.flush-denormals.out
// RUN: %{build} -fno-builtin -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -o %t.precise.out
// RUN: %{run} %t.precise.out
// RUN: %{build} -fno-builtin -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt -o %t.relaxed-precision.out
// RUN: %{run} %t.relaxed-precision.out
// RUN: %{build} -fno-builtin -ffp-contract=off -o %t.fp-contract-off.out
// RUN: %{run} %t.fp-contract-off.out

// UNSUPPORTED: target-native_cpu || cuda
// UNSUPPORTED-INTENDED: The Native CPU and CUDA backends lower std:: math
// functions and sycl:: math built-ins differently than SPIR-V. This test is not
// expected to pass on those backends

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <type_traits>
#include <vector>

template <typename T> struct MathTestCase {
  T first;
  T second;
  T third;
  int integer;
  unsigned argument_count;
  bool has_integer;
};

static std::size_t failure_count = 0;

template <typename T> const std::vector<T> &test_inputs() {
  using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  constexpr Bits sign = Bits{1} << (sizeof(T) * 8 - 1);
  constexpr Bits largest_subnormal =
      sizeof(T) == 4 ? 0x007fffffULL : 0x000fffffffffffffULL;
  static const std::vector<T> inputs{
      T{0.0},
      T{1.0},
      T{3.14159265},
      T{-1.0},
      T{-3.14159265},
      T{1.0e30},
      T{-1.0e20},
      sycl::bit_cast<T>(Bits{1}),
      sycl::bit_cast<T>(Bits(sign | Bits{1})),
      sycl::bit_cast<T>(largest_subnormal),
      sycl::bit_cast<T>(Bits(sign | largest_subnormal)),
  };
  return inputs;
}

template <typename T> std::vector<MathTestCase<T>> unary_cases() {
  std::vector<MathTestCase<T>> cases;
  for (T input : test_inputs<T>())
    cases.push_back({input, T{0}, T{0}, 0, 1, false});
  return cases;
}

template <typename T> std::vector<MathTestCase<T>> binary_cases() {
  std::vector<MathTestCase<T>> cases;
  for (T first : test_inputs<T>())
    for (T second : test_inputs<T>())
      cases.push_back({first, second, T{0}, 0, 2, false});
  return cases;
}

template <typename T> std::vector<MathTestCase<T>> ternary_cases() {
  std::vector<MathTestCase<T>> cases;
  for (T first : test_inputs<T>())
    for (T second : test_inputs<T>())
      for (T third : test_inputs<T>())
        cases.push_back({first, second, third, 0, 3, false});
  return cases;
}

template <typename T> std::vector<MathTestCase<T>> integer_exponent_cases() {
  constexpr int exponents[] = {-100, -10, -1, 0, 1, 10, 100};
  std::vector<MathTestCase<T>> cases;
  for (T input : test_inputs<T>())
    for (int exponent : exponents)
      cases.push_back({input, T{0}, T{0}, exponent, 1, true});
  return cases;
}

static sycl::queue &test_queue() {
  static sycl::queue queue;
  return queue;
}

template <typename T> bool outputs_equal(T output, T expected) {
  using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  const Bits actual_bits = sycl::bit_cast<Bits>(output);
  const Bits expected_bits = sycl::bit_cast<Bits>(expected);
  return actual_bits == expected_bits;
}

template <typename T, typename Fn>
void test_math_fn(const std::string &name,
                  const std::vector<MathTestCase<T>> &host_cases, Fn fn) {
  auto &queue = test_queue();
  const std::size_t case_count = host_cases.size();
  auto *device_cases = sycl::malloc_shared<MathTestCase<T>>(case_count, queue);
  auto *device_failures = sycl::malloc_shared<std::uint8_t>(case_count, queue);
  for (std::size_t index = 0; index < case_count; ++index)
    device_cases[index] = host_cases[index];

  queue.submit([&](sycl::handler &handler) {
    handler.parallel_for(sycl::range<1>(case_count), [=](sycl::id<1> id) {
      const std::size_t index = id[0];
      device_failures[index] =
          static_cast<std::uint8_t>(fn(device_cases[index]));
    });
  });
  queue.wait_and_throw();

  for (std::size_t index = 0; index < case_count; ++index) {
    if (!device_failures[index])
      continue;
    const auto &test_case = host_cases[index];
    std::cerr << "Test failed: " << name << '<'
              << (std::is_same_v<T, float> ? "float" : "double") << ">("
              << test_case.first;
    if (test_case.argument_count > 1)
      std::cerr << ", " << test_case.second;
    if (test_case.argument_count > 2)
      std::cerr << ", " << test_case.third;
    if (test_case.has_integer)
      std::cerr << ", " << test_case.integer;
    std::cerr << ") device output mismatch\n";
    ++failure_count;
  }

  sycl::free(device_cases, queue);
  sycl::free(device_failures, queue);
}

int main() {
  const bool test_fp64 = test_queue().get_device().has(sycl::aspect::fp64);

#define TEST_TYPES(FN)                                                         \
  test_##FN(float{});                                                          \
  if (test_fp64)                                                               \
    test_##FN(double{});

  // genfloat -> genfloat
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    test_math_fn<T>(#FN, unary_cases<T>(), [](MathTestCase<T> args) {          \
      const T output = sycl::FN(args.first);                                   \
      const T expected = std::FN(args.first);                                  \
      return !outputs_equal(output, expected);                                 \
    });                                                                        \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(acos)
  MATH_FN(acosh)
  MATH_FN(asin)
  MATH_FN(asinh)
  MATH_FN(atan)
  MATH_FN(atanh)
  MATH_FN(cbrt)
  MATH_FN(ceil)
  MATH_FN(cos)
  MATH_FN(cosh)
  MATH_FN(erfc)
  MATH_FN(erf)
  MATH_FN(exp)
  MATH_FN(exp2)
  MATH_FN(expm1)
  MATH_FN(fabs)
  MATH_FN(floor)
  MATH_FN(lgamma)
  MATH_FN(log)
  MATH_FN(log2)
  MATH_FN(log10)
  MATH_FN(log1p)
  MATH_FN(logb)
  MATH_FN(rint)
  MATH_FN(round)
  MATH_FN(sin)
  MATH_FN(sinh)
  MATH_FN(sqrt)
  MATH_FN(tan)
  MATH_FN(tanh)
  MATH_FN(tgamma)
  MATH_FN(trunc)
#undef MATH_FN

  // genfloat -> int
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    test_math_fn<T>(#FN, unary_cases<T>(), [](MathTestCase<T> args) {          \
      const int output = sycl::FN(args.first);                                 \
      const int expected = std::FN(args.first);                                \
      return !outputs_equal(output, expected);                                 \
    });                                                                        \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(ilogb)
#undef MATH_FN

  // genfloat * genfloat -> genfloat
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    test_math_fn<T>(#FN, binary_cases<T>(), [](MathTestCase<T> args) {         \
      const T output = sycl::FN(args.first, args.second);                      \
      const T expected = std::FN(args.first, args.second);                     \
      return !outputs_equal(output, expected);                                 \
    });                                                                        \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(atan2)
  MATH_FN(copysign)
  MATH_FN(fdim)
  MATH_FN(fmax)
  MATH_FN(fmin)
  MATH_FN(fmod)
  MATH_FN(hypot)
  MATH_FN(nextafter)
  MATH_FN(pow)
  MATH_FN(remainder)
#undef MATH_FN

  // genfloat * genfloat * genfloat -> genfloat
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    test_math_fn<T>(#FN, ternary_cases<T>(), [](MathTestCase<T> args) {        \
      const T output = sycl::FN(args.first, args.second, args.third);          \
      const T expected = std::FN(args.first, args.second, args.third);         \
      return !outputs_equal(output, expected);                                 \
    });                                                                        \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(fma)
#undef MATH_FN

  // genfloat * int -> genfloat
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    test_math_fn<T>(#FN, integer_exponent_cases<T>(),                          \
                    [](MathTestCase<T> args) {                                 \
                      const T output = sycl::FN(args.first, args.integer);     \
                      const T expected = std::FN(args.first, args.integer);    \
                      return !outputs_equal(output, expected);                 \
                    });                                                        \
  };                                                                           \
  TEST_TYPES(FN)

  // Only test ldexp if double precision is supported since certain <cmath>
  // headers promote float to double in this function.
  if (test_fp64) {
    MATH_FN(ldexp)
  }
#undef MATH_FN

  // genfloat -> genfloat * OUTPUT_TYPE (modf: genfloat; frexp: int)
#define MATH_FN(FN, OUTPUT_TYPE)                                               \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    test_math_fn<T>(#FN, unary_cases<T>(), [](MathTestCase<T> args) {          \
      OUTPUT_TYPE output_part{};                                               \
      OUTPUT_TYPE expected_part{};                                             \
      const T output = sycl::FN(args.first, &output_part);                     \
      const T expected = std::FN(args.first, &expected_part);                  \
      return !(outputs_equal(output, expected) &&                              \
               outputs_equal(output_part, expected_part));                     \
    });                                                                        \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(modf, T)
  // Only test frexp if double precision is supported since certain <cmath>
  // headers promote float to double in this function.
  if (test_fp64) {
    MATH_FN(frexp, int)
  }
#undef MATH_FN

  // genfloat * genfloat -> genfloat * int
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    test_math_fn<T>(#FN, binary_cases<T>(), [](MathTestCase<T> args) {         \
      int output_quotient = 0;                                                 \
      int expected_quotient = 0;                                               \
      const T output = sycl::FN(args.first, args.second, &output_quotient);    \
      const T expected = std::FN(args.first, args.second, &expected_quotient); \
      return !(outputs_equal(output, expected) &&                              \
               (args.second == T{0} ||                                         \
                outputs_equal(output_quotient, expected_quotient)));           \
    });                                                                        \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(remquo)
#undef MATH_FN
#undef TEST_TYPES

  return failure_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
