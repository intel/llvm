// RUN: %{build} -o %t.default.out
// RUN: %{run} %t.default.out
// RUN: %{build} -ffast-math -o %t.fast-math.out
// RUN: %{run} %t.fast-math.out
// RUN: %{build} -fgpu-flush-denormals-to-zero -o %t.flush-denormals.out
// RUN: %{run} %t.flush-denormals.out
// RUN: %{build} -foffload-fp32-prec-div -o %t.prec-div.out
// RUN: %{run} %t.prec-div.out
// RUN: %{build} -foffload-fp32-prec-sqrt -o %t.prec-sqrt.out
// RUN: %{run} %t.prec-sqrt.out
// RUN: %{build} -ffp-contract=fast -o %t.fp-contract.out
// RUN: %{run} %t.fp-contract.out
// RUN: %{build} -ffp-contract=on -o %t.fp-contract-on.out
// RUN: %{run} %t.fp-contract-on.out
// RUN: %{build} -ffp-contract=fast-honor-pragmas -o %t.fp-contract-fast-honor-pragmas.out
// RUN: %{run} %t.fp-contract-fast-honor-pragmas.out
// RUN: %{build} -fgpu-approx-transcendentals -o %t.approx-transcendentals.out
// RUN: %{run} %t.approx-transcendentals.out

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <type_traits>
#include <vector>

static std::size_t failure_count = 0;

template <typename T> const std::vector<T> &test_inputs() {
  using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  constexpr Bits sign = Bits{1} << (sizeof(T) * 8 - 1);
  constexpr Bits largest_subnormal =
      sizeof(T) == 4 ? 0x007fffffULL : 0x000fffffffffffffULL;
  static const std::vector<T> inputs{
      T{0.0},
      T{1.0},
      T{1.23456789},
      T{3.14159265},
      T{2.71828183},
      T{-0.0},
      T{-1.0},
      T{-1.23456789},
      T{-3.14159265},
      T{-2.71828183},
      T{1.0e10},
      T{1.0e20},
      T{1.0e30},
      T{-1.0e20},
      sycl::bit_cast<T>(Bits{1}),
      sycl::bit_cast<T>(Bits(sign | Bits{1})),
      sycl::bit_cast<T>(largest_subnormal),
      sycl::bit_cast<T>(Bits(sign | largest_subnormal)),
  };
  return inputs;
}

static sycl::queue &test_queue() {
  static sycl::queue queue;
  return queue;
}

template <typename T> bool outputs_equal(T output, T expected) {
  if constexpr (std::is_integral_v<T>) {
    return output == expected;
  } else {
    using Bits =
        std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    constexpr Bits magnitude_mask =
        sizeof(T) == 4 ? 0x7fffffffULL : 0x7fffffffffffffffULL;
    constexpr Bits infinity =
        sizeof(T) == 4 ? 0x7f800000ULL : 0x7ff0000000000000ULL;
    const Bits actual_bits = sycl::bit_cast<Bits>(output);
    const Bits expected_bits = sycl::bit_cast<Bits>(expected);
    return actual_bits == expected_bits ||
           ((actual_bits & magnitude_mask) == 0 &&
            (expected_bits & magnitude_mask) == 0) ||
           ((actual_bits & magnitude_mask) > infinity &&
            (expected_bits & magnitude_mask) > infinity);
  }
}

template <typename T, typename Fn, typename... Inputs>
void test_math_fn(const std::string &name, T input, const Fn &fn,
                  Inputs... remaining_inputs) {
  const T device_error = [&] {
    auto &q = test_queue();
    T *const error = sycl::malloc_shared<T>(1, q);
    *error = 1000.0;
    q.submit(
        [&](sycl::handler &h) { h.single_task([=]() { *error = fn(); }); });
    q.wait_and_throw();
    const T result = *error;
    sycl::free(error, q);
    return result;
  }();

  const auto report_failure = [&](const char *location) {
    std::cerr << "Test failed: " << name << '<'
              << (std::is_same_v<T, float> ? "float" : "double") << ">("
              << input;
    ((std::cerr << ", " << remaining_inputs), ...);
    std::cerr << ") " << location << " output mismatch\n";
    ++failure_count;
  };

  if (device_error != T{0})
    report_failure("device");
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
    for (T input : test_inputs<T>()) {                                         \
      test_math_fn<T>(#FN, input, [=] {                                        \
        const T output = sycl::FN(input);                                      \
        const T expected = std::FN(input);                                     \
        return !outputs_equal(output, expected);                               \
      });                                                                      \
    }                                                                          \
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
    for (T input : test_inputs<T>()) {                                         \
      test_math_fn<T>(#FN, input, [=] {                                        \
        const int output = sycl::FN(input);                                    \
        const int expected = std::FN(input);                                   \
        return !outputs_equal(output, expected);                               \
      });                                                                      \
    }                                                                          \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(ilogb)
#undef MATH_FN

  // genfloat * genfloat -> genfloat
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    for (T input : test_inputs<T>()) {                                         \
      for (T second : test_inputs<T>()) {                                      \
        test_math_fn<T>(                                                       \
            #FN, input,                                                        \
            [=] {                                                              \
              const T output = sycl::FN(input, second);                        \
              const T expected = std::FN(input, second);                       \
              return !outputs_equal(output, expected);                         \
            },                                                                 \
            second);                                                           \
      }                                                                        \
    }                                                                          \
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
    for (T input : test_inputs<T>()) {                                         \
      for (T second : test_inputs<T>()) {                                      \
        for (T third : test_inputs<T>()) {                                     \
          test_math_fn<T>(                                                     \
              #FN, input,                                                      \
              [=] {                                                            \
                const T output = sycl::FN(input, second, third);               \
                const T expected = std::FN(input, second, third);              \
                return !outputs_equal(output, expected);                       \
              },                                                               \
              second, third);                                                  \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(fma)
#undef MATH_FN

  // genfloat * int -> genfloat
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    for (T input : test_inputs<T>()) {                                         \
      const std::vector<int> exponents{-100, -10, -1, 0, 1, 10, 100};          \
      for (int exponent : exponents) {                                         \
        test_math_fn<T>(                                                       \
            #FN, input,                                                        \
            [=] {                                                              \
              const T output = sycl::FN(input, exponent);                      \
              const T expected = std::FN(input, exponent);                     \
              return !outputs_equal(output, expected);                         \
            },                                                                 \
            exponent);                                                         \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(ldexp)
#undef MATH_FN

  // genfloat -> genfloat * OUTPUT_TYPE (modf: genfloat; frexp: int)
#define MATH_FN(FN, OUTPUT_TYPE)                                               \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    for (T input : test_inputs<T>()) {                                         \
      test_math_fn<T>(#FN, input, [=] {                                        \
        OUTPUT_TYPE output_part{};                                             \
        OUTPUT_TYPE expected_part{};                                           \
        const T output = sycl::FN(input, &output_part);                        \
        const T expected = std::FN(input, &expected_part);                     \
        return !(outputs_equal(output, expected) &&                            \
                 outputs_equal(output_part, expected_part));                   \
      });                                                                      \
    }                                                                          \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(modf, T)
  MATH_FN(frexp, int)
#undef MATH_FN

  // genfloat * genfloat -> genfloat * int
#define MATH_FN(FN)                                                            \
  const auto test_##FN = [](auto sample) {                                     \
    using T = decltype(sample);                                                \
    for (T input : test_inputs<T>()) {                                         \
      for (T second : test_inputs<T>()) {                                      \
        test_math_fn<T>(                                                       \
            #FN, input,                                                        \
            [=] {                                                              \
              int output_quotient = 0;                                         \
              int expected_quotient = 0;                                       \
              const T output = sycl::FN(input, second, &output_quotient);      \
              const T expected = std::FN(input, second, &expected_quotient);   \
              return !(outputs_equal(output, expected) &&                      \
                       (second == T{0} ||                                      \
                        outputs_equal(output_quotient, expected_quotient)));   \
            },                                                                 \
            second);                                                           \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  TEST_TYPES(FN)

  MATH_FN(remquo)
#undef MATH_FN
#undef TEST_TYPES

  return failure_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
